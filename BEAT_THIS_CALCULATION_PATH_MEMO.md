# Beat This 계산 경로 메모

이 문서는 `beat_this` 프로젝트의 실제 계산 경로를 코드 기준으로 정리한 메모다.  
핵심은 이 프로젝트의 모델 직접 출력이 최종 beat 목록이 아니라 `framewise beat/downbeat logits`라는 점이다.

## 1. 전체 경로 요약

실사용 추론 경로는 아래 순서로 진행된다.

1. 오디오 로드
2. mono 변환 및 `22050 Hz` 리샘플링
3. `128-bin log-mel spectrogram` 생성
4. 긴 입력은 `1500` 프레임 단위로 분할
5. `BeatThis` 모델 추론
6. chunk 출력 병합
7. 후처리로 beat/downbeat 시점 추출
8. 필요하면 `.beats` 파일 저장

코드 엔트리 포인트는 대체로 아래 순서다.

- `beat_this/cli.py`
- `beat_this/inference.py`
- `beat_this/model/beat_tracker.py`
- `beat_this/model/postprocessor.py`
- `beat_this/utils.py`

## 2. 추론 전처리

### 2.1 오디오 로딩

`beat_this/preprocessing.py`의 `load_audio()`가 입력 오디오를 읽는다.

- 1차: `torchaudio.load()`
- 실패 시 2차: `soundfile.read()`
- 실패 시 3차: `madmom.io.load_audio_file()`

즉, 입력 포맷 호환성은 여러 backend fallback으로 확보한다.

### 2.2 채널 처리

`beat_this/inference.py`의 `Audio2Frames.signal2spect()`에서:

- 입력이 2차원이면 stereo로 보고 채널 평균을 내서 mono로 변환
- 입력이 1차원이 아니면 에러

따라서 모델 입력은 항상 mono waveform 기준이다.

### 2.3 리샘플링

샘플레이트가 `22050 Hz`가 아니면 `soxr.resample()`로 `22050 Hz`로 변환한다.

이 샘플레이트는 프로젝트 전체에서 사실상 표준 입력 샘플레이트다.

### 2.4 스펙트로그램 생성

`beat_this/preprocessing.py`의 `LogMelSpect`가 waveform을 log-mel spectrogram으로 바꾼다.

주요 파라미터는 다음과 같다.

- `sample_rate = 22050`
- `n_fft = 1024`
- `hop_length = 441`
- `f_min = 30`
- `f_max = 11000`
- `n_mels = 128`
- `power = 1`
- `mel_scale = "slaney"`
- `normalized = "frame_length"`
- `log_multiplier = 1000`

출력 수식은 아래와 같다.

```text
log_mel = log1p(1000 * MelSpectrogram(x))
```

출력 텐서 shape은 `(F, 128)`이다.

- `F`: 시간 프레임 수
- `128`: mel bin 수

또한 `22050 / 441 = 50`이므로 모델 시간축 해상도는 `50 fps`다.  
즉 한 프레임은 `20 ms`다.

## 3. 모델 입력과 분할 처리

### 3.1 모델 입력 형태

모델 본체 `BeatThis`는 spectrogram을 입력으로 받는다.

- 단일 chunk 입력 shape: `(1, T, 128)`
- 여기서 `1`은 batch size
- `T`는 chunk 길이

### 3.2 긴 곡 분할

`beat_this/inference.py`의 `split_predict_aggregate()`에서 긴 곡은 아래 기준으로 분할된다.

- `chunk_size = 1500` 프레임
- `border_size = 6` 프레임
- `overlap_mode = "keep_first"`

`1500` 프레임은 `50 fps` 기준 약 `30초`다.

분할 방식의 목적은 다음과 같다.

- 훈련 시 사용한 길이에 맞춰 추론
- 긴 곡도 메모리 폭증 없이 처리
- 경계 부근 예측 품질 저하를 완화

첫 chunk와 마지막 chunk는 필요하면 zero padding된다.

## 4. 모델 구조

모델 정의는 `beat_this/model/beat_tracker.py`에 있다.

전체 구조는 다음과 같다.

1. CNN stem
2. 3개의 frontend block
3. 선형 projection
4. RoFormer transformer blocks
5. 출력 head

### 4.1 Stem

입력 `(B, T, 128)`에 대해:

- 시간/주파수 축 재배열
- `BatchNorm1d`
- channel 추가
- `Conv2d(1 -> 32, kernel=(4, 3), stride=(4, 1), padding=(0, 1))`
- `BatchNorm2d`
- `GELU`

이 단계에서 주파수 축은 stride `4`로 줄어든다.

### 4.2 Frontend blocks

frontend block은 총 3개다.

각 block은 대체로 아래 순서다.

1. optional `PartialFTTransformer`
2. `Conv2d(in_dim -> out_dim, kernel=(2, 3), stride=(2, 1), padding=(0, 1))`
3. `BatchNorm2d`
4. `GELU`

기본 설정에서는 partial transformer가 켜져 있다.

이 3개 block을 지나며:

- 채널: `32 -> 64 -> 128 -> 256`
- 주파수 bin: `32 -> 16 -> 8 -> 4`

즉 최종 frontend 출력은 대략 `(B, 256, 4, T)` 구조가 된다.

### 4.3 PartialFTTransformer

이 블록은 frontend 안에서:

- 한 번은 주파수 축 방향 self-attention
- 한 번은 시간 축 방향 self-attention

을 수행한다.

즉 초기 단계에서 시간 축과 주파수 축 관계를 모두 부분적으로 섞어준다.

### 4.4 Transformer 본체

frontend 출력은 `(B, T, C*F)`로 펼친 뒤 선형 projection으로 transformer 차원으로 보낸다.

기본값:

- `transformer_dim = 512`
- `n_layers = 6`
- `head_dim = 32`
- head 수는 `512 / 32 = 16`

메인 시퀀스 모델은 `beat_this/model/roformer.py`의 RoFormer 기반 transformer다.

각 layer는:

- rotary positional embedding이 적용된 self-attention
- feed-forward
- residual connection

으로 구성된다.

## 5. 모델의 직접 출력 형태

모델 직접 출력은 최종 beat 시점 배열이 아니다.

`BeatThis.forward()`의 반환값은 아래 딕셔너리다.

```python
{
    "beat": Tensor,
    "downbeat": Tensor,
}
```

shape은 다음과 같다.

- batched일 때: `(B, T)`
- unbatched일 때: `(T,)`

여기서 값은 확률이 아니라 `logit`이다.  
즉 sigmoid를 아직 통과하지 않은 raw activation이다.

### 5.1 Head 동작

기본 head는 `SumHead`다.

마지막 선형층이 `2`채널을 만든 뒤:

- 하나는 `downbeat_raw`
- 하나는 `beat_raw`

처럼 분리하고, 최종 beat logit은 아래처럼 다시 만든다.

```text
beat = beat_raw + downbeat_raw
downbeat = downbeat_raw
```

이 설계 때문에 beat 채널은 "모든 beat", downbeat 채널은 "그중 downbeat"라는 의미를 갖는다.

## 6. chunk 출력 병합

긴 곡을 여러 chunk로 나눠 추론한 뒤 `aggregate_prediction()`으로 합친다.

병합 시 핵심 규칙은 다음과 같다.

- 각 chunk 출력 양 끝 `6` 프레임은 버림
- 남은 중앙부만 전체 시퀀스에 복원
- 겹치는 부분은 기본적으로 먼저 나온 chunk 출력을 유지

버리는 이유는 훈련 손실이 경계에서 불안정하기 때문이다.

## 7. 후처리

후처리는 `beat_this/model/postprocessor.py`에 있다.  
두 가지 모드가 있다.

- `minimal`
- `dbn`

기본 추론은 `minimal`이다.

### 7.1 Minimal 후처리

입력은 framewise beat/downbeat logits다.

처리 순서는 다음과 같다.

1. padding 위치를 `-1000`으로 마스킹
2. `7`프레임 max-pooling으로 local peak만 남김
3. `logit > 0`인 peak만 유지
4. 인접 peak를 평균 위치로 합침
5. frame index를 seconds로 변환
6. 각 downbeat를 가장 가까운 beat 시점으로 snap
7. 중복된 downbeat 제거

여기서 `7`프레임 창은 `50 fps` 기준 약 `140 ms`다.  
또 `logit > 0`은 sigmoid 기준 `0.5` 초과와 같다.

최종 출력은 아래 두 배열이다.

- `beats: np.ndarray`
- `downbeats: np.ndarray`

두 배열의 원소는 모두 "초 단위 시점"이다.

### 7.2 DBN 후처리

`dbn` 옵션을 켜면 madmom의 `DBNDownBeatTrackingProcessor`를 사용한다.

처리 순서는 다음과 같다.

1. logits에 sigmoid 적용
2. `0`과 `1`이 되지 않도록 epsilon clipping
3. 다음 2차원 activation을 구성

```text
combined[:, 0] = beat_prob - downbeat_prob
combined[:, 1] = downbeat_prob
```

4. DBN 실행
5. DBN 결과에서 beat 시점과 downbeat 시점 추출

즉 DBN 모드는 프레임별 확률을 리듬 상태 전이 모델로 해석해 더 구조적인 출력을 만든다.

## 8. 파일 출력 형태

CLI 또는 `File2File`은 최종 결과를 `.beats` 파일로 저장할 수 있다.

저장 형식은 아래와 같다.

```text
time_in_seconds<TAB>beat_number
```

예:

```text
0.48    1
0.97    2
1.46    3
1.95    4
```

여기서 beat number는 `infer_beat_numbers()`가 생성한다.

- downbeat는 `1`
- 그 사이 beat는 2, 3, 4, ... 로 증가

전제는 모든 downbeat가 beat 배열에도 포함되어 있어야 한다는 점이다.

## 9. `--activations` 옵션의 raw output 저장 형식

CLI에서 `--activations`를 켜면 후처리 전 raw logits를 `.npy` 파일로 저장한다.

저장 내용은 아래와 같다.

```python
np.vstack([beat_logits, downbeat_logits])
```

즉 shape은 `(2, T)`다.

- 첫 번째 행: beat logits
- 두 번째 행: downbeat logits

이 파일이 모델의 직접 출력에 가장 가까운 저장 형태다.

## 10. 학습 경로에서의 전처리

실제 배포 추론과 별개로, 학습 쪽에서는 사전 계산된 spectrogram과 annotation을 사용한다.

### 10.1 원본 오디오 전처리

`launch_scripts/preprocess_audio.py`가 원본 오디오에서 학습용 데이터를 만든다.

순서는 다음과 같다.

1. 원본 오디오 로드
2. mono 변환
3. `22050 Hz`로 저장
4. augmentation용으로 필요하면 `44100 Hz` 기준 처리
5. precomputed pitch shift 생성
6. precomputed time stretch 생성
7. 각 오디오에서 spectrogram `.npy` 생성
8. dataset별 `.npz` bundle 생성

### 10.2 학습 augmentation

`beat_this/dataset/augment.py` 기준으로 학습 시 augmentation은 세 종류다.

- `pitch`: precomputed pitch-shifted spectrogram 선택
- `tempo`: precomputed time-stretched spectrogram 선택, annotation 시간도 함께 scale
- `mask`: spectrogram 일부 구간을 zeroing 또는 permutation

이 중 `pitch`, `tempo`는 spectrogram 파일 선택 단계에서 적용되고, `mask`는 로딩 후 온라인으로 적용된다.

## 11. 학습용 라벨 형식

`beat_this/dataset/dataset.py`의 `prepare_annotations()`가 라벨을 만든다.

원래 `.beats` annotation은 초 단위 시점 정보다. 이를 다음처럼 바꾼다.

1. beat/downbeat 시점을 `fps=50` 기준 frame index로 양자화
2. 현재 excerpt 구간에 맞게 잘라냄
3. beat와 downbeat를 각각 framewise boolean sequence로 변환

즉 학습용 타깃은 다음 두 시퀀스다.

- `truth_beat`: shape `(T,)`, bool 계열
- `truth_downbeat`: shape `(T,)`, bool 계열

또한 평가용으로는 양자화 전 원본 초 단위 annotation도 별도로 유지한다.

## 12. 학습 손실과 추론 경계 버림의 연결

기본 손실은 `ShiftTolerantBCELoss`다.

이 손실은:

- positive target 주변에서 작은 시간 오차를 허용하기 위해
- prediction에 temporal max-pooling을 적용한다

기본 tolerance는 `3`프레임이다.

따라서 경계부는 학습상 gradient 해석이 애매해지고,  
추론 시 `border_size = 2 * tolerance = 6` 프레임을 버리는 설계가 여기에 대응한다.

즉 추론의 border discard는 임의 선택이 아니라 손실 정의와 직접 연결되어 있다.

## 13. 평가 경로

평가 시에는 full piece 단위로:

1. spectrogram 전체를 읽고
2. chunk 추론 및 병합
3. 후처리로 beat/downbeat 시점 생성
4. `mir_eval`로 metric 계산

을 수행한다.

validation에서는 주로:

- `F-measure`
- `Cemgil`

을 계산하고, test에서는 추가로:

- `CMLt`
- `AMLt`

도 계산한다.

## 14. 최종 한 줄 요약

이 프로젝트의 계산 경로는 아래 한 줄로 요약할 수 있다.

```text
audio -> mono/22050Hz -> 128-bin log-mel spectrogram -> BeatThis -> framewise beat/downbeat logits -> peak/DBN postprocessing -> beat/downbeat times -> .beats file
```

모델의 직접 output은 끝까지 `framewise logits`이며,  
사람이 보는 beat 목록은 항상 그 위에 후처리를 거친 결과다.
