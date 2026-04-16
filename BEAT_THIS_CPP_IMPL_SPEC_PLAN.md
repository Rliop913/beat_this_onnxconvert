# Beat This C++ 컨버팅 스펙 시트

이 문서는 `beat_this`의 실제 추론 경로를 C++로 옮기기 위한 standalone 스펙이다.

핵심 목표는 현재 Python 추론의 기본 경로를 그대로 재현하는 것이다.

```text
float32 waveform + sample_rate (+ optional multi-channel)
-> channel normalization
-> resample to 22050 Hz
-> cpp_root-based STFT/log-mel frontend
-> spectrogram split
-> BeatThis chunk inference
-> chunk logits aggregation
-> minimal postprocess
-> beat/downbeat times
```

## 1. 목표

현재 Python 코드 기준 아래 두 인터페이스에 대응하는 C++ 추론 라이브러리를 만든다.

- framewise beat/downbeat logits 추출
- 최종 beat/downbeat 시점 추출

즉 이 스펙의 결과물은 `beat_this.inference.Audio2Frames`와 기본 `Audio2Beats` 경로에 대응한다.
파일 경로 기반 편의 인터페이스는 `miniaudio.h` 기반 adapter로 제공할 수 있다.
다만 이 문서의 핵심 parity 대상은 여전히 waveform 입력 경로다.

## 2. 범위

### 2.1 포함 범위

- 추론 전용 구현
- 전체 곡 기준 배치형 처리
- waveform 입력의 채널 정규화와 내부 리샘플링
- `miniaudio.h` 기반 파일 디코딩과 PCM 변환 adapter
- `cpp_root` 안의 STFT/window/mel 알고리즘 재사용
- raw framewise logits 추출 API
- minimal postprocess 기반 beat/downbeat 시점 추출 API
- ONNX Runtime 기반 모델 실행
- Python reference와의 수치 parity 검증

### 2.2 제외 범위

- 학습 경로
- 데이터셋 생성 및 전처리 스크립트
- augmentation
- metric 계산
- 대체 후처리 경로
- 상태 유지형 증분 처리 API

## 3. 구현 경계

새 구현의 계산 경계는 아래처럼 고정한다.

```text
encoded audio file or waveform
-> optional miniaudio decode
-> channel normalization
-> resample to 22050 Hz
-> cpp_root STFT/log-mel frontend
-> spectrogram (T x 128)
-> C++ split orchestration
-> ONNX model runner on each chunk
-> aggregated logits (T x 2)
-> C++ minimal postprocessor
-> beat/downbeat arrays
```

즉 다음 세 단계는 C++ 쪽에서 명시적으로 유지한다.

- spectrogram 기준 split
- chunk logits aggregation
- minimal postprocess

`waveform -> final output`을 하나의 ONNX graph로 단순화하지 않는다.
파일 입력이 들어오면 decode 단계는 `miniaudio.h`가 담당한다.
기본 shipping 경로의 spectrogram frontend는 `cpp_root` 안의 STFT/mel 구현을 재사용하는 native C++ 코드로 둔다.
preprocess ONNX artifact는 parity 검증이나 fallback 실험이 필요한 경우에만 선택적으로 둔다.

## 4. 입력 계약

공개 API의 입력 계약은 아래처럼 고정한다.

- 입력은 `float32` PCM waveform buffer다.
- public waveform API는 `sample_rate`와 채널 수를 명시적으로 받는다.
- mono 입력과 multi-channel 입력을 모두 허용한다.
- multi-channel 입력은 Python `Audio2Frames.signal2spect()`와 같은 의미로 채널 평균을 내서 mono로 변환한다.
- multi-channel 버퍼 layout은 공개 API에서 명시해야 한다.
- 입력 sample rate가 `22050`가 아니면 내부적으로 `22050 Hz`로 리샘플링한다.
- 파일 경로 입력은 `miniaudio` 기반 편의 API에서 직접 다룰 수 있다.
- dtype 변환은 코어 API 범위 밖이다.
- 입력 버퍼는 non-empty여야 한다.
- `NaN`, `Inf`가 포함된 입력은 에러다.

waveform API와 별도로 아래 편의 adapter를 둘 수 있다.

- `miniaudio.h`를 사용해 파일을 decode한다.
- decode 출력은 `float32` interleaved PCM으로 정규화한다.
- channel averaging은 현재 Python 의미와 동일하게 C++에서 명시적으로 수행한다.
- resample은 `miniaudio` resampler 또는 `ma_data_converter`를 사용해 `22050 Hz`로 수행한다.

공개 설정값으로 노출하지 않는 내부 상수:

- `sample_rate = 22050`
- `n_fft = 1024`
- `hop_length = 441`
- `fps = 50`
- `n_mels = 128`
- `f_min = 30`
- `f_max = 11000`
- `chunk_size = 1500`
- `border_size = 6`
- `minimal_peak_window = 7`
- `minimal_peak_threshold_logit = 0.0`
- `deduplicate_width = 1`

## 5. 런타임과 아티팩트

### 5.1 런타임

- 모델 실행 런타임은 `ONNX Runtime C++ API`로 고정한다.
- C++ 코드가 PyTorch를 링크하지는 않는다.
- ONNX export opset과 ORT 버전은 실제 export compatibility를 확인한 뒤 문서화해 고정한다.

### 5.2 아티팩트 구성

기본 shipping build에서 필수인 아티팩트는 아래 1종이다.

1. model artifact

preprocess artifact는 선택 사항이다.
`cpp_root` frontend parity 검증이나 fallback 경로가 필요할 때만 별도로 둔다.

권장 파일명:

- `re-impl/models/beat_this_model_<variant>.onnx`
- `re-impl/models/beat_this_preprocess.onnx` optional

`<variant>`는 Python 체크포인트 shortname과 일치시키는 것을 기본 규칙으로 한다.

예시:

- `final0`
- `final1`
- `final2`
- `small0`
- `small1`
- `small2`
- `hung0`
- `fold0`

### 5.3 optional preprocess artifact 계약

입력:

- `waveform[1, N]`, `float32`
- 입력 waveform은 `AudioNormalizer`를 지난 mono `22050 Hz` 기준으로 해석한다.

출력:

- `spect[T, 128]`, `float32`

의미:

- Python `LogMelSpect`와 동일한 의미의 log-mel spectrogram
- axis `0 = time`
- axis `1 = mel bins`
- 이 artifact는 shipping 필수 조건이 아니다.
- `cpp_root` 기반 C++ frontend와의 parity 확인 또는 fallback 경로가 필요할 때만 사용한다.

### 5.4 model artifact 계약

입력:

- `spect[1, T, 128]`, `float32`

출력:

- `logits[1, T, 2]`, `float32`

출력 layout:

- axis `0 = batch`
- axis `1 = time`
- axis `2 = [beat, downbeat]`

즉 Python의

```python
{"beat": (1, T), "downbeat": (1, T)}
```

출력을 C++에서는 contiguous row-major `1 x T x 2`로 받아서 사용한다.

## 6. C++ 공개 API

```cpp
namespace beat_this {

struct BeatThisConfig {
  std::optional<std::filesystem::path> preprocess_model_path;
  std::filesystem::path inference_model_path;
  bool use_cpp_root_frontend = true;
  bool enable_miniaudio_file_adapter = true;
};

enum class ChannelLayout {
  kInterleavedFrames,
};

struct AudioBufferView {
  std::span<const float> samples;
  int sample_rate = 22050;
  int num_channels = 1;
  ChannelLayout layout = ChannelLayout::kInterleavedFrames;
};

struct FrameLogits {
  int num_frames = 0;
  std::vector<float> beat;
  std::vector<float> downbeat;
};

struct BeatPositions {
  std::vector<double> beats_seconds;
  std::vector<double> downbeats_seconds;
};

struct BeatTrackingResult {
  FrameLogits logits;
  BeatPositions positions;
};

template <typename T>
concept FrontendRunner = requires(
    T& runner,
    std::span<const float> mono_waveform_22050) {
  { runner.ComputeSpectrogram(mono_waveform_22050) } -> std::same_as<std::vector<float>>;
  { runner.num_frames() } -> std::same_as<int>;
};

template <typename T>
concept ChunkModelRunner = requires(
    T& runner,
    std::span<const float> spect_chunk,
    int num_frames,
    int num_bins) {
  { runner.ComputeChunkLogits(spect_chunk, num_frames, num_bins) } -> std::same_as<std::vector<float>>;
};

template <typename T>
concept Postprocessor = requires(
    T& post,
    const FrameLogits& logits) {
  { post.Process(logits) } -> std::same_as<BeatPositions>;
};

class BeatThisCpp {
 public:
  explicit BeatThisCpp(BeatThisConfig config);
  FrameLogits ComputeLogits(AudioBufferView audio);
  BeatPositions Process(AudioBufferView audio);
  BeatTrackingResult ProcessWithLogits(AudioBufferView audio);
  FrameLogits ComputeLogitsFile(const std::filesystem::path& path);
  BeatPositions ProcessFile(const std::filesystem::path& path);
  BeatTrackingResult ProcessFileWithLogits(const std::filesystem::path& path);
  FrameLogits ComputeLogitsNormalized(std::span<const float> mono_waveform_22050);
};

template <
    FrontendRunner TFrontendRunner,
    ChunkModelRunner TChunkModelRunner,
    Postprocessor TPostprocessor>
class BeatThisPipeline {
 public:
  BeatThisPipeline(
      BeatThisConfig config,
      TFrontendRunner frontend_runner,
      TChunkModelRunner chunk_model_runner,
      TPostprocessor postprocessor);

  FrameLogits ComputeLogits(AudioBufferView audio);
  BeatPositions Process(AudioBufferView audio);
};

}
```

기본 사용자용 타입은 `BeatThisCpp` facade 하나로 충분하게 유지한다.

## 7. 내부 모듈 분리

기본 구성요소는 아래처럼 고정한다.

- `AudioNormalizer`
- `MiniaudioFileDecoder`
- `MiniaudioResampler`
- `CppRootStftFrontend`
- `OnnxPreprocessRunner` optional validation/fallback
- `OnnxChunkModelRunner`
- `SpectChunkSplitter`
- `ChunkPredictionAggregator`
- `MinimalPostprocessor`
- `BeatThisCpp` 오케스트레이터

정책 교체 실험이 필요하면 `BeatThisPipeline<T...>`으로 조합한다.

기본 설계에서는 내부 파이프라인에 virtual interface를 두지 않는다.

## 8. 디렉토리 구조

새 구현은 모두 `re-impl/` 아래에 둔다.

권장 디렉토리 구조:

- `re-impl/cpp_beat_this/`
- `re-impl/cpp_beat_this/include/beat_this/api/`
- `re-impl/cpp_beat_this/include/beat_this/config/`
- `re-impl/cpp_beat_this/include/beat_this/frontend/miniaudio/`
- `re-impl/cpp_beat_this/include/beat_this/frontend/cpp_root/`
- `re-impl/cpp_beat_this/include/beat_this/frontend/onnx/`
- `re-impl/cpp_beat_this/include/beat_this/pipeline/split/`
- `re-impl/cpp_beat_this/include/beat_this/pipeline/aggregate/`
- `re-impl/cpp_beat_this/include/beat_this/postprocess/minimal/`
- `re-impl/cpp_beat_this/include/beat_this/validation/`
- `re-impl/cpp_beat_this/src/frontend/cpp_root/`
- `re-impl/cpp_beat_this/src/frontend/miniaudio/`
- `re-impl/cpp_beat_this/src/frontend/onnx/`
- `re-impl/cpp_beat_this/src/pipeline/split/`
- `re-impl/cpp_beat_this/src/pipeline/aggregate/`
- `re-impl/cpp_beat_this/src/postprocess/minimal/`
- `re-impl/cpp_beat_this/src/core/`
- `re-impl/cpp_beat_this/tools/onnx_export/`
- `re-impl/cpp_beat_this/tools/reference/`
- `re-impl/cpp_beat_this/tests/`
- `re-impl/models/`
- `re-impl/docs/`

파일은 단일 책임을 유지한다.

- 가능하면 파일당 `200~300` 라인 내외
- `400` 라인 초과 금지

기존 `cpp_root/include/stft_wrapper/`와 `cpp_root/include/STFT/`는 spectrogram frontend 구현의 재사용 원천으로 간주한다.
새 구현은 이 코드를 직접 include하거나, `re-impl` 안의 adapter 계층으로 감싸서 사용한다.
`cpp_root/include/miniaudio.h`는 file decode, PCM format conversion, resampling의 재사용 원천으로 간주한다.

## 9. waveform 정규화와 spectrogram frontend 규약

### 9.1 waveform 정규화 규약

spectrogram 전 단계의 waveform 정규화는 현재 Python 추론과 같은 의미를 유지해야 한다.

고정 규칙:

- `num_channels == 1`이면 그대로 사용
- `num_channels > 1`이면 채널 평균으로 mono 변환
- 공개 API의 multi-channel 입력은 interleaved frame layout으로 해석한다
- `sample_rate != 22050`이면 `22050 Hz`로 리샘플링

즉 현재 Python의 `Audio2Frames.signal2spect()`가 하는 채널 평균과 리샘플링 의미를 C++에서도 보존해야 한다.

### 9.2 `miniaudio.h` 사용 규약

`miniaudio.h`는 아래 역할로 사용한다.

- 파일 경로 입력의 decode
- encoded PCM을 `float32` interleaved PCM으로 변환
- mono waveform에 대한 resample

원칙:

- 파일 decode는 `ma_decoder` 계열 API를 사용한다.
- sample format은 `ma_format_f32`로 정규화한다.
- channel folding은 `miniaudio`의 implicit default mixing에 전부 맡기지 않는다.
- parity를 위해 multi-channel 입력은 decode 후 C++에서 산술 평균으로 mono 변환한다.
- resample은 `ma_resampler` 또는 `ma_data_converter`를 사용하되, 입력은 mono average 이후의 1-channel waveform으로 둔다.
- 즉 `decode -> explicit mono average -> resample -> cpp_root frontend` 순서를 기본 경로로 둔다.

`miniaudio`는 `soxr`와 동일 구현이 아니므로, 리샘플링 수치 차이는 fixture 기반으로 허용 오차를 검증해야 한다.

### 9.3 spectrogram frontend 규약

Python `LogMelSpect`의 external semantics를 유지해야 한다.

즉 `cpp_root` 기반 frontend 또는 optional preprocess artifact는 아래 의미를 보존해야 한다.

- `MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128, power=1, mel_scale="slaney", normalized="frame_length")`
- `log1p(1000 * mel)`

원칙:

- generic mel 근사 구현은 허용하지 않는다.
- mel filterbank, log scaling, frame advance, output axis 의미는 Python reference와 맞아야 한다.
- 기본 경로는 `cpp_root`의 STFT/mel 코드를 감싼 native C++ frontend다.
- optional preprocess artifact는 parity 검증이나 debug fallback 용도로만 유지한다.
- `windowSizeEXP = 10`, `windowSize = 1024`
- `hop_length = 441`에 대응하는 step을 유지해야 한다.
- `overlapRatio = 1 - 441 / 1024 = 0.5693359375`를 기준값으로 사용한다.
- 출력은 row-major `T x 128` `float32`로 정규화한다.

### 9.4 `cpp_root` 재사용 방침

우선 재사용 대상은 아래 파일군이다.

- `cpp_root/include/stft_wrapper/MelFilterBank.hpp`
- `cpp_root/include/stft_wrapper/STFT_Parallel.hpp`
- `cpp_root/include/stft_wrapper/SerialBackend.hpp`
- `cpp_root/include/stft_wrapper/SerialBackend.cpp`
- `cpp_root/include/stft_wrapper/OpenclBackend.hpp` optional

재사용 허용 범위:

- Slaney mel filter bank 생성 로직
- window/STFT/FFT orchestration
- serial backend 기반 CPU spectrogram frontend
- parity가 확보된 뒤의 optional OpenCL 가속 경로

현재 `cpp_root` 기본 구현과 Beat This 기준 사이에 존재하는 차이:

- `SerialBackend`와 `OpenclBackend`의 기본 `kMelBins = 80`
- `SerialBackend`와 `OpenclBackend`의 기본 `kDefaultSampleRate = 48000`
- `POST_PROCESS::mel_scale`가 내부적으로 `to_bin`과 `toPower`를 강제함
- `DCRemove_Common(...)`가 기본 serial path에 들어 있음
- `to_db`, `normalize_min_max`, `to_rgb` 같은 후처리 옵션이 함께 존재함

따라서 `cpp_root` 코드는 아래 조건을 만족하도록 adapter 또는 patch를 반드시 둔다.

- mel bin 수를 `128`로 강제한다.
- sample rate 기준을 `22050`로 강제한다.
- `f_min = 30`, `f_max = 11000`를 강제한다.
- `MelFormula::Slaney`, `MelNorm::Slaney`를 강제한다.
- Beat This의 `power=1` 의미와 맞지 않는 `toPower` 강제 경로를 그대로 쓰지 않는다.
- `to_db`, `normalize_min_max`, `to_rgb`는 금지한다.
- Python reference에 없는 `DCRemove_Common(...)`는 우회하거나 비활성화한다. parity fixture로 동일성이 입증되기 전까지 유지하면 안 된다.
- 최종 log scaling은 `log1p(1000 * mel)`로 별도 적용한다.
- 출력 축은 반드시 `T x 128`이어야 한다.

즉 `cpp_root`는 그대로 link하는 black box가 아니라, Beat This 전용 frontend 규약으로 조정한 뒤 재사용하는 upstream 코드로 취급한다.

## 10. spect split 규약

split 알고리즘은 `beat_this.inference.split_piece()`와 동일한 의미를 유지해야 한다.

고정 규칙:

- `chunk_size = 1500`
- `border_size = 6`
- `avoid_short_end = true`

세부 규칙:

- start index는 `-border_size`부터 시작
- step은 `chunk_size - 2 * border_size`
- 마지막 chunk가 너무 짧아지지 않도록 start를 왼쪽으로 이동
- 첫/마지막 chunk는 필요 시 zero padding

## 11. model chunk inference 규약

모델 추론은 전체 spectrogram에 대해 한 번에 하지 않는다.

반드시 아래 순서를 따른다.

1. `spect[T, 128]` 생성
2. split
3. 각 chunk에 대해 `model artifact` 실행
4. chunk logits를 aggregate

즉 C++ 런타임은 chunk 단위 inference orchestration을 직접 담당한다.

## 12. chunk aggregation 규약

aggregation은 `beat_this.inference.aggregate_prediction()`과 동일한 의미를 유지한다.

고정 규칙:

- 각 chunk의 양 끝 `border_size = 6` 프레임은 버린다.
- 남은 중앙 영역만 전체 출력으로 복원한다.
- overlap이 생기면 기본 모드는 `keep_first`다.

결과 shape:

- 최종 aggregated beat logits: `(T,)`
- 최종 aggregated downbeat logits: `(T,)`

공개 API에서는 이를 `FrameLogits` 구조체로 반환한다.

## 13. minimal postprocess 규약

후처리 대상은 aggregated logits이다.  
즉 입력은 확률이 아니라 raw logit이다.

알고리즘은 Python `Postprocessor(type="minimal")`와 동일 의미를 유지한다.

고정 규칙:

1. padding 위치는 `-1000`으로 취급
2. `7`프레임 max-pooling으로 local maxima만 남김
3. `logit > 0`만 peak로 유지
4. 인접 peak는 `width=1` 규칙으로 deduplicate
5. frame index를 `fps=50` 기준 초 단위로 변환
6. 각 downbeat는 가장 가까운 beat 시점으로 snap
7. 중복 downbeat는 제거

출력:

- `beats_seconds`
- `downbeats_seconds`

## 14. 에러 처리 계약

- 입력 계약 위반은 `std::invalid_argument`
- ONNX model load/session/run 실패는 `std::runtime_error`
- shape 계약 위반이나 내부 invariant 깨짐은 `std::runtime_error`
- custom exception hierarchy는 만들지 않는다

## 15. 빌드 기준

- 빌드 시스템은 `CMake`
- 언어 표준은 `C++20`
- 기본 라이브러리 타깃 이름은 `beat_this_cpp`
- 테스트 프레임워크는 `GoogleTest`

공식 개발 경로는 아래를 만족해야 한다.

- `cpp_root/include/miniaudio.h`를 함께 빌드 가능
- `cpp_root` STFT frontend를 함께 빌드 가능
- ONNX Runtime C++ 사용 가능
- GoogleTest 사용 가능
- Windows와 Linux에서 재현 가능한 빌드 스크립트 제공

## 16. reference fixture 생성

Python reference generator를 별도 도구로 둔다.

저장 위치 권장:

- `re-impl/cpp_beat_this/tools/reference/`
- `re-impl/cpp_beat_this/testdata/`

fixture는 최소한 아래를 포함한다.

- `waveform_reference_<case>.npz`
- `spect_reference_<case>.npz`
- `chunk_logits_reference_<case>.npz`
- `aggregated_logits_reference_<case>.npz`
- `beats_reference_<case>.json`
- `reference_manifest.json`

fixture 생성 기준은 현재 Python 구현이다.

## 17. 테스트 계획

### 17.1 `cpp_root` frontend parity

검증 내용:

- `beat_this.preprocessing.LogMelSpect` 출력과 `cpp_root` 기반 frontend 출력 비교
- `sample_rate = 22050`, `n_fft = 1024`, `hop_length = 441`, `n_mels = 128`, `f_min = 30`, `f_max = 11000`가 실제로 override되었는지 확인
- `MelFormula::Slaney`, `MelNorm::Slaney`가 실제 적용되었는지 확인
- `to_db`, `normalize_min_max`, `to_rgb`, implicit `toPower` 경로가 비활성화되었는지 확인
- `DCRemove_Common(...)`를 우회한 결과가 reference와 맞는지 확인

optional validation:

- preprocess artifact가 존재하면 `cpp_root` frontend 출력과 artifact 출력도 비교

acceptance:

- `rtol=1e-4`
- `atol=1e-5`

### 17.2 model chunk parity

검증 내용:

- 동일 spect chunk 입력에 대해 Python reference logits와 ONNX model logits 비교

acceptance:

- `rtol=1e-4`
- `atol=1e-5`

### 17.3 split parity

검증 내용:

- chunk start positions
- zero padding 위치
- 각 chunk tensor shape

acceptance:

- exact match

### 17.4 aggregation parity

검증 내용:

- aggregated beat/downbeat logits가 Python reference와 같은 위치에 복원되는지 확인

acceptance:

- exact match

### 17.5 minimal postprocess parity

검증 내용:

- peak selection
- deduplication
- beat/downbeat time arrays

acceptance:

- beat/downbeat 개수 exact match
- 각 대응 시점 오차 `<= 1 / 50 = 0.02s`

### 17.6 end-to-end parity

최소 fixture 세트:

- 저장소 테스트 오디오에서 추출한 waveform fixture 1종
- 일정한 BPM의 synthetic click track
- 중간에 tempo change가 있는 synthetic track

acceptance:

- beat count 차이 `±1` 이내
- downbeat count 차이 `±1` 이내
- 대응 시점 오차 `<= 0.02s`

### 17.7 입력 계약 테스트

- mono 입력이 그대로 유지되는지 확인
- multi-channel 입력이 채널 평균으로 정규화되는지 확인
- `22050`가 아닌 입력 sample rate가 내부적으로 `22050`로 리샘플링되는지 확인
- `miniaudio` file adapter가 `float32` PCM으로 decode되는지 확인
- 동일 파일에 대해 `ProcessFile(...)`와 수동 waveform 입력 경로가 같은 결과를 내는지 확인
- empty buffer -> 에러
- `NaN` 포함 -> 에러
- `Inf` 포함 -> 에러
- non-finite 값 -> 에러
- sample rate와 channel count를 받는 waveform API가 존재하는지 확인

## 18. 구현 우선순위

구현은 아래 순서로 진행한다.

1. Python reference fixture generator 작성
2. `miniaudio` file decoder/resampler adapter 작성
3. `decode -> explicit mono average -> resample` parity 테스트 고정
4. `cpp_root` frontend adapter 작성
5. `cpp_root` 기본값 `80 mel`, `48 kHz`, `toPower`, `DCRemove` 등 Beat This와 어긋나는 동작 제거
6. `cpp_root` frontend parity 테스트 고정
7. model artifact export
8. ONNX runner 작성
9. split/aggregate C++ 구현
10. minimal postprocessor 구현
11. raw logits API
12. final beats API
13. optional preprocess artifact export
14. 전체 parity 테스트 고정

## 19. 최종 한 줄 요약

새 C++ 변환 결과물은 아래 경계를 재현해야 한다.

```text
encoded audio file or float32 waveform
-> optional miniaudio decode
-> channel normalization
-> resample to 22050
-> cpp_root STFT/log-mel frontend
-> spect(T x 128)
-> split(1500, border 6)
-> model artifact on each chunk
-> aggregate logits
-> minimal peak postprocess
-> beats/downbeats in seconds
```
