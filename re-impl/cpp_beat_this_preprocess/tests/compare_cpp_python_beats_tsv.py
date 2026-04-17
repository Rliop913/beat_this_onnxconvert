from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python final .beats TSV export against the C++ port."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cpp-exe", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    return parser.parse_args()


def make_deterministic_waveform(
    sample_rate: int = 44100, duration_seconds: float = 32.0
) -> tuple[np.ndarray, int]:
    num_frames = int(sample_rate * duration_seconds)
    time = np.arange(num_frames, dtype=np.float32) / np.float32(sample_rate)

    left = 0.30 * np.sin(2.0 * np.pi * 220.0 * time)
    left += 0.15 * np.sin(2.0 * np.pi * 440.0 * time)

    right = 0.28 * np.sin((2.0 * np.pi * 330.0 * time) + 0.25)
    right += 0.12 * np.cos(2.0 * np.pi * 550.0 * time)

    click_length = sample_rate // 100
    click = np.linspace(1.0, 0.0, click_length, dtype=np.float32)
    pulse_starts = np.arange(0, num_frames, sample_rate // 2, dtype=np.int64)

    for pulse_index, start in enumerate(pulse_starts):
        end = min(start + click_length, num_frames)
        amplitude = 0.80 if pulse_index % 4 == 0 else 0.45
        left[start:end] += amplitude * click[: end - start]

        shifted_start = start + (click_length // 4)
        shifted_end = min(shifted_start + click_length, num_frames)
        if shifted_start < num_frames:
            right[shifted_start:shifted_end] += (amplitude * 0.85) * click[
                : shifted_end - shifted_start
            ]

    stereo = np.stack([left, right], axis=1).astype(np.float32)
    np.clip(stereo, -1.0, 1.0, out=stereo)
    return stereo, sample_rate


def write_waveform_binary(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform.reshape(-1).astype("<f4").tofile(path)


def load_beats_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    numbers: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        time_text, number_text = stripped.split("\t")
        times.append(float(time_text))
        numbers.append(int(number_text))
    return np.asarray(times, dtype=np.float64), np.asarray(numbers, dtype=np.int64)


def max_abs_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 and rhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


def main() -> int:
    if np is None or torch is None:
        print(
            "Skipping beats TSV parity check because numpy or torch is unavailable in the selected Python.",
            file=sys.stderr,
        )
        return 125

    try:
        from beat_this.inference import Audio2Beats
        from beat_this.utils import save_beat_tsv
    except ModuleNotFoundError as error:
        print(
            f"Skipping beats TSV parity check because Python inference dependencies are missing: {error}",
            file=sys.stderr,
        )
        return 125

    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    model_path = Path(args.model).resolve()
    cpp_exe = Path(args.cpp_exe).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"onnx model not found: {model_path}")
    if not cpp_exe.exists():
        raise FileNotFoundError(f"cpp beats tsv dump executable not found: {cpp_exe}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    waveform, sample_rate = make_deterministic_waveform()
    waveform_bin = artifact_dir / "waveform_interleaved_f32.bin"
    cpp_tsv = artifact_dir / "cpp_beats.beats"
    py_tsv = artifact_dir / "python_beats.beats"
    summary_json = artifact_dir / "beats_tsv_parity_summary.json"

    write_waveform_binary(waveform_bin, waveform)

    runner = Audio2Beats(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        float16=False,
        dbn=False,
    )
    with torch.inference_mode():
        beat_ref, downbeat_ref = runner(waveform, sample_rate)
    beat_ref_np = np.asarray(beat_ref, dtype=np.float64)
    downbeat_ref_np = np.asarray(downbeat_ref, dtype=np.float64)
    save_beat_tsv(beat_ref_np, downbeat_ref_np, py_tsv)

    command = [
        str(cpp_exe),
        "--model",
        str(model_path),
        "--waveform-bin",
        str(waveform_bin),
        "--sample-rate",
        str(sample_rate),
        "--num-channels",
        "2",
        "--output",
        str(cpp_tsv),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cpp beats tsv dump failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    beat_cpp, numbers_cpp = load_beats_tsv(cpp_tsv)
    beat_ref_loaded, numbers_ref = load_beats_tsv(py_tsv)

    if beat_cpp.shape[0] != beat_ref_loaded.shape[0]:
        raise AssertionError(
            f"beat count mismatch: cpp={beat_cpp.shape[0]} python={beat_ref_loaded.shape[0]}"
        )

    np.testing.assert_allclose(beat_cpp, beat_ref_loaded, atol=args.atol, rtol=args.rtol)
    np.testing.assert_array_equal(numbers_cpp, numbers_ref)

    summary = {
        "num_beats": int(beat_cpp.shape[0]),
        "num_downbeats": int(np.count_nonzero(numbers_cpp == 1)),
        "atol": args.atol,
        "rtol": args.rtol,
        "beat_time_max_abs_error": max_abs_error(beat_cpp, beat_ref_loaded),
        "cpp_beats_path": str(cpp_tsv),
        "python_beats_path": str(py_tsv),
        "text_identical": cpp_tsv.read_text(encoding="utf-8")
        == py_tsv.read_text(encoding="utf-8"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
