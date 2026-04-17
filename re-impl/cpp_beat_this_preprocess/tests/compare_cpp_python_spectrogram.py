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
    import soxr
except ModuleNotFoundError:
    soxr = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python and C++ log-mel spectrogram generation."
    )
    parser.add_argument("--cpp-exe", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def make_deterministic_waveform(
    sample_rate: int = 44100, duration_seconds: float = 16.0
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


def write_reference_tsv(path: Path, spectrogram: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, spectrogram, fmt="%.9g", delimiter="\t")


def load_tsv_matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter="\t", dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def main() -> int:
    if np is None or soxr is None or torch is None:
        print(
            "Skipping spectrogram parity check because numpy, soxr, or torch is unavailable.",
            file=sys.stderr,
        )
        return 125

    try:
        from beat_this.preprocessing import LogMelSpect
    except ModuleNotFoundError as error:
        print(
            f"Skipping spectrogram parity check because preprocessing dependencies are missing: {error}",
            file=sys.stderr,
        )
        return 125

    args = parse_args()

    cpp_exe = Path(args.cpp_exe).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not cpp_exe.exists():
        raise FileNotFoundError(f"cpp dump executable not found: {cpp_exe}")

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
    cpp_tsv = artifact_dir / "cpp_spectrogram.tsv"
    py_tsv = artifact_dir / "python_spectrogram.tsv"
    summary_json = artifact_dir / "spectrogram_parity_summary.json"

    write_waveform_binary(waveform_bin, waveform)

    mono = waveform.mean(axis=1)
    mono = soxr.resample(mono, in_rate=sample_rate, out_rate=22050)
    spect = LogMelSpect(device="cpu")
    with torch.inference_mode():
        spect_ref = spect(torch.tensor(mono, dtype=torch.float32)).cpu().numpy()
    spect_ref = spect_ref.astype(np.float32)
    write_reference_tsv(py_tsv, spect_ref)

    command = [
        str(cpp_exe),
        "--format",
        "spectrogram",
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
            "cpp dump tool failed for spectrogram format\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    spect_cpp = load_tsv_matrix(cpp_tsv)
    if spect_cpp.shape != spect_ref.shape:
        raise AssertionError(
            f"spectrogram shape mismatch: cpp={spect_cpp.shape} python={spect_ref.shape}"
        )

    np.testing.assert_allclose(spect_cpp, spect_ref, atol=args.atol, rtol=args.rtol)

    summary = {
        "num_frames": int(spect_cpp.shape[0]),
        "num_bins": int(spect_cpp.shape[1]),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_error": float(np.max(np.abs(spect_cpp - spect_ref))),
        "mean_abs_error": float(np.mean(np.abs(spect_cpp - spect_ref))),
        "waveform_path": str(waveform_bin),
        "cpp_spectrogram_path": str(cpp_tsv),
        "python_spectrogram_path": str(py_tsv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
