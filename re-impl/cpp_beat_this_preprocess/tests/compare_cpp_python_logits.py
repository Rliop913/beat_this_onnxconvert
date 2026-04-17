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
        description="Compare end-to-end Python logits against the C++ ONNX Runtime runner."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cpp-exe", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
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


def write_reference_tsv(path: Path, beat: np.ndarray, downbeat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.arange(beat.shape[0], dtype=np.int64)
    stacked = np.column_stack((frame, beat, downbeat))
    np.savetxt(
        path,
        stacked,
        fmt=["%d", "%.9g", "%.9g"],
        delimiter="\t",
        header="frame\tbeat\tdownbeat",
        comments="",
    )


def load_cpp_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter="\t", skiprows=1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values[:, 1].astype(np.float32), values[:, 2].astype(np.float32)


def main() -> int:
    if np is None or torch is None:
        print(
            "Skipping parity check because numpy or torch is unavailable in the selected Python.",
            file=sys.stderr,
        )
        return 125

    try:
        from beat_this.inference import Audio2Frames
    except ModuleNotFoundError as error:
        print(f"Skipping parity check because Python inference dependencies are missing: {error}",
              file=sys.stderr)
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
        raise FileNotFoundError(f"cpp logits dump executable not found: {cpp_exe}")

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
    cpp_tsv = artifact_dir / "cpp_logits.tsv"
    py_tsv = artifact_dir / "python_logits.tsv"
    summary_json = artifact_dir / "parity_summary.json"

    write_waveform_binary(waveform_bin, waveform)

    runner = Audio2Frames(checkpoint_path=str(checkpoint_path), device="cpu", float16=False)
    with torch.inference_mode():
        beat_ref, downbeat_ref = runner(waveform, sample_rate)
    beat_ref_np = beat_ref.cpu().numpy().astype(np.float32)
    downbeat_ref_np = downbeat_ref.cpu().numpy().astype(np.float32)
    write_reference_tsv(py_tsv, beat_ref_np, downbeat_ref_np)

    command = [
        str(cpp_exe),
        "--model",
        str(model_path),
        "--format",
        "logits",
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
            "cpp dump tool failed for logits format\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    beat_cpp, downbeat_cpp = load_cpp_tsv(cpp_tsv)
    if beat_cpp.shape[0] != beat_ref_np.shape[0]:
        raise AssertionError(
            f"frame count mismatch: cpp={beat_cpp.shape[0]} python={beat_ref_np.shape[0]}"
        )

    np.testing.assert_allclose(beat_cpp, beat_ref_np, atol=args.atol, rtol=args.rtol)
    np.testing.assert_allclose(
        downbeat_cpp, downbeat_ref_np, atol=args.atol, rtol=args.rtol
    )

    summary = {
        "num_frames": int(beat_cpp.shape[0]),
        "atol": args.atol,
        "rtol": args.rtol,
        "beat_max_abs_error": float(np.max(np.abs(beat_cpp - beat_ref_np))),
        "beat_mean_abs_error": float(np.mean(np.abs(beat_cpp - beat_ref_np))),
        "downbeat_max_abs_error": float(np.max(np.abs(downbeat_cpp - downbeat_ref_np))),
        "downbeat_mean_abs_error": float(np.mean(np.abs(downbeat_cpp - downbeat_ref_np))),
        "waveform_path": str(waveform_bin),
        "cpp_logits_path": str(cpp_tsv),
        "python_logits_path": str(py_tsv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
