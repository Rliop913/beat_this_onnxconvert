"""
Utilities for exporting Beat This! checkpoints to ONNX.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key

DEFAULT_ONNX_OPSET = 17
DEFAULT_ORT_VERSION = "1.24.4"
DEFAULT_VERIFY_LENGTHS = (128, 321, 1500)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_checkpoint_dir() -> Path:
    return get_repo_root() / "checkpoints"


def get_default_onnx_output_path(variant: str) -> Path:
    return get_repo_root() / "re-impl" / "models" / f"beat_this_model_{variant}.onnx"


def _variant_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("beat_this-"):
        stem = stem[len("beat_this-") :]
    return stem


def resolve_export_checkpoint_path(checkpoint: str | Path) -> tuple[Path, str]:
    """
    Resolve a checkpoint for ONNX export.

    If `checkpoint` points to a file, use it directly. Otherwise interpret it as a
    checkpoint shortname and look for `checkpoints/beat_this-<name>.ckpt`.
    """

    repo_root = get_repo_root()
    checkpoint_text = str(checkpoint)
    candidate = Path(checkpoint_text)

    looks_like_path = (
        candidate.suffix == ".ckpt"
        or candidate.is_absolute()
        or len(candidate.parts) > 1
        or checkpoint_text.startswith(".")
    )
    if looks_like_path:
        resolved = candidate if candidate.is_absolute() else repo_root / candidate
        if not resolved.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {resolved}\n"
                f"Copy the checkpoint into {get_default_checkpoint_dir()} "
                "or pass an explicit .ckpt path."
            )
        return resolved.resolve(), _variant_from_path(resolved)

    resolved = get_default_checkpoint_dir() / f"beat_this-{checkpoint_text}.ckpt"
    if not resolved.exists():
        raise FileNotFoundError(
            f"Checkpoint shortname '{checkpoint_text}' was not found.\n"
            f"Expected file: {resolved}\n"
            "Copy the checkpoint there and rerun the export."
        )
    return resolved.resolve(), checkpoint_text


def _load_checkpoint_file(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    weights_only = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        weights_only["weights_only"] = True
    return torch.load(checkpoint_path, map_location=device, **weights_only)


def load_model_from_checkpoint_path(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> BeatThis:
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    hparams = checkpoint["hyper_parameters"]
    hparams = {
        key: value
        for key, value in hparams.items()
        if key in set(inspect.signature(BeatThis).parameters)
    }
    model = BeatThis(**hparams)
    state_dict = replace_state_dict_key(dict(checkpoint["state_dict"]), "model.", "")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


class OnnxSafeBeatThis(nn.Module):
    """
    Export-only wrapper that normalizes the model output to a single logits tensor.
    """

    def __init__(self, model: BeatThis):
        super().__init__()
        self.model = copy.deepcopy(model).eval()

    def forward(self, spect: torch.Tensor) -> torch.Tensor:
        outputs = self.model(spect)
        beat = outputs["beat"].float()
        downbeat = outputs["downbeat"].float()
        return torch.stack((beat, downbeat), dim=-1)


def build_onnx_export_model(model: BeatThis) -> OnnxSafeBeatThis:
    return OnnxSafeBeatThis(model).eval()


def make_deterministic_spectrogram(num_frames: int) -> torch.Tensor:
    values = torch.arange(num_frames * 128, dtype=torch.float32).reshape(
        1, num_frames, 128
    )
    return torch.sin(values / 31.0) + 0.5 * torch.cos(values / 17.0)


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required for model export. Install the project dependencies first."
        ) from exc
    return onnx


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for export verification. Install the project dependencies first."
        ) from exc
    version = getattr(ort, "__version__", "")
    if not str(version).startswith(DEFAULT_ORT_VERSION):
        raise RuntimeError(
            f"ONNX Runtime {DEFAULT_ORT_VERSION} is required, got {version!r}."
        )
    return ort


def _set_onnx_metadata(onnx_path: Path, metadata: dict[str, str]) -> None:
    onnx = _require_onnx()
    model_proto = onnx.load(str(onnx_path))
    existing = {prop.key: prop for prop in model_proto.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            prop = model_proto.metadata_props.add()
            prop.key = key
            prop.value = value
    onnx.save(model_proto, str(onnx_path))


def export_model_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    variant: str,
    checkpoint_path: str | Path,
    opset: int = DEFAULT_ONNX_OPSET,
    example_num_frames: int = 1500,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    example_input = make_deterministic_spectrogram(example_num_frames)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_input,),
            str(output_path),
            input_names=["spect"],
            output_names=["logits"],
            dynamic_axes={"spect": {1: "time"}, "logits": {1: "time"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx = _require_onnx()
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)

    _set_onnx_metadata(
        output_path,
        {
            "checkpoint_source": str(checkpoint_path),
            "variant": variant,
            "opset": str(opset),
            "target_ort_version": DEFAULT_ORT_VERSION,
            "input_contract": "spect[1,T,128] float32",
            "output_contract": "logits[1,T,2] float32",
        },
    )
    return output_path


def verify_onnx_export(
    model: nn.Module,
    onnx_path: str | Path,
    *,
    lengths: tuple[int, ...] = DEFAULT_VERIFY_LENGTHS,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    ort = _require_onnxruntime()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    report: list[dict[str, Any]] = []

    with torch.inference_mode():
        for length in lengths:
            spect = make_deterministic_spectrogram(length)
            torch_logits = model(spect).cpu().numpy()
            ort_logits = session.run(["logits"], {"spect": spect.cpu().numpy()})[0]

            np.testing.assert_allclose(
                ort_logits, torch_logits, atol=atol, rtol=rtol, equal_nan=False
            )
            report.append(
                {
                    "num_frames": length,
                    "shape": list(torch_logits.shape),
                    "max_abs_error": float(np.max(np.abs(ort_logits - torch_logits))),
                }
            )

    return {
        "onnx_path": str(onnx_path),
        "ort_version": ort.__version__,
        "cases": report,
        "atol": atol,
        "rtol": rtol,
    }


def export_checkpoint_to_onnx(
    checkpoint: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    skip_verify: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path, variant = resolve_export_checkpoint_path(checkpoint)
    if output_path is None:
        output_path = get_default_onnx_output_path(variant)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    base_model = load_model_from_checkpoint_path(checkpoint_path, device=device)
    export_model = build_onnx_export_model(base_model).to(device).eval()
    onnx_path = export_model_to_onnx(
        export_model,
        output_path,
        variant=variant,
        checkpoint_path=checkpoint_path,
    )

    result = {
        "checkpoint_path": str(checkpoint_path),
        "variant": variant,
        "output_path": str(onnx_path),
        "verified": False,
    }
    if not skip_verify:
        result["verification"] = verify_onnx_export(export_model, onnx_path)
        result["verified"] = True
    return result
