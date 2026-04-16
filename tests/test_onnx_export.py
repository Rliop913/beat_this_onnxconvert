from pathlib import Path
from uuid import uuid4

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("rotary_embedding_torch")

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.onnx_export import (
    OnnxSafeBeatThis,
    build_onnx_export_model,
    export_model_to_onnx,
    make_deterministic_spectrogram,
    resolve_export_checkpoint_path,
    verify_onnx_export,
)


def _make_workspace_test_dir(name: str) -> Path:
    path = Path("build") / "test-artifacts" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_export_checkpoint_path_shortname(monkeypatch):
    checkpoint_dir = _make_workspace_test_dir("checkpoints") / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "beat_this-final0.ckpt"
    checkpoint_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "beat_this.model.onnx_export.get_default_checkpoint_dir",
        lambda: checkpoint_dir,
    )

    resolved, variant = resolve_export_checkpoint_path("final0")
    assert resolved == checkpoint_path.resolve()
    assert variant == "final0"


def test_onnx_wrapper_matches_base_model_outputs():
    model = BeatThis().eval()
    wrapper = OnnxSafeBeatThis(model).eval()
    spect = make_deterministic_spectrogram(64)

    with torch.inference_mode():
        base_output = model(spect)
        wrapper_output = wrapper(spect)

    assert wrapper_output.shape == (1, 64, 2)
    assert torch.allclose(wrapper_output[..., 0], base_output["beat"])
    assert torch.allclose(wrapper_output[..., 1], base_output["downbeat"])


def test_export_and_verify_random_model():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    model = build_onnx_export_model(BeatThis().eval())
    output_path = _make_workspace_test_dir("onnx") / "beat_this_random.onnx"

    export_model_to_onnx(
        model,
        output_path,
        variant="random",
        checkpoint_path=Path("random.ckpt"),
    )
    report = verify_onnx_export(model, output_path, lengths=(128, 321))

    assert output_path.exists()
    assert report["cases"][0]["shape"] == [1, 128, 2]
    assert report["cases"][1]["shape"] == [1, 321, 2]


def test_missing_repo_checkpoint_error(monkeypatch):
    checkpoint_dir = _make_workspace_test_dir("missing") / "checkpoints"
    checkpoint_dir.mkdir()

    monkeypatch.setattr(
        "beat_this.model.onnx_export.get_default_checkpoint_dir",
        lambda: checkpoint_dir,
    )

    with pytest.raises(FileNotFoundError, match="checkpoints"):
        resolve_export_checkpoint_path("final0")
