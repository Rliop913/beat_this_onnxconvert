#!/usr/bin/env python3
"""
Export a Beat This! checkpoint to an ONNX model for ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import sys

from beat_this.model.onnx_export import export_checkpoint_to_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a repo-local Beat This checkpoint to ONNX and verify it with "
            "ONNX Runtime 1.24.4."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final0",
        help=(
            "Checkpoint shortname or .ckpt path. Shortnames are resolved to "
            "checkpoints/beat_this-<name>.ckpt."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to re-impl/models/beat_this_model_<variant>.onnx.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output ONNX file if it already exists.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime parity verification after export.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = export_checkpoint_to_onnx(
            args.checkpoint,
            output_path=args.output,
            overwrite=args.overwrite,
            skip_verify=args.skip_verify,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
