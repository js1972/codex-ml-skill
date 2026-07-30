#!/usr/bin/env python3
"""Check AutoGluon's FastAI dependency compatibility without fitting a model."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def fastai_optimizer_uses_legacy_starmap() -> bool | None:
    try:
        distribution = metadata.distribution("fastai")
    except metadata.PackageNotFoundError:
        return None

    optimizer = Path(distribution.locate_file("fastai/optimizer.py"))
    try:
        source = optimizer.read_text(encoding="utf-8")
    except OSError:
        return None
    return ".starmap(" in source


def fastcore_supports_starmap() -> bool | None:
    try:
        from fastcore.foundation import L
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        return callable(getattr(L([("learning_rate", 0.01)]), "starmap"))
    except AttributeError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the current interpreter for the FastAI/Fastcore compatibility "
            "needed by AutoGluon's FastAI neural-network family. No model is fitted."
        )
    )
    parser.add_argument(
        "--preset",
        required=True,
        help="Approved AutoGluon preset, recorded for the preflight result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actions: list[str] = []

    versions = {
        "autogluon.tabular": package_version("autogluon.tabular"),
        "fastai": package_version("fastai"),
        "fastcore": package_version("fastcore"),
        "torch": package_version("torch"),
    }

    print("AutoGluon compatibility preflight")
    print(f"preset: {args.preset}")
    for name, version in versions.items():
        print(f"{name}: {version or 'not installed'}")

    if versions["autogluon.tabular"] is None:
        actions.append(
            "Install the approved AutoGluon extras in this interpreter before fitting."
        )

    legacy_call = fastai_optimizer_uses_legacy_starmap()
    starmap_available = fastcore_supports_starmap()
    if (
        versions["fastai"] is None
        or versions["fastcore"] is None
        or versions["torch"] is None
    ):
        fastai_status = "install_required"
        actions.append(
            "Install a compatible FastAI/Fastcore/Torch set so AutoGluon's "
            "FastAI neural-network family can run."
        )
    elif legacy_call is True and starmap_available is False:
        fastai_status = "incompatible"
        actions.append(
            "The installed FastAI still calls L.starmap, which the installed "
            "Fastcore removed. Add the temporary constraint fastcore<2, "
            "resolve the environment, and rerun this check."
        )
    elif legacy_call is None or starmap_available is None:
        fastai_status = "unverified"
        actions.append(
            "Verify the FastAI optimizer API in this interpreter before fitting."
        )
    else:
        fastai_status = "ready"
    print(f"conventional-neural-fastai: {fastai_status}")

    if actions:
        print("overall: action_required")
        print("actions:")
        for action in actions:
            print(f"- {action}")
        return 2

    print("overall: ready")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    sys.exit(main())
