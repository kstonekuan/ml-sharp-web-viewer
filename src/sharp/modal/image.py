"""Modal container image definition for SHARP inference.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import modal


def create_modal_image() -> modal.Image:
    """Create Modal image with all dependencies for SHARP inference.

    Returns:
        A Modal Image configured with PyTorch, CUDA, and all SHARP dependencies.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.5.1",
            "torchvision==0.20.1",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "timm>=1.0.0",
            "scipy>=1.11.0",
            "plyfile>=1.0.0",
            "imageio>=2.30.0",
            "pillow-heif>=0.16.0",
            "numpy>=1.24.0",
            "click>=8.0.0",
        )
        # gsplat requires CUDA compilation - install with build isolation disabled
        .run_commands("pip install gsplat --no-build-isolation")
    )
