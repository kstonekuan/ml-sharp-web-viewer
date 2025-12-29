"""Modal app definition and inference function for SHARP.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import modal

from sharp.modal.image import create_modal_image

if TYPE_CHECKING:
    import numpy as np

    from sharp.utils.gaussians import Gaussians3D

LOGGER = logging.getLogger(__name__)

# Modal app configuration
APP_NAME = "sharp-gaussian-splat"
VOLUME_NAME = "sharp-model-cache"
MODEL_CACHE_PATH = "/cache/models"
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
TIMEOUT_SECONDS = 300

# GPU type mapping
GpuTier = Literal["t4", "l4", "a10", "a100", "h100"]

# Create Modal app and volume
app = modal.App(name=APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
modal_image = create_modal_image()


def _load_image_from_bytes(image_bytes: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Load an image from bytes and extract focal length.

    Args:
        image_bytes: Raw image file bytes.
        filename: Original filename (used for format detection).

    Returns:
        Tuple of (image array, focal length in pixels).
    """
    import numpy as np
    import pillow_heif
    from PIL import ExifTags, Image

    file_ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(image_bytes)

    if file_ext in [".heic"]:
        heif_file = pillow_heif.open_heif(buffer, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(buffer)

    # Extract EXIF data
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    # Handle image orientation
    exif_orientation = exif_dict.get("Orientation", 1)
    if exif_orientation == 3:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_180)
    elif exif_orientation == 6:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_270)
    elif exif_orientation == 8:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_90)

    # Extract focal length
    f_35mm = exif_dict.get("FocalLengthIn35mmFilm", exif_dict.get("FocalLenIn35mmFilm"))
    if f_35mm is None or f_35mm < 1:
        f_35mm = exif_dict.get("FocalLength")
        if f_35mm is None:
            LOGGER.warning(f"No focal length in EXIF for {filename}, using 30mm default.")
            f_35mm = 30.0
        elif f_35mm < 10.0:
            # Crude approximation for non-35mm sensors
            f_35mm *= 8.4

    img = np.asarray(img_pil)

    # Convert to RGB if needed
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))
    img = img[:, :, :3]  # Remove alpha if present

    # Convert focal length to pixels
    height, width = img.shape[:2]
    f_px = f_35mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)

    return img, f_px


def _serialize_ply_to_bytes(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int]
) -> bytes:
    """Serialize Gaussians3D to PLY bytes.

    Args:
        gaussians: The Gaussians3D to serialize.
        f_px: Focal length in pixels.
        image_shape: Image dimensions as (height, width).

    Returns:
        PLY file content as bytes.
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement

    from sharp.utils import color_space as cs_utils
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    # Convert linearRGB to sRGB for compatibility with public renderers
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (xyz, colors, opacity_logits, scale_logits, quaternions),
        dim=1,
    )

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    image_height, image_width = image_shape

    # Export image size
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # Export intrinsics
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # Export dummy extrinsics
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # Export frame info
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # Export disparity ranges
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # Export colorspace
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    # Export version
    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )

    # Write to bytes
    buffer = io.BytesIO()
    plydata.write(buffer)
    buffer.seek(0)
    return buffer.read()


# GPU-specific function variants
@app.function(
    gpu="t4",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_t4(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Run inference on T4 GPU ($0.59/hr, budget option)."""
    return _predict_impl(image_bytes, filename)


@app.function(
    gpu="l4",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_l4(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Run inference on L4 GPU ($0.80/hr)."""
    return _predict_impl(image_bytes, filename)


@app.function(
    gpu="a10",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_a10(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Run inference on A10 GPU ($1.10/hr, default)."""
    return _predict_impl(image_bytes, filename)


@app.function(
    gpu="a100",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_a100(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Run inference on A100 GPU ($2.50/hr)."""
    return _predict_impl(image_bytes, filename)


@app.function(
    gpu="h100",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_h100(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Run inference on H100 GPU ($3.95/hr, fastest)."""
    return _predict_impl(image_bytes, filename)


def _predict_impl(image_bytes: bytes, filename: str) -> tuple[str, bytes]:
    """Shared implementation for all GPU variants.

    This is called by the GPU-specific functions and contains the actual
    inference logic (same as predict_gaussian_splat).
    """
    import torch
    import torch.nn.functional as F

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import unproject_gaussians

    LOGGER.info(f"Processing {filename} on Modal GPU")

    # Load image from bytes
    image, f_px = _load_image_from_bytes(image_bytes, filename)
    height, width = image.shape[:2]

    device = torch.device("cuda")

    # Load or download model
    model_path = Path(MODEL_CACHE_PATH) / "sharp_model.pt"

    def download_model() -> dict:
        """Download model from URL and cache to volume."""
        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True, map_location=device
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, model_path)
        model_volume.commit()
        LOGGER.info("Model cached to volume")
        return state

    if model_path.exists():
        LOGGER.info("Loading cached model from volume")
        try:
            state_dict = torch.load(model_path, weights_only=True, map_location=device)
        except Exception as e:
            LOGGER.warning(f"Cached model is corrupted: {e}")
            LOGGER.info("Deleting corrupted cache and re-downloading...")
            model_path.unlink()
            model_volume.commit()
            state_dict = download_model()
    else:
        state_dict = download_model()

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running inference")
    with torch.no_grad():
        gaussians_ndc = gaussian_predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )

    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    LOGGER.info("Serializing to PLY")
    ply_bytes = _serialize_ply_to_bytes(gaussians, f_px, (height, width))

    output_filename = Path(filename).stem + ".ply"
    LOGGER.info(f"Done processing {filename}")

    return output_filename, ply_bytes


def get_predict_function(gpu_tier: GpuTier = "a10"):
    """Get the appropriate predict function for the GPU tier.

    Args:
        gpu_tier: One of 't4', 'l4', 'a10', 'a100', 'h100'.

    Returns:
        The Modal function for the specified GPU tier.
    """
    functions = {
        "t4": predict_gaussian_splat_t4,
        "l4": predict_gaussian_splat_l4,
        "a10": predict_gaussian_splat_a10,
        "a100": predict_gaussian_splat_a100,
        "h100": predict_gaussian_splat_h100,
    }
    return functions[gpu_tier]
