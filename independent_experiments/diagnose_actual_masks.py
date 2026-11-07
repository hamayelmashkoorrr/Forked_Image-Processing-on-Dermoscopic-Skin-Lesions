"""
Diagnostic for the ACTUAL detect_vignette() in src/segmentation.py.

Run this from the project root:
    python diagnose_actual_masks.py

Or specify image IDs:
    python diagnose_actual_masks.py --images ISIC_0000177 ISIC_0000879

By default, images are read from preprocessed_no_clahe/.
Change --input-dir if your images are stored elsewhere.

Outputs are saved in actual_mask_diagnostics/.

Important:
- FOV mask = field-of-view mask captured BEFORE sticker exclusion.
- Artifact mask = sticker mask returned by detect_stickers().
- Final valid mask = after sticker exclusion and boundary erosion.
- This script does not modify images, config.py, or segmentation.py.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.config import SEG_CFG
from src.segmentation import detect_vignette


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_IDS = ["ISIC_0000177", "ISIC_0000879", "ISIC_0000599", "ISIC_0000702", "ISIC_0000758"]


def find_image(input_dir, image_id):
    """Find an image by ID, allowing common image extensions."""
    for ext in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate

    matches = sorted(input_dir.glob(f"{image_id}.*"))
    matches = [p for p in matches if p.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {image_id} in {input_dir.resolve()}"
    )


def save_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def overlay_mask(bgr, mask, colour, alpha=0.38):
    """Colour only the True pixels of a boolean mask.
    An empty mask is a valid outcome (e.g. no sticker found): return the plain image."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = bgr.copy()
    colour_layer = np.zeros_like(bgr)
    colour_layer[:] = colour
    result[mask] = cv2.addWeighted(
        bgr[mask], 1.0 - alpha, colour_layer[mask], alpha, 0
    )
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def diagnose_one(image_path, output_dir):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not read {image_path}")

    # Call the real project function, not a copied implementation.
    final_valid, reported_fraction, fov_mask, artifact_mask = (
        detect_vignette(bgr, SEG_CFG, return_masks=True)
    )

    fov_mask = np.asarray(fov_mask, dtype=bool)
    artifact_mask = np.asarray(artifact_mask, dtype=bool)
    final_valid = np.asarray(final_valid, dtype=bool)

    if not (
        fov_mask.shape == artifact_mask.shape == final_valid.shape
    ):
        raise ValueError(
            "Returned masks do not have matching dimensions: "
            f"FOV={fov_mask.shape}, artifact={artifact_mask.shape}, "
            f"valid={final_valid.shape}"
        )

    # Expected final valid region before erosion is FOV minus artifacts.
    expected_before_erosion = fov_mask & ~artifact_mask

    # The final valid mask is eroded, so it should be contained in this region.
    unexpected_valid_pixels = final_valid & ~expected_before_erosion

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    fov_path = output_dir / f"{stem}_01_fov_mask.png"
    artifact_path = output_dir / f"{stem}_02_artifact_mask.png"
    expected_path = output_dir / f"{stem}_03_fov_minus_artifacts.png"
    valid_path = output_dir / f"{stem}_04_final_valid_mask.png"
    panel_path = output_dir / f"{stem}_diagnostic.png"

    save_mask(fov_path, fov_mask)
    save_mask(artifact_path, artifact_mask)
    save_mask(expected_path, expected_before_erosion)
    save_mask(valid_path, final_valid)

    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fov_overlay = overlay_mask(bgr, fov_mask, (0, 210, 0))
    # Colours are BGR because overlay_mask works on the OpenCV image.
    artifact_overlay = overlay_mask(bgr, artifact_mask, (40, 40, 255))   # red
    valid_overlay = overlay_mask(bgr, final_valid, (0, 170, 255))        # orange

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Actual segmentation masks: {image_path.name}",
        fontsize=15
    )

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original input")

    axes[0, 1].imshow(fov_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(
        f"1. FOV mask — white {fov_mask.mean():.2%}"
    )

    axes[0, 2].imshow(fov_overlay)
    axes[0, 2].set_title("FOV overlay — green = inside FOV")

    axes[1, 0].imshow(artifact_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"2. Artifact mask — white {artifact_mask.mean():.2%}"
    )

    axes[1, 1].imshow(artifact_overlay)
    axes[1, 1].set_title("Artifact overlay — red = detected sticker")

    axes[1, 2].imshow(valid_overlay)
    axes[1, 2].set_title(
        f"3. Final valid overlay — orange = valid "
        f"({final_valid.mean():.2%})"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(panel_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fov_pct = 100.0 * float(fov_mask.mean())
    artifact_pct = 100.0 * float(artifact_mask.mean())
    valid_pct = 100.0 * float(final_valid.mean())
    expected_pct = 100.0 * float(expected_before_erosion.mean())

    print(f"Image: {image_path.name}")
    print(f"  FOV white pixels:                  {fov_pct:.2f}%")
    print(f"  Artifact white pixels:             {artifact_pct:.2f}%")
    print(f"  FOV minus artifact white pixels:   {expected_pct:.2f}%")
    print(f"  Final valid white pixels:          {valid_pct:.2f}%")
    print(f"  Function-reported excluded frac:   {reported_fraction:.4f}")
    print(
        "  Final-valid pixels outside FOV-minus-artifacts: "
        f"{int(unexpected_valid_pixels.sum())}"
    )
    print(f"  Diagnostic panel: {panel_path}")
    print(f"  FOV mask:         {fov_path}")
    print(f"  Artifact mask:    {artifact_path}")
    print(f"  Final valid mask: {valid_path}")
    print()

    return {
        "image": image_path.name,
        "fov_pct": fov_pct,
        "artifact_pct": artifact_pct,
        "valid_pct": valid_pct,
        "reported_fraction": reported_fraction,
        "panel": str(panel_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualise masks returned by the actual segmentation function."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("preprocessed_no_clahe"),
        help="Input image folder (default: preprocessed_no_clahe)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("actual_mask_diagnostics"),
        help="Output folder for masks and diagnostic panels."
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=DEFAULT_IDS,
        help="Image IDs without extensions."
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_dir.resolve()}"
        )

    print("\nACTUAL SEGMENTATION MASK DIAGNOSTIC")
    print("===================================")
    print(f"Input folder: {args.input_dir.resolve()}")
    print(f"Sticker detection enabled: {SEG_CFG.sticker_detect}")
    print("FOV mask is captured before sticker exclusion.")
    print("No source files or images will be modified.\n")

    results = []
    for image_id in args.images:
        image_path = find_image(args.input_dir, image_id)
        results.append(diagnose_one(image_path, args.output_dir))

    print(f"Completed {len(results)} image(s).")
    print(f"All outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
