"""
Independent FOV diagnostic for dermoscopic images.

This tests ONLY the dermoscope field boundary.
Sticker detection and lesion segmentation are deliberately disabled.

Mask convention:
    White / True  = inside the dermoscopic field (valid)
    Black / False = outside the field (vignette / background)

Run from the project root:
    python diagnose_fov.py

Selected images:
    python diagnose_fov.py --images ISIC_0000177.png ISIC_0000879.png

Optional:
    python diagnose_fov.py --input-dir Images --output-dir fov_diagnostics
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Values from the active SegmentationConfig in src/config.py.
# Keep these synchronized if you later change that configuration.
FOV_REL_THR = 0.72
FOV_MIN_FRAC = 0.35
VIGNETTE_DARK_THRESH = 35
VIGNETTE_MIN_AREA_FRAC = 0.01
VIGNETTE_MARGIN_FRAC = 0.02

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def make_disk(radius):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def detect_fov_only(bgr):
    """Return raw FOV mask, eroded FOV mask, and diagnostic statistics."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or could not be decoded.")

    gray = cv2.GaussianBlur(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2
    ).astype(np.float32)

    h, w = gray.shape
    image_area = h * w
    ref = float(np.percentile(gray, 75))
    valid = np.ones((h, w), dtype=bool)

    # 1. Estimate the FOV using the convex hull of the largest bright region.
    bright = (gray >= FOV_REL_THR * ref).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright, connectivity=8
    )

    hull_accepted = False
    hull_fraction = 1.0

    if n > 1:
        largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest = (labels == largest_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            hull_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(
                hull_img,
                cv2.convexHull(max(contours, key=cv2.contourArea)),
                1,
            )
            hull = hull_img.astype(bool)
            excluded = ~hull
            hull_fraction = float(hull.mean())

            if FOV_MIN_FRAC < hull_fraction < 0.985:
                cy = max(2, int(0.03 * h))
                cx = max(2, int(0.03 * w))
                corners = [
                    hull[cy, cx],
                    hull[cy, w - 1 - cx],
                    hull[h - 1 - cy, cx],
                    hull[h - 1 - cy, w - 1 - cx],
                ]
                at_least_two_corners_outside = (
                    sum(not bool(c) for c in corners) >= 2
                )
                excluded_area_is_dark = (
                    excluded.any()
                    and float(gray[excluded].mean()) < 0.6 * ref
                )

                if at_least_two_corners_outside and excluded_area_is_dark:
                    valid = hull.copy()
                    hull_accepted = True

    # 2. Remove very dark connected components touching the image border.
    dark = (gray < VIGNETTE_DARK_THRESH).astype(np.uint8)
    n_dark, dark_labels, dark_stats, _ = (
        cv2.connectedComponentsWithStats(dark, connectivity=4)
    )

    dark_border_pixels_removed = 0
    for component_id in range(1, n_dark):
        x = int(dark_stats[component_id, cv2.CC_STAT_LEFT])
        y = int(dark_stats[component_id, cv2.CC_STAT_TOP])
        bw = int(dark_stats[component_id, cv2.CC_STAT_WIDTH])
        bh = int(dark_stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(dark_stats[component_id, cv2.CC_STAT_AREA])

        touches_border = (
            x == 0 or y == 0 or x + bw >= w or y + bh >= h
        )
        large_enough = area >= VIGNETTE_MIN_AREA_FRAC * image_area

        if touches_border and large_enough:
            component = (dark_labels == component_id).astype(np.uint8)
            component = cv2.dilate(
                component, np.ones((7, 7), dtype=np.uint8)
            ).astype(bool)
            before = int(valid.sum())
            valid &= ~component
            dark_border_pixels_removed += before - int(valid.sum())

    # Raw FOV is captured BEFORE sticker handling (which is absent here).
    raw_fov = valid.copy()

    # 3. Apply the same small boundary erosion used by segmentation.
    if not valid.all():
        margin = max(2, round(VIGNETTE_MARGIN_FRAC * min(h, w)))
        eroded_fov = (
            cv2.erode(valid.astype(np.uint8), make_disk(margin)) > 0
        )
    else:
        eroded_fov = valid.copy()

    stats_out = {
        "height": h,
        "width": w,
        "p75_gray": ref,
        "bright_threshold": FOV_REL_THR * ref,
        "hull_accepted": hull_accepted,
        "hull_fraction": hull_fraction,
        "raw_fov_fraction": float(raw_fov.mean()),
        "eroded_fov_fraction": float(eroded_fov.mean()),
        "dark_border_pixels_removed": dark_border_pixels_removed,
    }
    return raw_fov, eroded_fov, stats_out


def make_overlay(bgr, mask, colour=(0, 220, 0), alpha=0.32):
    overlay = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[:] = colour
    overlay[mask] = cv2.addWeighted(
        bgr[mask], 1.0 - alpha, tint[mask], alpha, 0
    )
    return overlay


def process_image(image_path, output_dir):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    raw_fov, eroded_fov, info = detect_fov_only(bgr)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw_overlay = cv2.cvtColor(
        make_overlay(bgr, raw_fov), cv2.COLOR_BGR2RGB
    )
    eroded_overlay = cv2.cvtColor(
        make_overlay(bgr, eroded_fov, colour=(0, 165, 255)),
        cv2.COLOR_BGR2RGB,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    raw_mask_path = output_dir / f"{stem}_fov_mask.png"
    eroded_mask_path = output_dir / f"{stem}_fov_mask_eroded.png"
    diagnostic_path = output_dir / f"{stem}_fov_diagnostic.png"

    cv2.imwrite(str(raw_mask_path), raw_fov.astype(np.uint8) * 255)
    cv2.imwrite(
        str(eroded_mask_path), eroded_fov.astype(np.uint8) * 255
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"FOV-only diagnostic: {image_path.name}", fontsize=15)

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original image")

    axes[0, 1].imshow(raw_fov, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(
        f"Raw FOV mask — white = valid ({info['raw_fov_fraction']:.1%})"
    )

    axes[1, 0].imshow(raw_overlay)
    axes[1, 0].set_title("Raw FOV overlay — green = valid field")

    axes[1, 1].imshow(eroded_overlay)
    axes[1, 1].set_title(
        "After safety erosion — orange = valid "
        f"({info['eroded_fov_fraction']:.1%})"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(diagnostic_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    info.update({
        "image": image_path.name,
        "diagnostic": str(diagnostic_path),
        "raw_mask": str(raw_mask_path),
        "eroded_mask": str(eroded_mask_path),
    })
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Check FOV detection independently of sticker detection."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("preprocessed"),
        help="Folder containing original images (default: Images)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fov_diagnostics"),
        help="Folder for diagnostic panels and masks."
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Optional exact filenames to inspect."
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum images when --images is not supplied (default: 20)."
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {args.input_dir.resolve()}. "
            "Run from the project root or specify --input-dir."
        )

    if args.images:
        paths = []
        for filename in args.images:
            path = args.input_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path.resolve()}")
            paths.append(path)
    else:
        paths = sorted(
            p for p in args.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if args.limit > 0:
            paths = paths[:args.limit]

    if not paths:
        raise RuntimeError(f"No supported images found in {args.input_dir}.")

    print("\nFOV-ONLY DIAGNOSTIC")
    print("===================")
    print("Sticker detection: OFF")
    print("Lesion segmentation: OFF")
    print("White = inside FOV; black = outside FOV\n")

    succeeded = 0
    for path in paths:
        try:
            result = process_image(path, args.output_dir)
            succeeded += 1
            print(f"Image: {result['image']}")
            print(f"  Bright-region hull accepted: {result['hull_accepted']}")
            print(f"  Raw FOV white pixels: {result['raw_fov_fraction']:.2%}")
            print(
                f"  Eroded FOV white pixels: "
                f"{result['eroded_fov_fraction']:.2%}"
            )
            print(
                f"  Pixels excluded by dark-border fallback: "
                f"{result['dark_border_pixels_removed']}"
            )
            print(f"  Diagnostic: {result['diagnostic']}\n")
        except Exception as exc:
            print(f"ERROR: {path.name}: {exc}\n")

    print(f"Finished: {succeeded}/{len(paths)} images processed.")
    print(f"Results saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
