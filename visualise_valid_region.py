
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.config import OUT_NO_CLAHE_DIR, SEG_CFG
from src.segmentation import detect_vignette


OUTPUT_DIR = Path("valid_region_diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return

    # Obtain all three masks.
    valid, vignette_frac, fov, artifacts = (
        detect_vignette(
            image,
            SEG_CFG,
            return_masks=True
        )
    )

    # -------------------------------------------------
    # Create RGB visualizations
    # -------------------------------------------------

    original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Artifact overlay: red
    artifact_overlay = original.copy()

    artifact_overlay[artifacts] = [255, 0, 0]

    # Valid region overlay: green
    valid_overlay = original.copy()

    green = np.zeros_like(original)
    green[:, :] = [0, 255, 0]

    valid_overlay[valid] = (
        0.65 * original[valid] +
        0.35 * green[valid]
    ).astype(np.uint8)

    # Final mask
    panels = [
        ("Original", original),
        ("FOV Mask", fov.astype(np.uint8) * 255),
        ("Artifact Mask", artifacts.astype(np.uint8) * 255),
        ("Valid Region", valid.astype(np.uint8) * 255),
        ("Artifact Overlay", artifact_overlay),
        ("Valid Overlay", valid_overlay)
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 11)
    )

    for ax, (title, panel) in zip(
        axes.flat,
        panels
    ):

        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray")
        else:
            ax.imshow(panel)

        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(Path(image_path).name)
    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        Path(image_path).stem + "_diagnostic.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    print("\nImage:", Path(image_path).name)

    print(
        "FOV fraction:",
        round(float(fov.mean()), 4)
    )

    print(
        "Artifact fraction:",
        round(float(artifacts.mean()), 4)
    )

    print(
        "Valid fraction:",
        round(float(valid.mean()), 4)
    )

    print(
        "Artifact pixels:",
        int(artifacts.sum())
    )

    print("Saved:", output_path)


if __name__ == "__main__":

    test_ids = [
        "ISIC_0000102",
        "ISIC_0000279",
        "ISIC_0000599",
        "ISIC_0000618",
        "ISIC_0000702",
        "ISIC_0000758",
        "ISIC_0000777",
        "ISIC_0000901",
        "ISIC_0000879"
    ]

    for image_id in test_ids:

        image_path = (
            OUT_NO_CLAHE_DIR /
            f"{image_id}.png"
        )

        process_image(image_path)
