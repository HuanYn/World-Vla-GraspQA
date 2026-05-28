from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sample_scenes"
OUTPUT_PATH = OUTPUT_DIR / "tabletop_dummy_scene.png"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (640, 480), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)

    # Table area
    draw.rectangle(
        (40, 80, 600, 430),
        fill=(220, 210, 190),
        outline=(120, 100, 80),
        width=4,
    )

    # Red cube
    draw.rectangle(
        (130, 230, 230, 330),
        fill=(220, 40, 40),
        outline=(120, 0, 0),
        width=3,
    )
    draw.text((145, 340), "red cube", fill=(0, 0, 0))

    # Blue bowl
    draw.ellipse(
        (350, 220, 500, 340),
        fill=(60, 120, 220),
        outline=(0, 50, 130),
        width=4,
    )
    draw.ellipse(
        (385, 250, 465, 315),
        fill=(180, 210, 255),
        outline=(0, 50, 130),
        width=2,
    )
    draw.text((385, 350), "blue bowl", fill=(0, 0, 0))

    # Yellow banana
    draw.arc(
        (240, 130, 390, 280),
        start=25,
        end=160,
        fill=(230, 200, 40),
        width=18,
    )
    draw.text((260, 150), "yellow banana", fill=(0, 0, 0))

    image.save(OUTPUT_PATH)
    print(f"[SampleScene] Saved sample scene to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
