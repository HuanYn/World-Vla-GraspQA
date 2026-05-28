from PIL import Image

from world_vla_graspqa.utils.visualization import (
    draw_scene_annotation,
    save_annotated_scene,
)


def test_draw_scene_annotation_returns_same_size_image():
    image = Image.new("RGB", (100, 80), color=(255, 255, 255))
    annotation = {
        "objects": [
            {
                "name": "red cube",
                "bbox": [10, 20, 40, 50],
            }
        ]
    }

    annotated_image = draw_scene_annotation(image, annotation)

    assert annotated_image.size == image.size
    assert annotated_image.mode == "RGB"


def test_save_annotated_scene_creates_output_file(tmp_path):
    image = Image.new("RGB", (100, 80), color=(255, 255, 255))
    annotation = {
        "objects": [
            {
                "name": "red cube",
                "bbox": [10, 20, 40, 50],
            }
        ]
    }
    output_path = tmp_path / "visualizations" / "annotated.png"

    save_annotated_scene(image, annotation, output_path)

    assert output_path.exists()
