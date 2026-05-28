from PIL import Image

from world_vla_graspqa.utils.image import get_image_info, load_image


def test_load_image_converts_to_rgb(tmp_path):
    image_path = tmp_path / "test_image.png"
    image = Image.new("L", (32, 24), color=128)
    image.save(image_path)

    loaded_image = load_image(image_path)

    assert loaded_image.mode == "RGB"
    assert loaded_image.size == (32, 24)


def test_get_image_info_returns_metadata(tmp_path):
    image_path = tmp_path / "test_image.png"
    image = Image.new("RGB", (64, 48), color=(255, 255, 255))
    image.save(image_path)

    loaded_image = load_image(image_path)
    image_info = get_image_info(loaded_image, image_path)

    assert image_info["image_path"] == str(image_path)
    assert image_info["width"] == 64
    assert image_info["height"] == 48
    assert image_info["mode"] == "RGB"
