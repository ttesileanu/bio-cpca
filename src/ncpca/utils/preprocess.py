"""Various data preprocessing routines."""
from PIL import Image


def resize_and_crop(img, size=(100, 100), crop_type="middle"):
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally
    # depending on the ratio
    if ratio > img_ratio:
        img = img.resize(
            (size[0], int(round(size[0] * img.size[1] / img.size[0]))), Image.ANTIALIAS
        )
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, img.size[0], size[1])
        elif crop_type == "middle":
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)),
            )
        elif crop_type == "bottom":
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize(
            (int(round(size[1] * img.size[0] / img.size[1])), size[1]), Image.ANTIALIAS
        )
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, size[0], img.size[1])
        elif crop_type == "middle":
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1],
            )
        elif crop_type == "bottom":
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]), Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    return img
