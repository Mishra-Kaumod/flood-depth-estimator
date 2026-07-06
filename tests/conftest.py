import io
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def sample_image():
    """A small synthetic flood-ish image: brown lower half, grey upper half."""
    img = Image.new("RGB", (128, 128), (150, 150, 155))
    for y in range(64, 128):
        for x in range(128):
            img.putpixel((x, y), (120, 90, 60))
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()
