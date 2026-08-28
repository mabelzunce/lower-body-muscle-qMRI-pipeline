import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.utils import apply_bias_correction_2


def test_apply_bias_correction_accepts_tuple_shrink_factors():
    image = np.ones((72, 80, 96), dtype=np.float32)

    corrected = apply_bias_correction_2(image, (3, 8, 8))

    assert corrected.shape == image.shape
    assert corrected.dtype == np.float32
