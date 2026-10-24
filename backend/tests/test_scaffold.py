from backend.app.models import Component
from backend.app.config import CONFIG
import pytest

def test_models_and_config():
    assert abs(sum(CONFIG['detection']['weights'].values()) - 1) < 1e-6
    assert Component(designator='R1').roi is None
    with pytest.raises(ValueError):
        Component(designator='R1', roi=[0, 0, -1, 5])

def test_synthetic_generator_is_deterministic():
    import numpy as np
    from backend.vision.synthetic import fixture, augment, PARTS
    image = fixture()
    first, _ = augment(image, PARTS, ['C1'], 5)
    second, _ = augment(image, PARTS, ['C1'], 5)
    assert np.array_equal(first, second)
    assert len(PARTS) == 6
