import numpy as np

from connectomics.data.augmentation.augment_ops import smart_normalize
from connectomics.data.augmentation.transforms import RandViewDropoutd


def test_smart_normalize_can_scale_views_independently():
    first = np.linspace(10.0, 20.0, 24, dtype=np.float32).reshape(2, 3, 4)
    second = np.linspace(1000.0, 2000.0, 24, dtype=np.float32).reshape(2, 3, 4)
    result = smart_normalize(np.stack((first, second)), "0-1", channelwise=True)
    np.testing.assert_allclose(result[:, 0, 0, 0], [0.0, 0.0])
    np.testing.assert_allclose(result[:, -1, -1, -1], [1.0, 1.0])


def test_view_dropout_drops_exactly_one_complete_channel():
    transform = RandViewDropoutd(keys=["image"], prob=1.0, channels=(0, 1), fill_value=0.0)
    transform.set_random_state(seed=7)
    result = transform({"image": np.ones((2, 3, 4, 5), dtype=np.float32)})["image"]
    channel_sums = result.reshape(2, -1).sum(axis=1)
    assert sorted(channel_sums.tolist()) == [0.0, 60.0]

