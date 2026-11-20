import numpy as np
import pytest
from aggregation import cluster_boot


def test_finite_filter_and_scene_clusters():
    mean, lower, upper = cluster_boot([1., 3., np.nan, 9.], ['a', 'a', 'b', 'b'])
    assert mean == pytest.approx(13/3)
    assert lower == 2.
    assert upper == 9.


def test_empty_and_invalid_inputs():
    assert all(np.isnan(value) for value in cluster_boot([np.nan], ['a']))
    with pytest.raises(ValueError):
        cluster_boot([1, 2], ['a'])
