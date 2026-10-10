"""Check synthetic uniform image descriptors and error handling."""
import numpy as np
from PIL import Image
import pytest
from rasher.features import extract_features
from rasher.baseline import metrics


def test_uniform_red_image(tmp_path):
    path=tmp_path/'red.png';Image.new('RGB',(120,90),(255,0,0)).save(path)
    f=extract_features(path,2).reshape(4,8)
    np.testing.assert_allclose(f,np.tile([1,0,0,0,0,0,0,0],(4,1)))


def test_bad_grid(tmp_path):
    with pytest.raises(ValueError):extract_features(tmp_path/'absent.png',3)


def test_metric_units():
    m=metrics([0,2],[1,1]);assert m['mae']==1;assert m['rmse']==1;assert m['r2']==0
