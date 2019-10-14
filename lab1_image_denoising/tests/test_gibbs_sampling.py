import sys
sys.path.append('../src')

from src.gibbs_sampling import (
    get_labeling,
    almost_equal_labelings,
)

from numpy import ndarray, array


def test_get_labeling():
    sums_of_zero_labels = ndarray(
        shape=(2, 2), buffer=array([1, 3, 0, 4]), dtype=int)
    sums_of_unit_labels = ndarray(
        shape=(2, 2), buffer=array([3, 1, 4, 0]), dtype=int)
    true_rezult = ndarray(shape=(2, 2), buffer=array([1, 0, 1, 0]), dtype=int)
    labeling = get_labeling(sums_of_zero_labels, sums_of_unit_labels)
    assert (labeling == true_rezult).all()


def test_almost_equal_labelings():
    labeling1 = ndarray(shape=(2, 2), buffer=array([1, 0, 0, 0]), dtype=int)
    labeling2 = ndarray(shape=(2, 2), buffer=array([1, 0, 1, 0]), dtype=int)
    assert almost_equal_labelings(labeling1, labeling2, 25)
    assert not almost_equal_labelings(labeling1, labeling2, 5)
