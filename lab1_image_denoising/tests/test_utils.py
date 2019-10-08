import sys
sys.path.append('../')

from src.utils import (
    neighbor_exists,
    get_neighbor_coordinate,
    edge_weight,
    node_weight,
)

from numpy import log


def test_neighbor_exists():
    assert not neighbor_exists(1, 1, 0, 0, 0)
    assert not neighbor_exists(2, 1, 0, 0, 0)
    assert not neighbor_exists(2, 1, 0, 0, 1)
    assert not neighbor_exists(2, 1, 0, 0, 2)
    assert neighbor_exists(2, 1, 0, 0, 3)
    assert neighbor_exists(2, 1, 1, 0, 1)


def test_get_neighbor_coordinate():
    assert get_neighbor_coordinate(0, 0, 3) == (1, 0)
    assert get_neighbor_coordinate(1, 0, 1) == (0, 0)


def test_edge_weight():
    assert edge_weight(0, 0, 1) == 0
    assert edge_weight(1, 0, 0) == 0
    assert edge_weight(1, 0, 1) == 1


def test_node_weight():
    assert node_weight(0, 0, 0.5) == -log(0.5)
    assert node_weight(0, 1, 1) == 0
