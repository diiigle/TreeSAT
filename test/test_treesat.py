from timeit import timeit
import pytest
import time
from TreeSAT import SATTileTree
import numpy as np

@pytest.fixture
def input():
    return np.random.rand(3,5,7,1)

def test_construction():
    input = np.random.rand(5,10,20,1)
    print(input.shape)
    tree = SATTileTree(input)
    sat = tree.convert_dense()
    print(tree.size())
    print(sat.shape, sat)
    assert (tree.shape == input.shape).all()
    assert (tree.shape == sat.shape).all


    assert np.isclose(sat[-1,-1,-1,-1], np.sum(input))
    assert np.isclose(tree.query_singular(tree.shape[2]-1,tree.shape[1]-1,tree.shape[0]-1), sat[-1,-1,-1,-1])

def test_tile_sizes(input):
    sizes = []
    timings = []
    for tile_size in [4,8,10,16,25,32,48,50,64,128]:
        start = time.time()
        tree = SATTileTree(input, tile_size)
        elapsed = time.time() - start
        sizes.append(tree.size())
        timings.append(elapsed)

    print(sizes)
    with np.printoptions(precision=2):
        print(np.array(timings))
    assert False

def test_sparsities(input):
    sizes = []
    timings = []
    tile_size = 50
    for x in range((input.shape[2]//tile_size)+1):
        print(x)
        data = input.copy()
        data[:,:,:x*tile_size] = 0.0
        start = time.time()
        tree = SATTileTree(data, tile_size)
        elapsed = time.time() - start
        sizes.append(tree.size())
        timings.append(elapsed)

    print(sizes)
    with np.printoptions(precision=2):
        print(np.array(timings))
    assert False

def test_single_query_times(input):
    tree = SATTileTree(input, 3)

    for x in range(50):
        for y in range(50):
            for z in range(50):
                val = tree.query_singular(x,y,z)
                print(x,y,z, val)
                assert np.isclose(val, np.sum(input[:z+1, :y+1, :x+1])).all()
    #print(timeit('tree.query_average(*np.s_[:1,:2,:3])', globals={'tree':tree, 'np': np}))
