import pytest
import numpy as np
from expipe_io_neuro.openephys.tools import _start_from_zero_time, _zeros_to_nan, _cut_to_same_len

def test_start_from_zero_time():
    t = np.arange(11)
    t[0] = 10
    t[1] = 0.0
    x = np.arange(11)
    y = np.arange(11, 22)
    t_, (x_, y_) = _start_from_zero_time(t, x, y)
    assert np.array_equal(t_, t[1:])
    assert np.array_equal(x_, x[1:])
    assert np.array_equal(y_, y[1:])
    # not equal length
    with pytest.raises(ValueError):
        y = np.arange(11, 23)
        t_, (x_, y_) = _start_from_zero_time(t, x, y)
    # Multiple starting times
    with pytest.raises(ValueError):
        y = np.arange(11, 22)
        t[4] = 0.
        t_, (x_, y_) = _start_from_zero_time(t, x, y)
    # no starting time
    with pytest.raises(ValueError):
        y = np.arange(11, 22)
        t[4] = 1.
        t[1] = 1.
        t_, (x_, y_) = _start_from_zero_time(t, x, y)
    # not equal length in time
    with pytest.raises(ValueError):
        t = np.arange(12)
        t[0] = 10
        t[1] = 0.0
        t_, (x_, y_) = _start_from_zero_time(t, x, y)

def test_cut_to_same_len():
    t = np.arange(12)
    x = np.arange(11)
    y = np.arange(11, 24)
    t_, x_, y_ = _cut_to_same_len(t, x, y)
    assert np.array_equal(t_, t[:-1])
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y[:-2])
