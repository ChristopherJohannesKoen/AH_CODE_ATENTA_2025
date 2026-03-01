import pytest
from math_utils import factorial

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_positive():
    assert factorial(5) == 120

def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)


def test_factorial_non_integer():
    with pytest.raises(TypeError):
        factorial(3.5)
