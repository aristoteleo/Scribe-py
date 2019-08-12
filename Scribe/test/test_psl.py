import numpy as np
import Scribe
from Scribe.psl import psl_py
import pytest

y = np.array([0.5396,0.0071,-0.0029,-0.0063,0.0011,-0.007,0.8656,0.547,0.1385,0.0379,0.1752,0.4897,0.011,-0.0023,-0.0061,0.0013,-0.0034,0.8759,0.4998,0.1496,0.0459,0.1747]).reshape(2,11).T

@pytest.mark.parametrize('y',[y] )
def test_psl_py(y):
    psl_py(y)