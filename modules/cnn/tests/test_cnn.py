import numpy as np
import pytest
from modules.cnn.cnn_dev import conv2d_naive, Conv2D
from tinytorch.core.tensor import Tensor

def test_conv2d_naive_small():
    input = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    kernel = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.float32)
    expected = np.array([
        [1*1+2*0+4*0+5*(-1), 2*1+3*0+5*0+6*(-1)],
        [4*1+5*0+7*0+8*(-1), 5*1+6*0+8*0+9*(-1)]
    ], dtype=np.float32)
    output = conv2d_naive(input, kernel)
    assert np.allclose(output, expected), f"conv2d_naive output incorrect!\nExpected:\n{expected}\nGot:\n{output}"

def test_conv2d_layer_shape():
    x = Tensor(np.ones((5, 5), dtype=np.float32))
    conv = Conv2D((3, 3))
    out = conv(x)
    assert out.shape == (3, 3), f"Conv2D output shape incorrect! Expected (3, 3), got {out.shape}" 