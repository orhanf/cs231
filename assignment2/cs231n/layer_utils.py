from cs231n.layers import *
from cs231n.fast_layers import *

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_tanh_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a tanh, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, tanh_cache = tanh_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, tanh_cache, pool_cache)
  return out, cache


def conv_tanh_pool_backward(dout, cache):
  """
  Backward pass for the conv-tanh-pool convenience layer
  """
  conv_cache, tanh_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = tanh_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_stanh_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, an stanh, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, tanh_cache = stanh_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, stanh_cache, pool_cache)
  return out, cache


def conv_stanh_pool_backward(dout, cache):
  """
  Backward pass for the conv-stanh-pool convenience layer
  """
  conv_cache, stanh_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = stanh_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_relu_dropout_forward(x, w, b, p=0.5):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  and a dropout layer

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - p: 1/dropout probablility

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  s, relu_cache = relu_forward(a)
  out, drop_cache = dropout_forward(s, p)
  cache = (fc_cache, relu_cache, drop_cache)
  return out, cache


def affine_relu_dropout_backward(dout, cache):
  """
  Backward pass for the affine-relu-dropout convenience layer
  """
  fc_cache, relu_cache, drop_cache = cache
  ds = dropout_backward(dout, drop_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def conv_relu_conv_relu_pool_forward(x,
                                     w1, b1,
                                     w2, b2,
                                     conv_param1,
                                     conv_param2,
                                     pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a1, conv_cache1 = conv_forward_fast(x, w1, b1, conv_param1)
  s1, relu_cache1 = relu_forward(a1)
  a2, conv_cache2 = conv_forward_fast(s1, w2, b2, conv_param2)
  s2, relu_cache2 = relu_forward(a2)
  out, pool_cache = max_pool_forward_fast(s2, pool_param)
  cache = (conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache)
  return out, cache


def conv_relu_conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-conv-relu-pool convenience layer
  """
  conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache = cache
  ds2 = max_pool_backward_fast(dout, pool_cache)
  da2 = relu_backward(ds2, relu_cache2)
  dx2, dw2, db2 = conv_backward_fast(da2, conv_cache2)
  da1 = relu_backward(dx2, relu_cache1)
  dx1, dw1, db1 = conv_backward_fast(da1, conv_cache1)

  return dx1, dw1, db1, dx2, dw2, db2


