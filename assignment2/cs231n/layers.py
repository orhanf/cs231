import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = np.dot(x.reshape(x.shape[0],-1),w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape(x.shape[0],-1).T, dout)
  db = dout.sum(axis=0)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0., x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout * sp.sign(np.maximum(0., x))
  return dx


def prelu_forward(x, alpha):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape
  - alpha: Negative slope parameter,same shape with x

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0., x) + alpha * np.minimum(0., x)
  cache = (x, alpha)
  return out, cache


def prelu_backward(dout, cache):
  """
  Computes the backward pass for a layer of parametrized rectified
  linear units (PReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  - da: Gradient with respect to alpha
  """
  x, alpha = cache
  dx = dout * (sp.sign(np.maximum(0., x)) - alpha * sp.sign(np.minimum(0., x)))
  da = dout * np.minimum(x, 0.)
  return dx, da


def tanh_forward(x):
  """
  Computes the forward pass for a layer of hyperbolic tangent units (tanh).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.tanh(x)
  cache = x
  return out, cache


def tanh_backward(dout, cache):
  """
  Computes the backward pass for a layer of hyperbolic tangent units (tanh).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = (1. -(np.tanh(x)**2)) * dout
  return dx


def stanh_forward(x):
  """
  Computes the forward pass for a layer of scaled hyperbolic tangent units (tanh).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = 1.7159 * np.tanh(2./3. * x)
  cache = x
  return out, cache


def stanh_backward(dout, cache):
  """
  Computes the backward pass for a layer of scaled hyperbolic tangent units (tanh).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = (1.7159 * 2./3. * (1. - np.tanh(2./3. * x) ** 2)) * dout
  return dx


def dropout_forward(x, p=0.5):
  """
  Computes the forward pass for a dropout layer.

  Input:
  - x: Inputs, of any shape
  - p: Probability of keeping unit active, higher means less dropout
  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  mask = (np.random.rand(*x.shape) < p) / p
  out = x * mask
  cache = (mask, p)
  return out, cache


def dropout_backward(dout, cache):
  """
  Computes the backward pass for dropout layer.

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  mask = cache[0]
  p = cache[1]
  dx = dout * sp.sign(mask) / p
  return dx


def bn_transform_forward(x, gamma, beta, eps=1e-8):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  mu = np.mean(x, axis=0)
  var = np.var(x, axis=0)
  x_hat = (x - mu) / np.sqrt(var + eps)
  out = gamma * x_hat + beta
  cache = (x, gamma, beta, mu, var, x_hat)
  return out, cache


def bn_transform_backward(dout, cache, eps=1e-8):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, gamma, beta, mu, var, x_hat = cache
  n_samples = x.shape[0]
  dx_hat = dout * gamma
  dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
  dmu = (np.sum(dx_hat * -(1. / np.sqrt(var + eps)), axis=0)) + \
         dvar * (np.sum(-2. * (x - mu), axis=0) / n_samples )

  dx = dx_hat * (1. / np.sqrt(var + eps) ) + \
       dvar * (2. * (x - mu) / n_samples) + \
       dmu * (1. / n_samples)

  dgamma = np.sum(dout * x_hat, axis=0)
  dbeta = np.sum(dout, axis=0)

  return dx, dgamma, dbeta


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']

  Hp = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
  Wp = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

  out = np.zeros((N, F, Hp, Wp))
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  '''
  print out.shape
  for n in xrange(N):
      for f in xrange(F):
          cout = np.zeros((Hp, Wp))
          for c in xrange(C):
              xp  = np.pad(x[n, c, :, :], 1, 'constant')
              #xp  = x[n, c, :, :]
              flt = np.rot90(w[f, c, :, :], 2)
              #flt = w[f, c, :, :]
              cv = signal.convolve2d(xp, flt, mode='same')

              cout += cv[:stride , :stride] + b[f]
          out[n, f, :, :] = cout
  '''

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

