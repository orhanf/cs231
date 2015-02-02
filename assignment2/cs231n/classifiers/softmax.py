import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  num_samples = X.shape[1]
  num_classes = W.shape[0]
  Y  = np.eye(num_classes)[:, y]

  M = np.dot(W, X)
  M = M - np.max(M, axis=0)
  expM = np.exp(M)
  probs = expM / np.sum(expM, axis=0)
  log_probs = np.log(probs)

  G = (0.5 * reg) * np.sum(W*W)
  loss = -( 1.0 /num_samples ) * np.sum( Y * log_probs ) + G

  dW = -(1.0/num_samples) * np.transpose(np.dot(X, np.transpose(Y - probs))) + (reg * W);

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_samples = X.shape[1]
  num_classes = W.shape[0]
  Y  = np.eye(num_classes)[:, y]

  M = np.dot(W, X)
  M = M - np.max(M, axis=0)
  expM = np.exp(M)
  probs = expM / np.sum(expM, axis=0)
  log_probs = np.log(probs)

  G = (0.5 * reg) * np.sum(W*W)
  loss = -( 1.0 /num_samples ) * np.sum( Y * log_probs ) + G

  dW = -(1.0/num_samples) * np.transpose(np.dot(X, np.transpose(Y - probs))) + (reg * W);

  return loss, dW

def softmax_loss_b(W, b, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_samples = X.shape[1]
  num_classes = W.shape[0]
  Y  = np.eye(num_classes)[:, y]

  M = np.dot(W, X) + b
  M = M - np.max(M, axis=0)
  expM = np.exp(M)
  probs = expM / np.sum(expM, axis=0)
  log_probs = np.log(probs)

  G = (0.5 * reg) * np.sum(W*W)
  loss = -( 1.0 /num_samples ) * np.sum( Y * log_probs ) + G

  dW = -(1.0/num_samples) * np.transpose(np.dot(X, np.transpose(Y - probs))) + (reg * W);

  return loss, dW
