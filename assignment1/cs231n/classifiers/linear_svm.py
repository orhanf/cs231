import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    margins = scores - correct_class_score + 1
    sum_of_margins = sum([1 for ii, x in enumerate(margins) if ii!=y[i] and x>0])

    dW[y[i],:] += -sum_of_margins * X[:,i]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j,:] += X[:,i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += 0.5 * reg * W


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[0]
  num_train = X.shape[1]

  WX = np.dot(W,X)
  Y  = np.eye(num_classes)[:, y]
  score_y = np.sum(WX * Y, axis=0)

  # loss
  max_margins = np.maximum(0., WX - score_y + 1) * (1-Y)
  loss = np.sum(max_margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  # gradient
  max_margins[max_margins>0] = 1.
  max_margins[y,range(num_train)] = -max_margins.sum(axis=0)
  dW = np.dot(max_margins, X.T)
  dW /= num_train
  dW += 0.5 * reg * W

  return loss, dW
