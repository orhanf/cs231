import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import OrderedDict
from softmax import softmax_loss_b as softmax_loss
def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0, p_to_info=None):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """
  if p_to_info:
      model= deserialize_model(model, p_to_info)

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  a1 = np.maximum(0., np.dot(X, W1) + b1)
  a1mask = scipy.sign(a1)
  scores = np.dot(a1, W2) + b2

  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  num_samples = scores.shape[0]
  num_classes = scores.shape[1]
  Y = np.eye(num_classes)[y, :]

  scores = scores - np.max(scores, axis=1)[:, None]
  expM = np.exp(scores)
  probs = expM / np.sum(expM, axis=1)[:, None]
  log_probs = np.log(probs)

  G = (0.5 * reg) * (np.sum(W1*W1) + np.sum(W2*W2))
  loss = -( 1.0 /num_samples ) * np.sum( Y * log_probs ) + G

  # compute the gradients
  d3 = probs - Y
  d2 = np.dot(d3, W2.T) * a1mask
  grads = {}
  grads['W2'] = (1./num_samples) * np.dot(a1.T, d3) + (reg * W2)
  grads['b2'] = (1./num_samples) * np.sum(d3,axis=0)
  grads['W1'] = (1./num_samples) * np.dot(X.T, d2) + (reg * W1)
  grads['b1'] = (1./num_samples) * np.sum(d2,axis=0)

  if p_to_info:
    grads, _ = serialize_model(grads, p_to_info)

  return loss, grads

def serialize_model(model, p_to_info=None):
    """
    vectorizes the whole model parameters
    model is a dictionaty of actual parameters
    returns a vector of parameters and
    a dictionary from parameter name to (shape, startIdx)
    """
    vec_model = np.array([],dtype=np.float32)
    if not p_to_info:
        p_to_info = OrderedDict()
        idx = 0
        for p, v in model.iteritems():
            p_to_info[p] = (v.shape, idx)
            idx += np.prod(v.shape)
            vec_model = np.concatenate((vec_model, v.flatten()))
    else:
        for p, v in p_to_info.iteritems():
            vec_model = np.concatenate((vec_model, model[p].flatten()))

    return vec_model, p_to_info


def deserialize_model(vec_model, p_to_info):
    """
    write me
    """
    model = OrderedDict()
    for p, v in p_to_info.iteritems():
        shapeThis = v[0]
        startIdx = v[1]
        endIdx = startIdx + np.prod(shapeThis)
        model[p] = vec_model[startIdx:endIdx].reshape(shapeThis)
    return model

