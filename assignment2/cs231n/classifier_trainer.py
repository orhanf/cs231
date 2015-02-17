import numpy as np
import scipy.optimize as optimize
from collections import OrderedDict

class ClassifierTrainer(object):
    """ The trainer class performs SGD with momentum on a cost function """
    def __init__(self):
        self.step_cache = {} # for storing velocities in momentum update

    def train(self, X, y, X_val, y_val,
              model, loss_function,
              reg=0.0,
              learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
              update='momentum', sample_batches=True,
              num_epochs=30, batch_size=100, acc_frequency=None,
              adarho=0.95, adaeps=1e-6,
              beta1=0.1, beta2=0.001,
              verbose=False):
        """
        Optimize the parameters of a model to minimize a loss function. We use
        training data X and y to compute the loss and gradients, and periodically
        check the accuracy on the validation set.

        Inputs:
        - X: Array of training data; each X[i] is a training sample.
        - y: Vector of training labels; y[i] gives the label for X[i].
        - X_val: Array of validation data
        - y_val: Vector of validation labels
        - model: Dictionary that maps parameter names to parameter values. Each
          parameter value is a numpy array.
        - loss_function: A function that can be called in the following ways:
          scores = loss_function(X, model, reg=reg)
          loss, grads = loss_function(X, model, y, reg=reg)
        - reg: Regularization strength. This will be passed to the loss function.
        - learning_rate: Initial learning rate to use.
        - momentum: Parameter to use for momentum updates.
        - learning_rate_decay: The learning rate is multiplied by this after each
          epoch.
        - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
        - sample_batches: If True, use a minibatch of data for each parameter update
          (stochastic gradient descent); if False, use the entire training set for
          each parameter update (gradient descent).
        - num_epochs: The number of epochs to take over the training data.
        - batch_size: The number of training samples to use at each iteration.
        - acc_frequency: If set to an integer, we compute the training and
          validation set error after every acc_frequency iterations.
        - verbose: If True, print status after each epoch.

        Returns a tuple of:
        - best_model: The model that got the highest validation accuracy during
          training.
        - loss_history: List containing the value of the loss function at each
          iteration.
        - train_acc_history: List storing the training set accuracy at each epoch.
        - val_acc_history: List storing the validation set accuracy at each epoch.
        """

        N = X.shape[1]

        if sample_batches:
            iterations_per_epoch = N / batch_size # using SGD
        else:
            iterations_per_epoch = 1 # using GD
        num_iters = num_epochs * iterations_per_epoch
        epoch = 0
        best_val_acc = 0.0
        best_model = {}
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        # some second order methods
        if update in ['lbfgs', 'cg']:
            vec_model, p_to_info = serialize_model(model)
            f = lambda m : loss_function(X, m, y, reg, p_to_info)[0]
            fprime = lambda m : loss_function(X, m, y, reg, p_to_info)[1]


            if update == 'lbfgs':
                opts = {'maxiter' : 15000,   # default value.
                        'disp'    : verbose, # non-default value.
                        'pgtol'   : 1e-05,   # default value
                        'epsilon' : 1e-08}   # default value.
                model, final_cost, add_ = optimize.fmin_l_bfgs_b(f,
                                                                 vec_model,
                                                                 fprime=fprime,
                                                                 **opts)
                print add_
            elif update == 'cg':
                opts = {'maxiter'    : None,    # default value.
                        'disp'       : verbose, # non-default value.
                        'gtol'       : 1e-5,    # default value.
                        'norm'       : np.inf,  # default value.
                        'full_output': True,    # non-default value.
                        'epsilon'    : 1.4901161193847656e-08}  # default value.
                model, final_cost, func_calls,\
                   grad_calls, warnflag = optimize.fmin_cg(f,
                                                           vec_model,
                                                           fprime=fprime,
                                                           **opts)
                warnmsg = ['Success.',
                           'The maximum number of iterations was exceeded.',
                           'Gradient and/or function calls were not changing. May indicate' +\
                           'that precision was lost, i.e., the routine did not converge.']
                print warnmsg[warnflag]

            model = deserialize_model(model, p_to_info)
            loss_history = [final_cost]

            # evaluate train accuracy
            if N > 1000:
                train_mask = np.random.choice(N, 1000)
                X_train_subset = X[train_mask]
                y_train_subset = y[train_mask]
            else:
                X_train_subset = X
                y_train_subset = y
            scores_train = loss_function(X_train_subset, model)
            y_pred_train = np.argmax(scores_train, axis=1)
            train_acc = np.mean(y_pred_train == y_train_subset)
            train_acc_history.append(train_acc)

            # evaluate val accuracy
            scores_val = loss_function(X_val, model)
            y_pred_val = np.argmax(scores_val, axis=1)
            val_acc = np.mean(y_pred_val ==  y_val)
            val_acc_history.append(val_acc)

            # keep track of the best model based on validation accuracy
            if val_acc > best_val_acc:
                # make a copy of the model
                best_val_acc = val_acc
                best_model = {}
                for p in model:
                    best_model[p] = model[p].copy()

        # first order methods
        else:
            for it in xrange(num_iters):
                if it % 100 == 0 and verbose:  print 'starting iteration ', it

                # get batch of data
                if sample_batches:
                    batch_mask = np.random.choice(N, batch_size)
                    X_batch = X[batch_mask]
                    y_batch = y[batch_mask]
                else:
                    # no SGD used, full gradient descent
                    X_batch = X
                    y_batch = y

                # evaluate cost and gradient
                cost, grads = loss_function(X_batch, model, y_batch, reg)
                loss_history.append(cost)

                # perform a parameter update
                for p in model:
                    # compute the parameter step
                    if update == 'sgd':
                        dx = -learning_rate * grads[p]
                    elif update == 'momentum':
                        if not p in self.step_cache:
                            self.step_cache[p] = np.zeros(grads[p].shape)
                        self.step_cache[p] = momentum * self.step_cache[p] - \
                                             (learning_rate * grads[p])
                        dx = self.step_cache[p]
                    elif update == 'rmsprop':
                          decay_rate = 0.99 # you could also make this an option
                          if not p in self.step_cache:
                              self.step_cache[p] = np.zeros(grads[p].shape)
                          self.step_cache[p] = (decay_rate * self.step_cache[p]) + \
                                               ((1. - decay_rate) * (grads[p]**2))
                          dx = - learning_rate * grads[p] / np.sqrt(self.step_cache[p]+ 1e-8)
                    elif update == 'adadelta':
                          if not p in self.step_cache:
                                self.step_cache[p] = [np.zeros(grads[p].shape),\
                                                      np.zeros(grads[p].shape)]
                          self.step_cache[p][0] = (adarho * self.step_cache[p][0]) + \
                                                  ((1. - adarho) * (grads[p]**2))
                          dx = - np.sqrt((self.step_cache[p][1] + adaeps)/(self.step_cache[p][0] + adaeps)) * grads[p]
                          self.step_cache[p][1] = adarho * self.step_cache[p][1] + \
                                                  (1 - adarho) * dx**2
                    elif update == 'adam':
                          if not p in self.step_cache:
                                self.step_cache[p] = [np.zeros(grads[p].shape),
                                                      np.zeros(grads[p].shape),
                                                      np.zeros(grads[p].shape)]
                          mg_t = (1.0 - beta1) * self.step_cache[p][1] + beta1 * grads[p]
                          r_t = (1.0 - beta2) * self.step_cache[p][0] + beta2 * grads[p]**2
                          dx = learning_rate * -mg_t / (np.sqrt(r_t) + adaeps)
                          self.step_cache[p][0] = r_t
                          self.step_cache[p][1] = mg_t
                    else:
                          raise ValueError('Unrecognized update type "%s"' % update)

                    # update the parameters
                    model[p] += dx

                # every epoch perform an evaluation on the validation set
                first_it = (it == 0)
                epoch_end = (it + 1) % iterations_per_epoch == 0
                acc_check = (acc_frequency is not None and it % acc_frequency == 0)
                if first_it or epoch_end or acc_check:
                    if it > 0 and epoch_end:
                        # decay the learning rate
                        learning_rate *= learning_rate_decay
                        epoch += 1

                # evaluate train accuracy
                if N > 1000:
                    train_mask = np.random.choice(N, 1000)
                    X_train_subset = X[train_mask]
                    y_train_subset = y[train_mask]
                else:
                    X_train_subset = X
                    y_train_subset = y
                scores_train = loss_function(X_train_subset, model)
                y_pred_train = np.argmax(scores_train, axis=1)
                train_acc = np.mean(y_pred_train == y_train_subset)
                train_acc_history.append(train_acc)

                # evaluate val accuracy
                scores_val = loss_function(X_val, model)
                y_pred_val = np.argmax(scores_val, axis=1)
                val_acc = np.mean(y_pred_val ==  y_val)
                val_acc_history.append(val_acc)

                # keep track of the best model based on validation accuracy
                if val_acc > best_val_acc:
                    # make a copy of the model
                    best_val_acc = val_acc
                    best_model = {}
                    for p in model:
                        best_model[p] = model[p].copy()

                # print progress if needed
                if verbose:
                    print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                            % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

        if verbose:
            print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
        # return the best model and the training history statistics
        return best_model, loss_history, train_acc_history, val_acc_history


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



