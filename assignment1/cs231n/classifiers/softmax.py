import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(N):
    score = np.dot(X[i], W)
    score -= np.max(score) # for numerically stability
    exp_score = np.exp(score)
    sum_exp_score = np.sum(exp_score)
    probs = exp_score / sum_exp_score
    loss += -np.log(probs[y[i]])
    for j in xrange(num_class):
      dW[:,j] += (probs[j] - (y[i] == j)) * X[i,:]

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.dot(X, W) #(N,C)
  score -= np.max(score, axis=1).reshape((N, 1)) #for numerically stability
  exp_score = np.exp(score) #(N,C)
  sum_exp_score = np.sum(exp_score, axis=1) #(N,)
  gold_standard_score = exp_score[np.arange(N), y] #(N,)
  loss += np.sum(-np.log(gold_standard_score / sum_exp_score))
  probs = exp_score / sum_exp_score[:, None] #(N,C)
  ones = np.zeros((N,C)) #(N,C)
  ones[np.arange(N), y] = 1 #(N,C)
  dW += np.dot(X.T, probs - ones) #(D,C)

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

