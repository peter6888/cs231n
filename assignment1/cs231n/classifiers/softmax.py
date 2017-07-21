import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #forward pass
  num_train, num_feature = X.shape
  num_class = W.shape[1]
  L = 0
  for i in range(num_train):
        predict_i = X[i, :].dot(W)
        # To imrove numeric instability, normalize by remove a max value.
        # Brief prove below:
        # exp(x_j+m)/(exp(x_1+m)+exp(x_2+m)+...+exp(x_n+m)) 
        #    = exp(x_j) * exp(m) / exp(m) * (exp(x_1) + exp(x_2) + ... + exp(x_n))
        #    = exp(x_j) / (exp(x_1) + exp(x_2) + ... + exp(x_n))
        predict_i -= np.max(predict_i)
        # L_i = -np.log(np.exp(predict_i[y[i]]) / np.sum(np.exp(predict_i)))
        # log(x/y) = log(x) - log(y)
        L_i = -predict_i[y[i]] + np.log(np.sum(np.exp(predict_i)))
        L += L_i
        
        p = np.exp(predict_i) / np.sum(np.exp(predict_i))
        
        #back prop for gradient
        for j in range(num_class):
            dW[:,j] += p[j] * X[i, :]
        dW[:,y[i]] -= X[i, :]
        
  loss = L/num_train + reg * np.sum(W*W)
  dW /= num_train
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  predict = X.dot(W) # predict.shape-->(N,C)
  num_train, num_class = predict.shape
    
  #normalize to avoid the numeric instability
  predict -= np.max(predict, axis=1).reshape(-1,1)
    
  correct_predict = predict[range(num_train), y] # correct_predict.shape --> (N,1)
  sum_exp_train = np.sum(np.exp(predict), axis=1).reshape(-1,1)

  # -log(x/y) = -log(x)+log(y), which here log(x)=correct_predict
  L = -correct_predict + np.log(np.sum(np.exp(predict), axis=1)) 
  loss = np.sum(L) / num_train + reg * np.sum(W*W)
    
  prob = np.exp(predict) / sum_exp_train
  prob[range(num_train),y] -= 1
  
  dW = np.dot(X.T, prob)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

