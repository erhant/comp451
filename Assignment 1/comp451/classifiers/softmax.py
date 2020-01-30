from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
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
    - regtype: Regularization type: L1 or L2

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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # probability is calculated as: for image x_i where y_i = k
    # P(Y = k | X = x_i) = (e^{s_k})/(sum of all e^{s_j} over j)
    # We have s_k over e, because softmax accepts them as unnormalized log probabilities 
    # to deal with unnormalization, we divide by sum, therefore normalize the data
    
    # Then we find L_i = -log P(Y=y_i|X=x_i)
    # We sum those L_i's to find the total loss
    
    num_train = X.shape[0] # (500, 3073)
    num_classes = W.shape[1] # (3073, 10)

    for i in range(num_train):
        logits = X[i].dot(W) # unnormalized log probabilities (1, 3073).(3073, 10) = (1, 10)
        logits -= np.max(logits) # we do this to avoid overflowing. Highest value becomes 0. (see stanford notes) (1, 10) 
        probabilities = np.exp(logits) / np.sum(np.exp(logits)) # (1, 10) actually (10, )
        loss += -np.log(probabilities[y[i]]) # (1), this is the loss 
        
        #Gradient https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        for k in range(num_classes):
            probability_k = probabilities[k]
            dW[:, k] += (probability_k - (k == y[i])) * X[i] # (3073, 10) = (:, 10) + (1) () (500, 3072) 
            
    #Regularize
    R = 0
    Rgrad = 0
    if regtype == 'L2':
    #L2 reg
        R = reg * np.sum(W * W) # L2 reg R(W) = sum over W_{k,l}^2
        Rgrad = 2 * reg * W
    else:
    #L1 Reg
        R = reg * np.sum(np.abs(W)) # L1 reg R(W) = sum over |W_{k,l}|
        Rgrad = reg * W # not sure of this!
        
    # Normalize and regularize
    loss /= num_train
    loss += R
    dW /= num_train
    dW += Rgrad
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # (500, 3073)
    num_classes = W.shape[1] # (3073, 10)
    
    logits = X.dot(W) # (500 ,3073).(3073, 10) = (500, 10)
    logits -= np.max(logits, axis = 1, keepdims=True) # (500, 10) - (500,1) = (500, 10) broadcast
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True) # (500, 10) / (500,1)  = (500, 10) broadcast
    # Imagine this:
    # we have indexes for class like [1, 6, 2, 9, ...] etc
    # we create a base index array like [0, 1, 2, 3, ...]
    # and then combine them as columns
    # [[0, 1],
    #  [1, 6],
    #  [2, 2],
    #  ...
    indexer = np.arange(y.shape[0]).T.reshape(y.shape[0],1) # column vector like [0, 1, 2, 3, ..., y.shape[0]-1)
    indexerStack = np.hstack((indexer, y[:,np.newaxis])) # a 2 column vector where first column is the one above and second is y
    # then use this to choose correct column per row as seen below
    loss = -np.log(probabilities[indexerStack[:,0],indexerStack[:,1]]) # (500, 1) 
    loss = np.sum(loss) # (1)
    
    #Gradient
    # dW: (3073, 10)
    # X: (500, 3073)
    # Y: (500, )
    # probabilities: (500, 10)
    #print("dW: ", dW.shape,"\nProbabilities: ",probabilities.shape,"\nX :",X.shape,"\ny :",y.shape)
    # dW[:, k] += (probabilities[k] - (k == y[i])) * X[i] # (3073, 10) = (:, 10) + (1) () (500, 3072) for every i
    smt = np.exp(logits) / np.matrix(np.sum(np.exp(logits), axis=1)).T
    smt[np.arange(num_train),y] -= 1
    dW = X.T.dot(smt)
            
    #Regularize
    R = 0
    Rgrad = 0
    if regtype == 'L2':
    #L2 reg
        R = reg * np.sum(W * W) # L2 reg R(W) = sum over W_{k,l}^2
        Rgrad = 2 * reg * W
    else:
    #L1 Reg
        R = reg * np.sum(np.abs(W)) # L1 reg R(W) = sum over |W_{k,l}|
        Rgrad = reg * W # not sure of this!
        
    loss /= num_train
    loss += R
    dW /= num_train
    dW += Rgrad
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
