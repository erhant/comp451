from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, hidden layer dimension of H, another hidden layer of dimension H, and performs 
    classification over C classes. We train the network with a softmax loss function 
    and L2 regularization on the weight matrices. The network uses a ReLU nonlinearity 
    after the first and second fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-3):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H)
        b2: Second layer biases; has shape (H,)
        W3: Third layer weights; has shape (H, C)
        b3: Third layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Relu
        ReLU = lambda x : np.maximum(0, x) 
        def ReLUgrad(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x # we chose the gradient at 0 to be 0 for the sake of simplicity
        
        #X : Inputs; has shape (N, D)
        #W1: First layer weights; has shape (D, H)
        #b1: First layer biases; has shape (H,)
        #W2: Second layer weights; has shape (H, H)
        #b2: Second layer biases; has shape (H,)
        #W3: Third layer weights; has shape (H, C)
        #b3: Third layer biases; has shape (C,)     
        
        FC_XW1 = np.dot(X, W1) + b1 # (N, D)(D, H) = (N, H), FC stands for fully-connected
        H1 = ReLU(FC_XW1) # (N, H)
        FC_H1W2 = np.dot(H1, W2) + b2 # (N, H)(H, H) = (N, H), FC stands for fully-connected
        H2 = ReLU(FC_H1W2) # (N, H)
        scores = np.dot(H2, W3) + b3  # these are the predictions (y_pred as we see mostly) (N, H)(H, C) = (N, C)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # scores are (N, C) shape
        scores -= np.max(scores, axis=1, keepdims=True) # avoids overflows, still (N, C)
        softmax_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # (N, C) / (N, 1) = (N, C)
        loss = np.sum(-np.log(softmax_scores[np.arange(N), y])) # losses are treated as probabilities. Also, this is a better version than my first implementation in softmax.py
        loss /= N
        loss += reg*(np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)) # regularization for W1 and W2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #X : Inputs; has shape (N, D)
        #W1: First layer weights; has shape (D, H)
        #b1: First layer biases; has shape (H,)
        #W2: Second layer weights; has shape (H, H)
        #b2: Second layer biases; has shape (H,)
        #W3: Third layer weights; has shape (H, C)
        #b3: Third layer biases; has shape (C,)    
        #scores: Outputs; has shape (N, C)
        
        # As given above, we are told to do:
        # input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax
        # X     - X.W1 + b1 = FC_XW1    - H1   - H1.W2 + b2 = FC_H1W2  - H2   - H2.W3 + b3 = scores   - softmax_scores
        #(N, H) - (N, H)(H, H) = (N, H) -(N, H)- (N, H)(H, H) = (N, H) -(N, H)- (N, H)(H, C) = (N, C) - (N, C)
        # Now we go backwards, first the gradient of softmax
        dsoftmax_scores = softmax_scores # for naming purposes
        dsoftmax_scores[np.arange(N) ,y] -=  1 # (N, C)
        dsoftmax_scores /= N # (N, C)
        
        # Downstream = Upstream * Local
        # W3 and b3 (has to be (H, C) and (H, ))
        dW3 = H2.T.dot(dsoftmax_scores) # (N, H).T (N, C) = (H, N)(N, C) = (H, C) 
        db3 = dsoftmax_scores.sum(axis=0) # (C, )
        
        # W2 and b2 (has to be (H, H) and (H, ))
        dW2 = dsoftmax_scores.dot(W3.T) # (N, C)(C, H) = (N, H)
        dFC_H1W2 = dW2 * ReLUgrad(FC_H1W2) # (N, H) * (N, H) = (N, H)
        dW2 = H1.T.dot(dFC_H1W2) # (H, N)(N, H) = (H, H)
        db2 = dFC_H1W2.sum(axis = 0) # (H, )
        
        # W1 and b1 (has to be (N, H) and (H, ))
        dW1 = dFC_H1W2.dot(W2.T) # (N, H)(H, H) = (N, H)
        dFC_XW1 = dW1 * ReLUgrad(FC_XW1) # (N, H)* (N, H) = (N, H)
        dW1 = X.T.dot(dFC_XW1) # (D, N)(N, H) = (D, H)
        db1 = dFC_XW1.sum(axis = 0) # (H, )
        
        # Regularizations
        dW3 += reg * 2 * W3
        dW2 += reg * 2 * W2
        dW1 += reg * 2 * W1
        
        # Assign all of them to grads dictionary
        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        #print("Softmax scores: ",softmax_scores.shape,"\nInput X: ",X.shape,"\nOutput Scores: ",scores.shape)
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            for key in self.params:
                self.params[key] -= learning_rate * grads[key]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_pred = np.argmax(self.loss(X), axis = 1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
