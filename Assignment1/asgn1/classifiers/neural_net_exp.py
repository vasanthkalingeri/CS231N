import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNetExp(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)# * np.sqrt(1.0 / input_size)
    self.params['b1'] = np.zeros(hidden_size)# + 2.5 #To prevent sleepy Relu
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)# * np.sqrt(1.0 / hidden_size)
    self.params['b2'] = np.zeros(output_size)# + 2.5 #To prevent sleepy Relu

  def loss(self, X, y=None, reg=0.0, prob_dropout = 0):
    """
    Compute the loss and gradients for a two layer fully connected neural
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
    num_train, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    fc1 = X.dot(W1) + b1 # (N x H)
    #Apply the RELU
    relu = fc1 # (N x H)
    mask = fc1 < 0
    relu[mask] = 0
    
    drop_mask = np.random.rand(*relu.shape) < prob_dropout
    
    relu *= drop_mask
    
    #Find number of sleeping nodes.
    nsleep = np.sum(relu == 0) / (1.0 * relu.size)
    
    scores = relu.dot(W2) + b2 # (N x C)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    prob = scores
    prob -= np.max(prob, axis=1).reshape(prob.shape[0], 1)
    prob = np.exp(prob)
    prob /= (np.sum(prob, axis=1).reshape(prob.shape[0], 1))
    data_loss = (-1.0 / num_train) * np.sum(np.log(prob[np.arange(num_train), y] + 1e-13)) # (N x 1)
    loss = data_loss + 0.5*reg*(np.sum(W1 * W1) + np.sum(W2 * W2))
    #############################################################################
    #                              END OF YOUR CODE                            #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    grads['b2'] = np.zeros((W2.shape[1], 1)) # (C, 1)
    grads['W2'] = np.zeros(W2.shape)
    grads['W1'] = np.zeros(W1.shape)
    grads['b1'] = np.zeros((W1.shape[1], 1))
    db2 = np.zeros((W2.shape[1],)) # (C x 1)
    dW2 = np.zeros(W2.shape) # (H x C)
    dW1 = np.zeros(W1.shape)
    db1 = np.zeros((W1.shape[1],))
    
    prob[np.arange(num_train), y] -= 1
    d1 = prob
    db2 = np.sum(prob, axis=0)
    #Relu = (N x H) and d1 = (N x C)
    dW2 = (relu.T).dot(d1)
    d3 = d1.dot(W2.T) #(N x H)
    d3[fc1 <= 0] = 0
    
    #Added dropout
    d3 *= drop_mask
    
    db1 = np.sum(d3, axis=0)
    d4 = X.T.dot(d3)
    dW1 = d4
    
    #Compute the derivative wrt input images
    dX = d3.dot(W1.T) # (N x D)
    
    grads['W2'] = dW2 / num_train + reg * W2    
    grads['b2'] = db2 / num_train
    grads['W1'] = dW1 / num_train + reg * W1
    grads['b1'] = db1 / num_train
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads, nsleep, dX

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            prob_dropout=0.5,
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
    nsleep = 0
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = np.random.choice(np.arange(num_train),size=batch_size,replace=True)
      X_batch = X[indices, :]
      y_batch = y[indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads, nslp, dX = self.loss(X_batch, y=y_batch, reg=reg, prob_dropout=prob_dropout)
      nsleep += nslp
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
        
      for (i, key) in enumerate(grads.keys()):
        self.params[key] -= learning_rate * grads[key]


      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        if loss > 2.5:
            return None

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        print 'TRAIN: %f VAL %f Sleep %f'%(train_acc, val_acc, (nsleep / iterations_per_epoch))
        nsleep = 0
#        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
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
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    num_train, D = X.shape
    fc1 = X.dot(W1) + b1
    #Apply the RELU
    relu = fc1
    relu[fc1 < 0] = 0
    scores = relu.dot(W2) + b2
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


