import numpy as np

"""
We are going to use the California housing dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2-layer fully connected neural net. We are going to build the neural network from scratch.
"""


class NeuralNet:
    def __init__(
        self,
        y,
        use_dropout,
        use_adam,
        lr=0.01,
        batch_size=64,
        dropout_prob=0.3,
    ):
        """
        This method initializes the class, it is implemented for you.
        Args:
                y (np.ndarray): labels
                use_dropout (bool): flag to enable dropout
                use_adam (bool): flag to use adam
                lr (float): learning rate
                batch_size (int): batch size to use for training
                dropout_prob (float): dropout probability
        """
        self.y = y  # ground truth labels

        # OTHER HYPERPARAMTERS
        self.y_hat = np.zeros((self.y.shape[0], 3))  # estimated labels
        self.dimensions = [8, 15, 7, 3]  # dimensions of different layers
        self.alpha = 0.05
        self.eps = 1e-8

        # DROPOUT
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # PARAMETERS
        self.parameters = {}  # dictionary for different layer variables
        self.cache = (
            {}
        )  # cache for holding variables during forward propagation to use them in back prop
        self.loss = []  # list to store loss values
        self.batch_y = []  # list of y batched numpy arrays

        # TRAINING HYPERPARAMETERS
        self.iteration = 0  # iterator to index into data for making a batch
        self.batch_size = batch_size  # batch size

        # NEURAL NETWORK INFORMATION
        self.learning_rate = lr  # learning rate
        self.sample_count = self.y.shape[0]  # number of training samples we have
        self._estimator_type = "regression"
        self.neural_net_type = "SiLU -> SiLU -> Softmax"

        # Adam parameters
        self.use_adam = use_adam
        self.t = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.first_moment = {}
        self.second_moment = {}

    def init_parameters(self, param=None):
        """
        This method initializes the neural network variables, it is already implemented for you.
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.

        Args:
                param (dict): Optional dictionary of parameters to use instead of initializing.
        """
        if param is None:
            np.random.seed(0)
            self.parameters["theta1"] = np.random.randn(
                self.dimensions[0], self.dimensions[1]
            ) / np.sqrt(self.dimensions[0])
            self.parameters["b1"] = np.zeros((self.dimensions[1]))
            self.parameters["theta2"] = np.random.randn(
                self.dimensions[1], self.dimensions[2]
            ) / np.sqrt(self.dimensions[1])
            self.parameters["b2"] = np.zeros((self.dimensions[2]))
            self.parameters["theta3"] = np.random.randn(
                self.dimensions[2], self.dimensions[3]
            ) / np.sqrt(self.dimensions[2])
            self.parameters["b3"] = np.zeros((self.dimensions[3]))

        else:
            self.parameters = param
            self.parameters["theta1"] = self.parameters["theta1"]
            self.parameters["theta2"] = self.parameters["theta2"]
            self.parameters["theta3"] = self.parameters["theta3"]
            self.parameters["b1"] = self.parameters["b1"]
            self.parameters["b2"] = self.parameters["b2"]
            self.parameters["b3"] = self.parameters["b3"]

        for parameter in self.parameters:
            self.first_moment[parameter] = np.zeros_like(self.parameters[parameter])
            self.second_moment[parameter] = np.zeros_like(self.parameters[parameter])

    def softmax(self, u):
        """
        Performs softmax function function element-wise.
        To prevent overflow, begin by subtracting each row in u by its maximum!
        Input:
                u (np.ndarray: (N, D)): logits
        Output:
                o (np.ndarray: (N, D)): N probability distributions over D classes
        """
        max = np.max(u, axis=1, keepdims=True) # axis=1 means I get the max along the rows; axis=0 is max along columns
        reduced_u = u - max
        
        exp_u = np.exp(reduced_u)
        o = exp_u / np.sum(exp_u, axis=1, keepdims=True) # have to do keepdims to make sure we don't get a 1D array and broadcasting can occur
        
        return o
        

    def silu(self, u):
        """
        The SiLU (Sigmoid Linear Unit) activation function, also known as the Swish function.
        It is defined as: SiLU(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x)).

        Args:
            u (np.ndarray): input array with any shape.

        Returns:
           o (np.ndarray): output, same shape as input u.
        """
        sigmoid_u = 1 / (1 + np.exp(-u))
        o = u * sigmoid_u
        
        return o 

    def derivative_silu(self, x):
        """
        Derivative of the SiLU (Sigmoid Linear Unit) activation function.

        Args:
                x (np.ndarray): Input array.

        Returns:
                np.ndarray: Derivative of SiLU.
        """
        sigmoid_x = 1 / (1 + np.exp(-x))
        derivative_SiLu = sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        return derivative_SiLu
        

    @staticmethod
    def _dropout(u, prob):
        """
        Implement the dropout layer. Refer to the description for implementation details.
        Args:
                u (np.ndarray: (N, D)): input to dropout layer
                prob: the probability of dropping an unit
        Returns:
                u_after_dropout (np.ndarray: (N, D)): output of dropout layer
                dropout_mask (np.ndarray: (N, D)): dropout mask indicating which units were dropped

        Hint: scale the units after dropout
                  use np.random.choice to sample from Bernoulli(prob) the inactivated nodes for each iteration
        """
        N, K = u.shape
        dropout_mask = np.random.choice([0, 1], size=(N, K), p=[prob, 1 - prob]) # for each index in our u, assign a 0 with 0.3% and 1 with 0.7%
        
        u_after_dropout = u * dropout_mask
        u_after_dropout *= 1 / (1 - prob) # rescale (why do we do this?)
                
        return u_after_dropout, dropout_mask
        

    def cross_entropy_loss(self, y, y_hat):
        """
        Computes cross entropy loss.
        Refer to the description in the notebook and implement the appropriate mathematical equation.
        To avoid log(0) errors, add a small constant 1e-15 to the input to np.log
        Args:
                y (np.ndarray: (N, D)): one-hot ground truth labels
                y_hat (np.ndarray: (N, D)): predictions
        Returns:
                loss (float): average cross entropy loss
        """
        N, _ = y.shape
        eps = 1e-15
        loss= -np.sum(y * np.log(y_hat + eps)) / N
        
        return loss

    def forward(self, x, use_dropout):
        """
        We have provided an outline of the function you can use below. Fill in the "..." with the appropriate code.
        Check init_parameters method and use variables from there as well as other implemented methods.
        Implement the appropriate mathematical equations as described in the notebook.
        NOTE: Implement dropout only on the first layer!

        Args:
                x (np.ndarray: (N, M)): input to neural network
                use_dropout (bool): true if using dropout in forward.
        Returns:
                o3 (np.ndarray: (N, D)): output of neural network

        Note: The shapes of the variables in self.cache should be:
                u1 (np.ndarray: (N, K)): output after first linear layer
                o1 (np.ndarray: (N, K)): output after applying activation and dropout (if specified)
                u2 (np.ndarray: (N, R)): output after second linear layer
                o2 (np.ndarray: (N, R)): output of nueral network
                u3 (np.ndarray: (N, D)): output after third linear layer
                o3 (np.ndarray: (N, D)): output of nueral network

        where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on the forward pass.

        Outline:
        """
        self.cache["X"] = x
        u1 = (self.cache["X"] @ self.parameters["theta1"]) + self.parameters["b1"] # 1st layer
        o1 = self.silu(u1) # contains output for all neurons in a layer since we're working with matrices

        if use_dropout:
                o1, dropout_mask = self._dropout(o1, self.dropout_prob)
                self.cache["mask"] = dropout_mask

        self.cache["u1"], self.cache["o1"] = u1, o1

        u2 = (self.cache["o1"] @ self.parameters["theta2"]) + self.parameters["b2"] # 2nd layer
        o2 = self.silu(u2)
        self.cache["u2"], self.cache["o2"] = u2, o2

        u3 = (self.cache["o2"] @ self.parameters["theta3"]) + self.parameters["b3"] # 3rd layer
        o3 = self.softmax(u3)
        self.cache["u3"], self.cache["o3"] = u3, o3

        return o3
        

    def compute_gradients(self, y, yh):
        """
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Args:
            y (np.ndarray: (N, D)): ground truth values
            yh (np.ndarray: (N, D)): predicted outputs

        Returns:
            gradients (dict): dictionary with the following mapping and shapes:
                dLoss_theta3 (np.ndarray: (R, D)): gradients for theta3
                dLoss_b3 (np.ndarray: (D,)): gradients for b3
                dLoss_theta2 (np.ndarray: (K, R)): gradients for theta2
                dLoss_b2 (np.ndarray: (R,)): gradients for b2
                dLoss_theta1 (np.ndarray: (M, K)): gradients for theta1
                dLoss_b1 (np.ndarray: (K,)): gradients for b1
                where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features


        Note: You will have to use the cache (self.cache) to retrieve the values
        from the forward pass!

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on computing gradients.

        HINT 2: Division by N only needs to occur ONCE for any derivative that requires a division
        by N. Make sure you avoid cascading divisions by N where you might accidentally divide your
        derivative by N^2 or greater.

        HINT 3: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        Note that we only implemented drop out on the first hidden layer.
        """
        N, _ = y.shape
        dLoss_u3 = yh - y

        dLoss_theta3 = (self.cache["o2"].T @ dLoss_u3) / N  # dLoss_theta3 = dLoss / du3 @ du3 / dtheta3
        dLoss_b3 = ((dLoss_u3.T @ np.ones((N,1))).flatten()) / N # dLoss / du3 * du3 / dbias3 (1)
        dLoss_o2 = dLoss_u3 @ self.parameters["theta3"].T # dLoss / du3 @ theta3

        dLoss_u2 = dLoss_o2 * self.derivative_silu(self.cache["u2"]) # dLoss / do2 * dSiLU / du2

        dLoss_theta2 = (self.cache["o1"].T @ dLoss_u2) / N
        dLoss_b2 = ((dLoss_u2.T @ np.ones((N,1))).flatten()) / N
        dLoss_o1 = dLoss_u2 @ self.parameters["theta2"].T

        if self.use_dropout:
                """Problem is here"""
                dropout_mask = self.cache["mask"]
                dLoss_o1 *= dropout_mask
                dLoss_o1 *= 1 / (1 - self.dropout_prob)  # scale like in forward
                dLoss_u1 = dLoss_o1 * self.derivative_silu(self.cache["u1"])
        else:
                dLoss_u1 = dLoss_o1 * self.derivative_silu(self.cache["u1"])

        dLoss_theta1 = (self.cache["X"].T @ dLoss_u1) / N
        dLoss_b1 = ((dLoss_u1.T @ np.ones((N,1))).flatten()) / N
        
        gradients = {
                "theta1" : dLoss_theta1,
                "b1" : dLoss_b1,
                "theta2" : dLoss_theta2,
                "b2" : dLoss_b2,
                "theta3" : dLoss_theta3,
                "b3" : dLoss_b3
        }
        
        return gradients

    def update_weights(self, dLoss):
        """
        Update the weights in the network using the gradients of the parameters (dLoss). You should update self.parameters.
        For every parameter:
        - if self.use_adam is true:
            - self.first_moment (M), self.second_moment (V), self.beta1, self.beta2, self.t have been initialized for you.
            - You can either increment self.t here (once at the end), or increment it in every iteration in gradient_descent/minibatch_gradient_descent.
            - Refer to init_parameters to understand how to index into M & V to get M_t and V_t for every parameter t.
            - Update self.first_moment and self.second_moment using dLoss
            - Normalize the first & second moments and calculate the learning update
            - Update the parameters using the learning update and learning rate
        - else:
            - Update the parameters using dLoss and learning rate

        Args:
                dLoss (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)
                use_adam (bool): flag to use adam or not

        Return:
                None

        Hint: You must update self.parameters, self.first_moment and self.second_moment if use_adam = True.
        Otherwise, you only need to update self.parameters.
        """
        if self.use_adam:
                for parameter in self.parameters:
                        
                        # Update self.first_moment and self.second_moment using dLoss
                        self.first_moment[parameter] = (self.beta1 * self.first_moment[parameter]) + (1 - self.beta1) * dLoss[parameter]
                        self.second_moment[parameter] = (self.beta2 * self.second_moment[parameter]) + (1 - self.beta2) * np.square(dLoss[parameter])
                        
                        # Normalize the first & second moments and calculate the learning update
                        standardized_first_moment = self.first_moment[parameter] / (1 - (self.beta1 ** self.t))
                        standardized_second_moment = self.second_moment[parameter] / (1 - (self.beta2 ** self.t))
                        
                        # Update the parameters using the learning update and learning rate
                        self.parameters[parameter] = self.parameters[parameter] - (self.learning_rate * 
                                                                                   (standardized_first_moment / 
                                                                                    (np.sqrt(standardized_second_moment) + self.eps)))
                        self.t += 1
        else:
                self.parameters["theta3"] -= self.learning_rate * dLoss["theta3"]
                self.parameters["b3"] -= self.learning_rate * dLoss["b3"]
                self.parameters["theta2"] -= self.learning_rate * dLoss["theta2"]
                self.parameters["b2"] -= self.learning_rate * dLoss["b2"]
                self.parameters["theta1"] -= self.learning_rate * dLoss["theta1"]
                self.parameters["b1"] -= self.learning_rate * dLoss["b1"]

    def backward(self, y, yh):
        """
        Implement a complete backward pass using compute_gradients and update_weights

        Args:
                y (np.ndarray: (N, D)): ground truth labels
                yh (np.ndarray: (N, D)): neural network predictions
                use_dropout (bool): flag to use dropout
                use_adam (bool): flag to use adam

        Note: This function does not return anything. It should update the weights in the network using the gradients.
        """
        gradients = self.compute_gradients(y, yh)
        self.update_weights(gradients)

    def gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them.
        2. One iteration here is one round of forward and backward propagation on the complete dataset.
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss
        4. Increment self.t at every iteration if you haven't done so in update_weights.
        **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        Args:
                x (np.ndarray: N x M): input
                y (np.ndarray: N x D): ground truth labels
                iter (int): number of iterations to train for
                use_adam (bool): flag to use adam or not
                local_test (bool): flag to indicate if local test is being run or not

        HINT: Here's an outline of the function you can use.
        """        
        self.init_parameters()
        for i in range(iter):
                # TODO: implement training loop
                yh = self.forward(x, self.use_dropout)
                self.backward(y, yh)
                loss = self.cross_entropy_loss(y, yh)
                
                # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
                print_multiple = 1 if local_test else 1000
                if i % print_multiple == 0:
                        print("Loss after iteration %i: %f" % (i, loss))
                        self.loss.append(loss)
                self.t += 1 # increment self.t in update_weights

    def minibatch_gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        This function is an implementation of the batch gradient descent algorithm

        Notes:
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient
        2. One iteration here is one round of forward and backward propagation on one minibatch.
           You will use self.iteration and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations.
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th,
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail). Please
           batch your data in a wraparound manner. For example, given a dataset of 9 numbers,
           [1, 2, 3, 4, 5, 6, 7, 8, 9], and a batch size of 6, the first iteration batch will
           be [1, 2, 3, 4, 5, 6], the second iteration batch will be [7, 8, 9, 1, 2, 3],
           the third iteration batch will be [4, 5, 6, 7, 8, 9], etc...
        7. Increment self.t at every iteration if you haven't done so in update_weights.

        Args:
                x (np.ndarray: N x M): input data
                y (np.ndarray: N x D): ground truth labels
                use_adam (bool): flag to use adam or not
                iter (int): number of BATCHES to iterate through
                local_test (bool): True if calling local test, default False for autograder and Q1.3
                                this variable can be used to switch between autograder and local test requirement for
                                appending/printing out loss and y batch arrays
        HINT:  Here's an outline of the function you can use.
        """
        self.init_parameters()
        N, _ = x.shape
        for i in range(iter):
                # TODO: implement training loop
                start = (i * self.batch_size) % N
                end = start + self.batch_size

                if end <= N:
                        x_batch = x[start:end]
                        y_batch = y[start:end]
                else:
                        x_batch = np.concatenate((x[start:], x[:end - N]), axis=0)
                        y_batch = np.concatenate((y[start:], y[:end - N]), axis=0)
                
                yh_batch = self.forward(x_batch, self.use_dropout)
                self.backward(y_batch, yh_batch)
                loss = self.cross_entropy_loss(y_batch, yh_batch)
                
                # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
                print_multiple = 1 if local_test else 1000
                if i % print_multiple == 0:
                        print("Loss after iteration %i: %f" % (i, loss))
                        self.loss.append(loss)
                        self.batch_y.append(y_batch)
                self.t += 1 # increment self.t in update_weights

    def predict(self, x):
        """
        This function predicts new data points
        It is implemented for you

        Args:
                x (np.ndarray: (N, M)): input data
        Returns:
                y (np.ndarray: (N,)): predictions
        """
        yh = self.forward(x, False)  # (N, D) = forward((N, M))
        pred = np.argmax(yh, axis=1)  # (N,) = argmax((N, D), axis=1)
        return pred
