from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression(object):

    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.epoch_list = []

    def sigmoid(self, s: np.ndarray) ->np.ndarray:
        """		
        Sigmoid.
        
        Args:
            s: (N, D) numpy array
        Return:
            (N, D) numpy array, whose values are transformed by sigmoid function to the range (0, 1)
        """
        return (1 / (1 + np.exp(-s)))

    def bias_augment(self, x: np.ndarray) ->np.ndarray:
        """		
        Prepend a column of 1's to the x matrix
        
        Args:
            x (np.ndarray): (N, D) numpy array, N data points each with D features
        Returns:
            x_aug: (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
        """
        N, _ = x.shape
        column = np.ones((N, 1))
        x_aug = np.concatenate((column, x), axis=1) # preprend column to x 
        
        return x_aug

    def predict_probs(self, x_aug: np.ndarray, theta: np.ndarray) ->np.ndarray:
        """		
        Given model weights theta and input data points x, calculate the logistic regression model's
        predicted probabilities for each point
        
        Args:
            x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
            theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model
        Returns:
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label
                this result is h(x) = P(y = 1 | x)
        """
        h_x = self.sigmoid(x_aug @ theta)
        return h_x

    def predict_labels(self, h_x: np.ndarray, threshold: float) ->np.ndarray:
        """		
        Given model weights theta and input data points x, calculate the logistic regression model's
        predicted label for each point based on the threshold
        
        Args:
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label
            threshold (float): Float between 0 and 1, anything above the threshold should be classified as 1.
        Returns:
            y_hat (np.ndarray): (N, 1) numpy array, the predicted labels of each data point
                0 for negative label, 1 for positive label
        """
        # np.isin() checks to see if certain values are within an array
        # np.where() finds the indices for a specific value in an array
        N, _ = h_x.shape
        boolean_mask = h_x > threshold
        true_indices = np.where(boolean_mask == True)
        false_indices = np.where(boolean_mask == False)
        
        y_hat = np.zeros((N, 1))
        y_hat[true_indices] = 1
        y_hat[false_indices] = 0
        
        return y_hat
            

    def loss(self, y: np.ndarray, h_x: np.ndarray) ->float:
        """		
        Given the true labels y and predicted probabilities h_x, calculate the
        binary cross-entropy loss.
        
        Args:
            y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of being positive
        Returns:
            loss (float)
        """
        N, _ = y.shape
        loss_term = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)
        loss = (-1 / N) * np.sum(loss_term)
        
        return loss
        

    def gradient(self, x_aug: np.ndarray, y: np.ndarray, h_x: np.ndarray
        ) ->np.ndarray:
        """		
        Calculate the gradient of the loss function with respect to the parameters theta.
        
        Args:
            x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
            y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
            h_x: (N, 1) numpy array, the predicted probabilities of being positive
                    it is calculated as sigmoid(x multiply theta)
        Returns:
            grad (np.ndarray): (D + 1, 1) numpy array,
                the gradient of the loss function with respect to the parameters theta.
        Hints:
            Matrix multiplication takes care of the summation part of the definition.
        """
        N, _ = x_aug.shape
        term = x_aug.T @ (h_x - y)
        grad = (1 / N) * term
        
        return grad

    def accuracy(self, y: np.ndarray, y_hat: np.ndarray) ->float:
        """		
        Calculate the accuracy of the predicted labels y_hat
        
        Args:
            y (np.ndarray): (N, 1) numpy array, true labels
            y_hat (np.ndarray): (N, 1) numpy array, predicted labels
        Returns:
            accuracy of the given parameters theta on data x, y
        """
        correct_predictions = np.sum(y == y_hat)
        accuracy = correct_predictions / len(y)
        
        return accuracy

    def evaluate(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
        threshold: float) ->Tuple[float, float]:
        """		
        Given data points x, labels y, and weights theta
        Calculate the loss and accuracy
        
        Don't forget to add the bias term to the input data x.
        
        Args:
            x (np.ndarray): (N, D) numpy array, N data points each with D features
            y (np.ndarray): (N, 1) numpy array, true labels
            theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model
            threshold (float): Float between 0 and 1, anything above the threshold should be classified as 1
        Returns:
            Tuple[float, float]: loss, accuracy
        """
        x_aug = self.bias_augment(x) # add bias term column to x dataset
        h_x = self.predict_probs(x_aug, theta) # predicts x * theta 
        y_hat = self.predict_labels(h_x, threshold) # if the probability is above some threshold, then it'll assign a 1 to that index
        
        loss = self.loss(y, h_x)
        accuracy = self.accuracy(y, y_hat)
        
        return (loss, accuracy)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.
        ndarray, y_val: np.ndarray, lr: float, epochs: int, threshold: float
        ) ->Tuple[np.ndarray, List[float], List[float], List[float], List[
        float], List[int]]:
        """		
        Use gradient descent to fit a logistic regression model
        
        Pseudocode:
        1) Initialize weights and bias `theta` with zeros
        2) Augment the training data for simplified multication with the `theta`
        3) For every epoch
            a) For each point in the training data, predict the probability h(x) = P(y = 1 | x)
            b) Calculate the gradient of the loss using predicted probabilities h(x)
            c) Update `theta` by "stepping" in the direction of the negative gradient, scaled by the learning rate.
            d) If the epoch = 0, 100, 200, ..., call the self.update_evaluation_lists function
        4) Return the trained `theta`
        
        Args:
            x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
            y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
            x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
            y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
            lr (float): Learning Rate
            epochs (int): Number of epochs (e.g. training loop iterations)
            threshold (float): Float between 0 and 1, anything above the threshold should be classified as 1
        
        Return:
            theta: (D + 1, 1) numpy array, the parameters of the fitted/trained model
        """
        # Step 1: Initialize weights and bias `theta` with zeros
        _, D = x_train.shape
        theta = np.zeros((D + 1, 1))
        
        # Step 2: Augment the training data for simplified multication with the `theta`
        x_aug = self.bias_augment(x_train)
        
        # Step 3
        for epoch in range(epochs):
            # For each point in the training data, predict the probability h(x) = P(y = 1 | x)
            h_x = self.predict_probs(x_aug, theta)
            # Calculate the gradient of the loss using predicted probabilities h(x)
            gradient = self.gradient(x_aug, y_train, h_x)
            # Update `theta` by "stepping" in the direction of the negative gradient, scaled by the learning rate.
            theta -= lr * gradient
            if epoch % 100 == 0:
                self.update_evaluation_lists(x_train=x_train, y_train=y_train, 
                                             x_val=x_val, y_val=y_val, theta=theta, 
                                             epoch=epoch, threshold=threshold)
        
        return theta

    def update_evaluation_lists(self, x_train: np.ndarray, y_train: np.
        ndarray, x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray,
        epoch: int, threshold: float):
        """
        PROVIDED TO STUDENTS

        Updates lists of training loss, training accuracy, validation loss, and validation accuracy

        Args:
            x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
            y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
            x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
            y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
            theta: (D + 1, 1) numpy array, the current parameters of the model
            epoch (int): the current epoch number
            threshold (float): Float between 0 and 1, anything above the threshold should be classified as 1.
        """
        train_loss, train_acc = self.evaluate(x_train, y_train, theta,
            threshold)
        val_loss, val_acc = self.evaluate(x_val, y_val, theta, threshold)
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        if epoch % 1000 == 0:
            print(
                f"""Epoch {epoch}:
    train loss: {round(train_loss, 3)}	train acc: {round(train_acc, 3)}
    val loss:   {round(val_loss, 3)}	val acc:   {round(val_acc, 3)}"""
                )

    def plot_loss(self, train_loss_list: List[float]=None, val_loss_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        """
        PROVIDED TO STUDENTS

        Plot the loss of the train data and the loss of the test data.

        Args:
            train_loss_list: list of training losses from fit() function
            val_loss_list: list of validation losses from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_loss_list is None:
            assert hasattr(self, 'train_loss_list')
            assert hasattr(self, 'val_loss_list')
            assert hasattr(self, 'epoch_list')
            train_loss_list = self.train_loss_list
            val_loss_list = self.val_loss_list
            epoch_list = self.epoch_list
        fig = plt.figure()
        plt.plot(epoch_list, train_loss_list, label='Train Loss')
        plt.plot(epoch_list, val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self, train_acc_list: List[float]=None, val_acc_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        """
        PROVIDED TO STUDENTS

        Plot the accuracy of the train data and the accuracy of the test data.

        Args:
            train_acc_list: list of training accuracies from fit() function
            val_acc_list: list of validation accuracies from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_acc_list is None:
            assert hasattr(self, 'train_acc_list')
            assert hasattr(self, 'val_acc_list')
            assert hasattr(self, 'epoch_list')
            train_acc_list = self.train_acc_list
            val_acc_list = self.val_acc_list
            epoch_list = self.epoch_list
        fig = plt.figure()
        plt.plot(epoch_list, train_acc_list, label='Train Accuracy')
        plt.plot(epoch_list, val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()


def hyperparameter_tuning(model: LogisticRegression, x_test: np.ndarray,
    y_test: np.ndarray, theta: np.ndarray, thresholds: np.ndarray) ->Tuple[
    float, float]:
    """	
    Perform hyperparameter tuning by finding the threshold which generates the best accuracy.
    If the accuracies are the same for more than 1 threshold, pick the smaller one.
    
    Args:
        model: the Logistic Regression model we are passing in
        x_test (np.ndarray): (N, D) numpy array, N validation data points each with D features
        y_test (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
        theta (np.ndarray): the parameters of the fitted/trained model
        thresholds (np.ndarray): 1-D numpy array of threshold values that should be float between 0 and 1, anything above the threshold should be classified as 1
    Returns:
        t (tuple(float, float)): tuple in the format of (best-threshold, best-accuracy)
    Hints:
        model is the LogisticRegression you implemented. Use evaluate() from that class.
    """
    best_accuracy = 0
    best_threshold = 0
    for threshold in thresholds:
        _, accuracy = model.evaluate(x=x_test, y=y_test, theta=theta, threshold=threshold)
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
        elif accuracy == best_accuracy:
            best_threshold = min(best_threshold, threshold)
    
    t = (best_threshold, best_accuracy)
    
    return t
    
        
