import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class GradientDescent:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.theta = None
        pass

    def predict(self, X):
        if self.theta == None:
            raise Exception("Theta not initialized")

        return self.theta@X.T

    def _gradient(self, X, Y):
       return -2* (Y - self.predict(X))@X / Y.shape[0]  

    def _loss(self, X, Y):
        l = np.linalg.norm(Y - self.predict(X))**2 / X.shape[0]
        return l

    def gradient_descent(lr = 1e-10, tol = 1e-2):   
        X = self.X_train
        Y = self.Y_train
        l = []
        theta = np.ones(X.shape[1])
        velocity = np.random.randn(theta.shape[0])
        prev_velocity = velocity
        g = self._gradient(theta)
        k = 0
        l.append(loss(theta))
        while (np.linalg.norm(g) > tol and k < 1000):
            k = k+1
            velocity = theta - lr * g
            theta = velocity + (k-1)/ (k+1) * (velocity - prev_velocity)
            prev_velocity = velocity
            g = gradient(theta)
            l.append(loss(theta))
        plt.plot(l)
        plt.show()
        return theta