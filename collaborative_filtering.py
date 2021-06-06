import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils import *

def similarity(x, X):
    # Requires normalized x, X as input.
    x_mat = torch.tile(x, (X.shape[0], 1))
    return x_mat * X / (torch.norm(x) * torch.norm(X, axis=1))

def find_item_neighborhood():
    pass

def find_dissimilar_items():
    pass

def initialize_system(n_users,n_beers,n_latent):
    eps = .1 # Move to config.
    X_init = eps * torch.rand(size=[n_users, n_latent])
    theta_init = eps * torch.rand(size=[n_beers, n_latent])
    return X_init, theta_init

class Matrix_Factorization():
    def __init__(self, X_train, n_latent, alpha, lmda, epochs):
        eps = .1 # Move to config.
        self.theta = eps * torch.rand(size=[n_users, n_latent])
        self.X = eps * torch.rand(size=[n_beers, n_latent])
        n_users, n_beers = X_train.shape
        self.training_set = X_train
        self.optimizer = torch.optim.Adam([self.X, self.theta], lr=alpha,\
                                    weight_decay = lmda)
        self.criterion = torch.nn.MSELoss()
        self.epochs = epochs

    def loss_function(self, pred):
        # weight_decay param to optimizer adds regularization term.
        return self.loss(pred, self.training_set)

    def train():
        for e in range(epochs):
            self.optimizer.zero_grad()
            prediction = torch.matmul(self.X, self.theta.T)
            loss = self.criterion(prediction, self.training_set)
            loss.backward()
            self.optimizer.step()

    def test_model():
        pass

if __name__ == '__main__':
    config = read_config()
    training_df = pd.read_csv(config['data_path']+config['data_name'], nrows=10000)
    X_train = get_numpy_features(training_df)
    matrix_fact_model = Matrix_Factorization(X_train, config['latent_dims'], \
                          config['learning_rate'], config['regularization_lambda'], \
                          config['epochs'])
