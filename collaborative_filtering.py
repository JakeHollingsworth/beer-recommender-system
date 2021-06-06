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

def initialize_system(n_users,n_items,n_latent):
    eps = .1 # Move to config.
    X_init = eps * torch.rand(size=[n_users, n_latent])
    theta_init = eps * torch.rand(size=[n_items, n_latent])
    return X_init, theta_init

class Matrix_Factorization(torch.nn.Module):
    def __init__(self, X_train, n_latent, alpha, lmda, epochs):

        super(Matrix_Factorization, self).__init__()

        eps = .1 # Move to config.
        n_users, n_items = X_train.shape
        self.theta = torch.tensor(np.random.uniform(-eps, eps, size=[n_users, n_latent]), requires_grad=True)
        self.X = torch.tensor(np.random.uniform(-eps, eps, size=[n_items, n_latent]), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.X, self.theta], lr=alpha,\
                                    weight_decay = lmda)
        self.criterion = torch.nn.MSELoss()
        self.epochs = epochs

        self.data_normalizer = Normalize_Features(X_train)
        self.mask = torch.isnan(X_train).clone()
        self.training_set = self.data_normalizer.normalize_data(X_train)

    def loss_function(self, pred):
        # weight_decay param to optimizer adds regularization term.
        return self.loss(pred, self.training_set)

    def train(self):
        for e in range(self.epochs):
            self.optimizer.zero_grad()
            prediction = torch.matmul(self.theta, self.X.T)
            loss = self.criterion(prediction[self.mask], self.training_set[self.mask])
            loss.backward()
            self.optimizer.step()

    def test_model(self):
        pass

if __name__ == '__main__':
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    training_df = pd.read_csv(config['data_path']+config['data_name'], nrows=100000)
    X_train = torch.tensor(get_numpy_features(training_df), dtype=torch.float64)
    matrix_fact_model = Matrix_Factorization(X_train, config['latent_dims'], \
                          config['learning_rate'], config['regularization_lambda'], \
                          config['epochs'])
    matrix_fact_model.to(device)
    matrix_fact_model.train()
