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
    def __init__(self, X_train, user_indices, item_indices, n_latent, alpha, lmda, epochs):
        '''
        Creates a matrix factorization recommender system.

        Args:
            X_train - n_reviews x 1, overall review scores.
            user_indices - n_reviews x 1, index associated with user that submitted the review
            item_indices - n_reviews x 1, index associated with item that is being reviewed
            n_latent - number of latent features assigned to each item.
            alpha - learning rate
            lmbda - regularization parameter
            epochs - Number of training epochs
        '''

        super(Matrix_Factorization, self).__init__()
        eps = .1 # Move to config.

        self._theta = torch.tensor(np.random.uniform(-eps, eps, size=[n_users, n_latent]), requires_grad=True)
        self._X = torch.tensor(np.random.uniform(-eps, eps, size=[n_items, n_latent]), requires_grad=True)
        self._user_indices = user_indices
        self._item_indices = item_indices

        self._optimizer = torch.optim.Adam([self.X, self.theta], lr=alpha, weight_decay = lmda)
        self._criterion = torch.nn.MSELoss()
        self._epochs = epochs

        self._data_normalizer = Normalize_Features(X_train)
        self._training_set = self._data_normalizer.normalize_data(X_train, user_indices)

    def loss_function(self, pred):
        # weight_decay param to optimizer adds regularization term.
        return self.loss(pred, self._training_set)

    def train(self):
        for e in range(self.epochs):
            self._optimizer.zero_grad()
            # Dot's each user vector with each item vector for each reviewed item.
            prediction = torch.sum(self._theta[item_index] * self._X[user_index], 1)
            loss = self._criterion(prediction, self._training_set)
            loss.backward()
            self._optimizer.step()

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
