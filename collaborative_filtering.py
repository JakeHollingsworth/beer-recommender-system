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

def get_training_info(X_df):
    X_train = torch.tensor(training_df['review_overall'].values, dtype=torch.float64)
    user_indices = torch.tensor(training_df['user_index'].values, dtype=torch.int32)
    item_indices = torch.tensor(training_df['item_index'].values, dtype=torch.int32)
    return X_train, user_indices, item_indices

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
        self._user_indices = user_indices
        self._item_indices = item_indices
        n_users, n_items = torch.unique(user_indices).shape[0], torch.unique(item_indices).shape[0]
        self._theta = torch.tensor(np.random.uniform(-eps, eps, size=[n_users, n_latent]), requires_grad=True)
        self._X = torch.tensor(np.random.uniform(-eps, eps, size=[n_items, n_latent]), requires_grad=True)

        self._optimizer = torch.optim.Adam([self._X, self._theta], lr=alpha, weight_decay = lmda)
        self._criterion = torch.nn.MSELoss()
        self._epochs = epochs

        self._data_normalizer = Normalize_Features(X_train, user_indices)
        self._training_set = self._data_normalizer.normalize_data(X_train, user_indices)

    def loss_function(self, pred):
        # weight_decay param to optimizer adds regularization term.
        return self.loss(pred, self._training_set)

    def train(self):
        for e in range(self._epochs):
            self._optimizer.zero_grad()
            theta = torch.index_select(self._theta, 0, self._user_indices)
            x = torch.index_select(self._X, 0, self._item_indices)
            # Dot's each user vector with each item vector for each reviewed item.
            prediction = torch.sum(theta * x, 1)
            loss = self._criterion(prediction, self._training_set)
            loss.backward()
            self._optimizer.step()

    def test_model(self):
        pass

if __name__ == '__main__':
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    training_df = pd.read_csv(config['data_path']+config['data_name'], nrows=10000)
    X_train, user_indices, item_indices = get_training_info(training_df)
    matrix_fact_model = Matrix_Factorization(X_train, user_indices, item_indices, \
                          config['latent_dims'], config['learning_rate'], \
                          config['regularization_lambda'], config['epochs'])
    matrix_fact_model.to(device)
    matrix_fact_model.train()
