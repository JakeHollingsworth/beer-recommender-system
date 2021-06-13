import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle

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

def get_data_info(X_df):
    X_train = torch.tensor(X_df['review_overall'].values, dtype=torch.float64)
    user_indices = torch.tensor(X_df['user_index'].values, dtype=torch.int32)
    item_indices = torch.tensor(X_df['item_index'].values, dtype=torch.int32)
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

    def test_model(self, X_test, test_user_indices, test_item_indices):
        theta = torch.index_select(self._theta, 0, self._user_indices)
        x = torch.index_select(self._X, 0, self._item_indices)
        prediction = torch.sum(theta * x, 1)
        mean_training_error = self._criterion(prediction, self._training_set).item()

        theta_test = torch.index_select(self._theta, 0, test_user_indices)
        x_test = torch.index_select(self._X, 0, test_item_indices)
        test_predictions = torch.sum(theta_test * x_test, 1)
        print(test_predictions.shape, X_test.shape)
        mean_testing_error = self._criterion(test_predictions, X_test).item()

        print('Mean L2 training error : ', mean_training_error)
        print('Mean L2 testing error  : ', mean_testing_error)

    def save_model(self, model_file):
        save_dict = {
                      "X"         : self._X.detach().cpu().numpy(),
                      "Theta"     : self._theta.detach().cpu().numpy()
                    }
        with open(model_file, 'wb') as f:
            pickle.dump(save_dict, f)

class New_User(torch.nn.Module):
    def __init__(self, saved_model_path, alpha, lmda, epochs):
        super(Matrix_Factorization, self).__init__()
        model_tensors = self.read_trained_model(saved_model_path)
        self._X = model_tensors['X']
        self._n_latent = self._X.shape[1]
        self._lmda = lmda
        self._alpha = alpha
        self._epochs = epochs
        self._user_rated_item_indices = []
        self._user_rated_item_ratings = []

        self.initialize_new_user_model()

    def initialize_new_user_model(self):
        eps=.1
        self._user_theta = torch.tensor(np.random.uniform(-eps, eps, size=[1, self._n_latent]), requires_grad=True)
        self._optimizer = torch.optim.Adam(self._user_theta, lr=self._alpha, weight_decay = self._lmda)
        self._criterion = torch.nn.MSELoss()

    def read_trained_model(self, model_file):
        with open(model_file, 'rb') as f:
            model_tensors = pickle.load(f)
        return model_tensors

    def add_new_rating(self, new_item_ind, new_item_rating):
        self.user_rated_item_indices.append(new_item_ind)
        self.user_rated_item_ratings.append(new_item_rating)
        self.initialize_new_user_model()
        self.train()

    def train(self):
        for e in range(self._epochs):
            self._optimizer.zero_grad()
            theta = torch.tile(self._user_theta, (1, len(self._user_rated_item_indices)))
            x = torch.index_select(self._X, 0, np.array(self._user_rated_item_indices))
            # Dot's each user vector with each item vector for each reviewed item.
            prediction = torch.sum(theta * x, 1)
            loss = self._criterion(prediction, np.array(self.user_rated_item_ratings))
            loss.backward()
            self._optimizer.step()

    def get_top_N(self, N):
        theta = torch.tile(self._user_theta, (1, self._X.shape[0]))
        max()

if __name__ == '__main__':
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    full_df = pd.read_csv(config['data_path']+config['data_name'], nrows=100000)
    training_df, val_df, test_df = train_validation_test_split(full_df, [.7, .15], 4)
    X_train, user_indices, item_indices = get_data_info(training_df)
    matrix_fact_model = Matrix_Factorization(X_train, user_indices, item_indices, \
                          config['latent_dims'], config['learning_rate'], \
                          config['regularization_lambda'], config['epochs'])
    matrix_fact_model.to(device)
    matrix_fact_model.train()
    X_val, val_user_indices, val_item_indices = get_data_info(val_df)
    matrix_fact_model.test_model(X_val, val_user_indices, val_item_indices)
