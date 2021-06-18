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

    def fit_model(self):
        training_losses = []
        self.train()
        for e in range(self._epochs):
            self._optimizer.zero_grad()
            theta = torch.index_select(self._theta, 0, self._user_indices)
            x = torch.index_select(self._X, 0, self._item_indices)
            # Dot's each user vector with each item vector for each reviewed item.
            prediction = torch.sum(theta * x, 1)
            loss = self._criterion(prediction, self._training_set)
            training_losses.append(loss.item())
            loss.backward()
            self._optimizer.step()
        self.eval()
        plt.plot(training_losses)
        plt.show()


    def test_model(self, X_test, test_user_indices, test_item_indices):
        theta = torch.index_select(self._theta, 0, self._user_indices)
        x = torch.index_select(self._X, 0, self._item_indices)
        prediction = torch.sum(theta * x, 1)
        mean_training_error = self._criterion(prediction, self._training_set).item()

        testing_set = self._data_normalizer.normalize_data(X_test, test_user_indices)
        theta_t = torch.index_select(self._theta, 0, test_user_indices)
        x_t = torch.index_select(self._X, 0, test_item_indices)
        test_predictions = torch.sum(theta_t * x_t, 1)
        for i in range(10):
            print(test_predictions[i].item(), testing_set[i].item())
        mean_testing_error = self._criterion(test_predictions, testing_set).item()

        print('Mean L2 training error : ', mean_training_error)
        print('Mean L2 testing error  : ', mean_testing_error)

    def save_model(self, model_file):
        save_dict = {
                      "X"         : self._X.detach().cpu().numpy(),
                      "Theta"     : self._theta.detach().cpu().numpy()
                    }
        with open(model_file, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            model_tensors = pickle.load(f)
        self._X = torch.tensor(model_tensors['X'])
        self._theta = torch.tensor(model_tensors['Theta'])

class New_User(torch.nn.Module):
    def __init__(self, saved_model_path, alpha, lmda, epochs):
        super(New_User, self).__init__()
        model_tensors = self.read_trained_model(saved_model_path)
        self._X = torch.tensor(model_tensors['X'])
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
        self._optimizer = torch.optim.Adam([self._user_theta], lr=self._alpha, weight_decay = self._lmda)
        self._criterion = torch.nn.MSELoss()

    def read_trained_model(self, model_file):
        with open(model_file, 'rb') as f:
            model_tensors = pickle.load(f)
        return model_tensors

    def add_new_rating(self, new_item_inds, new_item_ratings):
        self.user_rated_item_indices = new_item_inds
        self.user_rated_item_ratings = new_item_ratings
        self.initialize_new_user_model()
        self.train()

    def fit_model(self):
        for e in range(self._epochs):
            self._optimizer.zero_grad()
            theta = self._user_theta.repeat(len(self._user_rated_item_indices),1)
            x = torch.index_select(self._X, 0, np.array(self._user_rated_item_indices))
            # Dot's each user vector with each item vector for each reviewed item.
            prediction = torch.sum(theta * x, 1)
            loss = self._criterion(prediction, np.array(self.user_rated_item_ratings))
            loss.backward()
            self._optimizer.step()

    def get_top_N(self, N):
        theta = self._user_theta.repeat(self._X.shape[0],1)
        predictions = torch.sum(theta * self._X, 1)
        _, top_N_indices = torch.topk(predictions, N)
        # Cast to python ints
        return [int(i) for i in top_N_indices]

if __name__ == '__main__':
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    full_df = pd.read_csv(config['data_path']+config['data_name'])
    training_df, val_df, test_df = train_validation_test_split(full_df, [.7, .15])
    X_train, user_indices, item_indices = get_data_info(training_df)
    matrix_factor_model = Matrix_Factorization(X_train, user_indices, item_indices, \
                          config['latent_dims'], config['learning_rate'], \
                          config['regularization_lambda'], config['epochs'])
    matrix_factor_model.to(device)
    matrix_factor_model.fit_model()
    matrix_factor_model.save_model('torch_model.pickle')
    #matrix_factor_model.load_model('torch_model.pickle')
    X_val, val_user_indices, val_item_indices = get_data_info(val_df)
    matrix_factor_model.test_model(X_val, val_user_indices, val_item_indices)
    '''
    user_ml_model = New_User(config['model_file'], config['learning_rate'], config['regularization_lambda'], config['epochs'])
    items = [10, 3000, 250, 760]
    ratings = [-2, 1, -1, 0]
    user_ml_model.add_new_rating(items, ratings)
    user_ml_model.get_top_N(10)
    '''
