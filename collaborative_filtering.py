import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle, sys
import os

from utils import *


def optimize_model(epochs=200,n_latent_range=[5,50],lr_range=[0.00001,0.1],lmbda_range=[10**(-8),10**(-3)],seed=4):
    def random_log_distribution(gen, low, high):
        value = gen.uniform(low=np.log10(low),high=np.log10(high))
        return 10**value

    gen = np.random.default_rng(seed=seed)

    # loading data and settings
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    full_df = pd.read_csv(config['data_path']+config['data_name'])
    training_df, val_df, test_df = train_validation_test_split(full_df, config['train_validation_test_split_ratio'])
    X_train, user_indices, item_indices = get_data_info(training_df)
    X_val, val_user_indices, val_item_indices = get_data_info(val_df)

    #create file and header if it doesn't exist
    if not(os.path.isfile(config['optimize_file'])):
        with open(config['optimize_file'],'w') as f:
            f.write('lr,lmbda,n_latent,training_loss,validation_loss\n')
    print("Running cross validation...")

    while True:
        lr = random_log_distribution(gen, low = lr_range[0], high = lr_range[1])
        lmbda = random_log_distribution(gen, low = lmbda_range[0], high = lmbda_range[1])
        n_latent = gen.integers(low = n_latent_range[0],high = n_latent_range[1])


        matrix_factor_model = Matrix_Factorization(X_train, user_indices, item_indices, \
                              n_latent, lr, \
                              lmbda, epochs)
        matrix_factor_model.to(device)
        _ = matrix_factor_model.fit_model()
        mean_training_error, mean_testing_error = matrix_factor_model.test_model(X_val, val_user_indices, val_item_indices)

        with open(config['optimize_file'],'a') as f:
            f.write(f'{lr},{lmbda},{n_latent},{mean_training_error},{mean_testing_error}\n')


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

        return mean_training_error, mean_testing_error

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
        self.read_dictionaries()
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

    def get_similarities(self, candidate_inds):
        # Item similarities between already rated items and candidates.
        user_ratings = torch.tensor(self.user_rated_item_ratings)
        # Weight according to user rating
        rated_thetas = torch.nn.functional.normalize(self._X[self.user_rated_item_indices]) * user_ratings[:,None]
        prediction_thetas = torch.nn.functional.normalize(self._X[candidate_inds]).T
        weighted_similaritys = torch.matmul(rated_thetas, prediction_thetas)
        return torch.sum(weighted_similaritys, 0)

    def get_popularities(self, candidate_inds):
        popularities = [self.popularities[i.item()] for i in candidate_inds]
        popularities = torch.tensor(popularities, dtype=torch.float32)
        transformation = lambda x: torch.log(x + 1) / torch.log(torch.max(x) + 1)
        return transformation(popularities)

    def get_top_N(self, N):
        model_weight = 1
        sim_weight = 1
        pop_weight = 1
        theta = self._user_theta.repeat(self._X.shape[0],1)
        predictions = torch.sum(theta * self._X, 1)
        # Predicted ratings
        scores, top_N_indices = predictions, torch.arange(predictions.shape[0])
        mask = [0 if k.item() in self.user_rated_item_indices else 1 for k in top_N_indices]
        mask = torch.tensor(mask, dtype=torch.bool)
        candidate_indices = top_N_indices[mask]
        similarities = self.get_similarities(candidate_indices)
        popularities = self.get_popularities(candidate_indices)
        ranking = model_weight * scores[mask] + sim_weight * similarities + \
                  pop_weight * popularities
        _, final_recs = torch.topk(ranking, N)
        best_indices = candidate_indices[final_recs]
        # Cast to python ints
        return [int(i) for i in best_indices]

    def read_dictionaries(self, dicts_file='data/dicts.pickle'):
        with open(dicts_file, 'rb') as f:
            dicts = pickle.load(f)
        self.popularities = dicts['id_to_pop']


if __name__ == '__main__':
    config = read_config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    full_df = pd.read_csv(config['data_path']+config['data_name'])
    training_df, val_df, test_df = train_validation_test_split(full_df, config['train_validation_test_split_ratio'])
    X_train, user_indices, item_indices = get_data_info(training_df)
    matrix_factor_model = Matrix_Factorization(X_train, user_indices, item_indices, \
                          config['latent_dims'], config['learning_rate'], \
                          config['regularization_lambda'], config['epochs'])
    matrix_factor_model.to(device)
#    matrix_factor_model.fit_model()
#    matrix_factor_model.save_model(config['model_file'])
    matrix_factor_model.load_model(config['model_file'])
    X_val, val_user_indices, val_item_indices = get_data_info(val_df)
    mean_training_error, mean_testing_error = matrix_factor_model.test_model(X_val, val_user_indices, val_item_indices)
    print('Mean L2 training error : ', mean_training_error)
    print('Mean L2 testing error  : ', mean_testing_error)


    '''
    user_ml_model = New_User(config['model_file'], config['learning_rate'], config['regularization_lambda'], config['epochs'])
    items = [10, 3000, 250, 760]
    ratings = [-2, 1, -1, 0]
    user_ml_model.add_new_rating(items, ratings)
    user_ml_model.get_top_N(10)
    '''
