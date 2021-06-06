import yaml
import torch
import pandas as pd
import numpy as np

def read_config():
    with open('config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict

def get_numpy_features(data_df):
    unique_users = data_df.user_profileName.unique()
    unique_beers = data_df.beer_beerId.unique()
    beerID_to_column = {id: i for i, id in enumerate(unique_beers)}

    data_np = np.empty(shape=(unique_users.shape[0], unique_beers.shape[0]))
    data_np.fill(np.nan)
    for i, user in enumerate(unique_users):
        user_reviews = data_df[data_df.user_profileName == user]
        user_reviewed_beerIds = user_reviews.beer_beerId
        for id in user_reviewed_beerIds:
            overall = user_reviews[user_reviews.beer_beerId == id].review_overall
            overall = overall.values.astype(float)[0]
            data_np[i, beerID_to_column[id]] = overall

    return data_nps


def test_train_validation_test_split():
    pass

class Normalize_Features():

    def __init__(self, X):
        self.means(X.means(axis=0))

    def normalize_data(X):
        return X - self.means

    def unnormalize_data(X):
        return X + self.means
