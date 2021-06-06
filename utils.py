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
            overall = user_reviews[user_reviewed_beerIds == id].review_overall
            overall = overall.values.astype(float)[0]
            data_np[i, beerID_to_column[id]] = overall

    return data_np


def train_validation_test_split(data_df,train_validation_test_split_ratio,random_state):
    #grabbing one sample for all unique users and add to train data
    train_df_index = data_df.sample(frac=1,random_state=random_state).drop_duplicates(subset='user_profileName').index
    test_df = data_df.drop(index=train_df_index)

    #adding additional samples to reach desired split
    number_additional_samples = int(train_validation_test_split_ratio[0]*len(data_df)-len(train_df_index))
    if number_additional_samples > 0:
        train_additional_indices = test_df.sample(number_additional_samples,
                                                    random_state=random_state).index
        test_df.drop(train_additional_indices,inplace=True)
        train_df_index = train_df_index.append(train_additional_indices)

    #building dataframes
    train_df = data_df.iloc[train_df_index]
    validation_df = test_df.sample(frac = train_validation_test_split_ratio[1]/(1-train_validation_test_split_ratio[0]),
                                    random_state = random_state)
    test_df.drop(validation_df.index,inplace=True)

    return train_df, validation_df, test_df

class Normalize_Features():

    def __init__(self, X):
        self._means = np.nanmean(X.detach().cpu().numpy(), axis=1)[:, None]
        self._means = torch.tensor(self._means)

    def normalize_data(self, X):
        X = X - self._means
        # nan_to_num() exists in documentation, but not live
        #return torch.nan_to_num(X)
        return torch.tensor(np.nan_to_num(X.detach().cpu().numpy()))

    def unnormalize_data(self, X):
        return X + self._means
