import yaml
import torch
import pandas as pd
import numpy as np
import pickle

def read_config():
    with open('config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict

def train_validation_test_split(data_df,train_validation_test_split_ratio,random_state=4):
    #grabbing one sample for all unique users and add to train data
    train_df_index_user = data_df.sample(frac=1,random_state=random_state).drop_duplicates(subset='user_profileName').index
    train_df_index_beer = data_df.sample(frac=1,random_state=random_state).drop_duplicates(subset='beer_beerId').index
    train_df_index = np.union1d(train_df_index_user, train_df_index_beer)
    #print(train_df_index_user.shape,train_df_index_beer.shape,train_df_index.shape)
    test_df = data_df.drop(index=train_df_index)
    #adding additional samples to reach desired split
    number_additional_samples = int(train_validation_test_split_ratio[0]*len(data_df)-len(train_df_index))
    if number_additional_samples > 0:
        train_additional_indices = test_df.sample(number_additional_samples,  random_state=random_state).index
        test_df.drop(train_additional_indices,inplace=True)
        train_df_index = np.union1d(train_df_index, train_additional_indices)

    #building dataframes
    train_df = data_df.iloc[train_df_index]
    validation_df = test_df.sample(frac = train_validation_test_split_ratio[1]/(1-train_validation_test_split_ratio[0]),
                                    random_state = random_state)
    test_df.drop(validation_df.index,inplace=True)

    return train_df, validation_df, test_df

def get_data_info(X_df):
    X_train = torch.tensor(X_df['review_overall'].values, dtype=torch.float64)
    user_indices = torch.tensor(X_df['user_index'].values, dtype=torch.long)
    item_indices = torch.tensor(X_df['item_index'].values, dtype=torch.long)
    return X_train, user_indices, item_indices

def write_dictionary(X_df):
    item_ids = list(X_df['item_index'].values)
    item_names = list(X_df['beer_name'].values)
    id_to_name = {id: name for id, name in zip(item_ids, item_names)}
    name_to_id = {name: id for id, name in zip(item_ids, item_names)}
    dicts = {"id_to_name": id_to_name, "name_to_id": name_to_id}
    with open("data/dicts.pickle", 'wb') as f:
        pickle.dump(dicts, f)

class Normalize_Features():

    def __init__(self, X, user_indices):
        mean_arr = [torch.mean(X[user_indices == i]) for i in range(np.unique(user_indices).shape[0])]
        self._means = torch.tensor(mean_arr)

    def normalize_data(self, X, user_indices):
        return X - torch.index_select(self._means, 0, torch.tensor(user_indices, dtype=torch.long))

    def unnormalize_data(self, X, user_indices):
        return X + torch.index_select(self._means, 0, torch.tensor(user_indices, dtype=torch.long))


if __name__=="__main__":
    config = read_config()
    X_df = pd.read_csv(config['data_path']+config['data_name'])
    write_dictionary(X_df)
