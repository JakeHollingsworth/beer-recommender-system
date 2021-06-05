import yaml
import torch

def read_config():
    with open('config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
        for key, item in config_dict.items():
            print(key, ':', item)

def get_numpy_features():
    pass

def test_train_validation_test_split():
    pass

def read_trained_model():
    pass

def write_trained_model():
    pass

class Normalize_Features():

    def __init__(self, X):
        self.means(X.means(axis=0))

    def normalize_data(X):
        return X - self.means

    def unnormalize_data(X):
        return X + self.means
