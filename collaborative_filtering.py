import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def similarity(x, X):
    # Requires normalized x, X as input.
    x_mat = torch.tile(x, (X.shape[0], 1))
    return x_mat * X / (torch.norm(x) * torch.norm(X, axis=1))

def find_item_neighborhood():
    pass

def find_dissimilar_items():
    pass

def initialize_system():
    eps = .1 # Move to config.


def loss_function():
    pass

def train_model():
    pass

def test_model():
    pass

def cross_validate():
    pass
