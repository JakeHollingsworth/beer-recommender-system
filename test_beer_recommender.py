import unittest
import numpy as np
import pandas as pd
from setup import *
from utils import *

class Test_Beer_Recommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pd_head = ['review_appearance', 'beer_style', 'review_palate', 'review_taste',
           'beer_name', 'review_timeUnix', 'beer_ABV', 'beer_beerId',
           'beer_brewerId', 'review_timeStruct', 'review_overall', 'review_text',
           'user_profileName', 'review_aroma', 'user_gender', 'user_birthdayRaw',
           'user_birthdayUnix', 'user_ageInSeconds']
        pd_data = np.array([[0,0,0,0,'0',0,0,0,0,0,2,0,'a',0,0,0,0,0],\
                            [0,0,0,0,'0',0,0,0,0,0,5,0,'b',0,0,0,0,0],\
                            [0,0,0,0,'0',0,0,0,0,0,3,0,'c',0,0,0,0,0],\
                            [0,0,0,0,'1',0,0,1,0,0,2,0,'b',0,0,0,0,0],\
                            [0,0,0,0,'1',0,0,1,0,0,3,0,'c',0,0,0,0,0],\
                            [0,0,0,0,'1',0,0,1,0,0,2,0,'d',0,0,0,0,0],\
                            [0,0,0,0,'2',0,0,2,0,0,3,0,'a',0,0,0,0,0],\
                            [0,0,0,0,'2',0,0,2,0,0,1,0,'c',0,0,0,0,0],\
                            [0,0,0,0,'2',0,0,2,0,0,2,0,'d',0,0,0,0,0]])
        cls._test_X_df = pd.DataFrame(pd_data, columns=pd_head)
        cls._test_X_np = np.array([[2, np.nan, 3],[5, 2, np.nan],[3,3,1], [np.nan, 2, 2]])
        cls._normalize_obj = Normalize_Features(cls._test_X_np)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_request_data(self):
        self.assertEqual(2, 2)

    def test_format_data(self):
        pass

    def test_write_data(self):
        pass

    def test_remove_raw_data(self):
        pass

    def test_read_data(self):
        pass

    def test_get_numpy_features(self):
        converted_df = get_numpy_features(self._test_X_df)
        self.assertTrue(np.allclose(converted_df, self._test_X_np, equal_nan=True))

    def test_train_validation_test_split(self):
        #train_validation_test_split(self._test_X_df,'user_profileName')
        #self.assertTrue(np.any(1))
        pass

    def test_read_trained_model(self):
        pass

    def test_write_trained_model(self):
        pass

    def test_read_config(self):
        pass

    def test_normalize_data(self):
        normalized_matrix = self._normalize_obj.normalize_data(self._test_X_np)
        expected_result = np.array([[-.5, 0, .5], [1.5, -1.5, 0], [2/3, 2/3, -4/3], [0,0,0]])
        self.assertTrue(np.allclose(normalized_matrix, expected_result))

    def test_unnormalize_data(self):
        X = np.array([[-.5, 0, .5], [1.5, -1.5, 0], [2/3, 2/3, -4/3], [0,0,0]])
        unnormalized_X = self._normalize_obj.unnormalize_data(X)
        expected_result = np.nan_to_num(self._test_X_np)
        self.assertTrue(np.allclose(unnormalized_X, expected_result))

    def test_similarity(self):
        pass

    def test_find_item_neighborhood(self):
        pass

    def test_find_dissimilar_items(self):
        pass

    def test_initialize_system(self):
        pass

    def test_loss_function(self):
        pass

    def test_train_model(self):
        pass

    def test_test_model(self):
        pass

    def test_cross_validate(self):
        pass

if __name__ == '__main__':
    unittest.main()
