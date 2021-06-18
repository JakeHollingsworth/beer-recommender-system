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
           'user_birthdayUnix', 'user_ageInSeconds', 'user_index', 'item_index']
        pd_data = np.array([[0,0,0,0,'0',0,0,0,0,0,2,0,'a',0,0,0,0,0,0,0],\
                            [0,0,0,0,'0',0,0,0,0,0,5,0,'b',0,0,0,0,0,1,0],\
                            [0,0,0,0,'0',0,0,0,0,0,3,0,'c',0,0,0,0,0,2,0],\
                            [0,0,0,0,'1',0,0,1,0,0,2,0,'b',0,0,0,0,0,1,1],\
                            [0,0,0,0,'1',0,0,1,0,0,3,0,'c',0,0,0,0,0,2,1],\
                            [0,0,0,0,'1',0,0,1,0,0,2,0,'d',0,0,0,0,0,3,1],\
                            [0,0,0,0,'2',0,0,2,0,0,3,0,'a',0,0,0,0,0,0,2],\
                            [0,0,0,0,'2',0,0,2,0,0,1,0,'c',0,0,0,0,0,2,2],\
                            [0,0,0,0,'2',0,0,2,0,0,2,0,'d',0,0,0,0,0,3,2]])
        cls._test_X_df = pd.DataFrame(pd_data, columns=pd_head)
        cls._user_indices = cls._test_X_df['user_index'].values.astype(np.uint8)
        cls._item_indices = cls._test_X_df['item_index'].values.astype(np.uint8)
        cls._test_X_np = np.array([[2, np.nan, 3],[5, 2, np.nan],[3,3,1], [np.nan, 2, 2]])
        cls._test_X_tens = torch.tensor(cls._test_X_df['review_overall'].values.astype(np.float64))
        cls._normalize_obj = Normalize_Features(cls._test_X_tens, cls._user_indices)

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

    def test_train_validation_test_split(self):
        train_validation_test_split_ratio = [0.7,0.1]
        train_df, validation_df, test_df = train_validation_test_split(self._test_X_df,train_validation_test_split_ratio,random_state=4)
        df_diff = pd.concat([train_df,validation_df,test_df]).drop_duplicates(keep=False)
        self.assertTrue(len(train_df) + len(validation_df) + len(test_df) == len(self._test_X_df))
        self.assertTrue(len(train_df.user_profileName.unique()) == len(self._test_X_df.user_profileName.unique()))
        self.assertTrue(len(df_diff) == len(self._test_X_df))
        self.assertTrue(len(self._test_X_df.user_profileName.unique())==len(train_df.user_profileName.unique()))

    def test_read_trained_model(self):
        pass

    def test_write_trained_model(self):
        pass

    def test_read_config(self):
        pass

    def test_normalize_data(self):
        normalized_matrix = self._normalize_obj.normalize_data(self._test_X_tens, self._user_indices)
        expected_result = np.array([-.5, 1.5, 2/3, -1.5, 2/3, 0, .5, -4/3, 0])
        self.assertTrue(np.allclose(normalized_matrix.detach().cpu().numpy(), expected_result))

    def test_unnormalize_data(self):
        X = np.array([-.5, 1.5, 2/3, -1.5, 2/3, 0, .5, -4/3, 0])
        unnormalized_X = self._normalize_obj.unnormalize_data(torch.tensor(X), self._user_indices)
        expected_result = self._test_X_df['review_overall'].values.astype(np.float64)
        self.assertTrue(np.allclose(unnormalized_X.detach().cpu().numpy(), expected_result))

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
