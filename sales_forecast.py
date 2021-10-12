import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt 
import seaborn as sns 
import pathlib

'''
Data fields

Date        - calendar date. 2 years of train data, and one month of test data
ts_id       - time-series identification, each of them describe sales of one SKU at one Store
isPromo     - how many types of promotion were active on this day. 0 = no promotions, 1 = a single promotion type 
              (for example, price discount), 2 = two promotion types simultaneously (for example, both price discount and special placement)
Sales       - actual sales, target. Note: some values are negative?
id          - identifier of rows in the submission file (the same order as in the test set). 
              This is the key column Kaggle uses to match the test labels and your predictions. Order matters
Store       - identifier of the stores (supermarket)
SKU         - Stock Keeping Unit. Identifier of the product
Segment     - price segment of the product
Pack        - package type
Product     - type of the drink
Brand       - product's brand
Volume      - package size
'''

class forecast_model:

    def __init__(self):

        # Loading data
        self.test_data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/beer_test.csv')      # id,       Date,       ts_id,      isPromo
        self.train_data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/beer_train.csv')    # id,       Date,       ts_id,      isPromo
        self.stores = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/id_store_sku.csv')      # ts_id,    Store,      SKU
        self.features = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/sku_features.csv')    # SKU,      Segment,    Pack,       Product,    Brand,  Volume

        self.describe()
        self.benchmark()

    def describe(self):

        print('TRAINING DATA \n\n' + 
            f'{self.train_data.head(6)}  \n\n' +
            f'{self.train_data.describe()} \n\n' + 
            'STORES \n\n' + 
            f'{self.stores.head(6)} \n\n' +
            'FEATURES \n\n' +
            f'{self.features.head(6)} \n\n' +
            f'{self.features.describe()}')
    
    def train(self):

        None

    def benchmark(self):

        submission = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/sampleSubmission.csv')

        submission['Sales'] = 0.32

        submission.to_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/submission.csv', index = False)

if __name__ == "__main__":

    f = forecast_model()
