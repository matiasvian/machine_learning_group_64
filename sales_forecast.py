import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt 
import seaborn as sns 
import pathlib
import sklearn
import statsmodels.tsa.arima.model as arima
import math
import warnings
warnings.filterwarnings('ignore')

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
        self.test_data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/beer_test.csv', parse_dates=['Date'])      # id,       Date,       ts_id,      isPromo
        self.train_data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/beer_train.csv', parse_dates=['Date'])    # id,       Date,       ts_id,      isPromo,    Sales
        self.stores = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/id_store_sku.csv')      # ts_id,    Store,      SKU
        self.features = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/sku_features.csv')    # SKU,      Segment,    Pack,       Product,    Brand,  Volume


    def run(self):
        
        self.train_features = self.generate_features(self.train_data, self.features, self.stores)
        self.test_features = self.generate_features(self.test_data, self.features, self.stores)

        # self.describe()
        # self.benchmark()

    def describe(self):

        print('TRAINING DATA \n\n' + 
            f'{self.train_data.head(6)}  \n\n' +
            f'{self.train_data.describe()} \n\n' + 
            'STORES \n\n' + 
            f'{self.stores.head(6)} \n\n' +
            'FEATURES \n\n' +
            f'{self.features.head(6)} \n\n' +
            f'{self.features.describe()}')

    def benchmark(self):

        submission = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/sampleSubmission.csv')

        submission['Sales'] = 0.32

        submission.to_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/submission.csv', index = False)

    def submit(self, y_pred):

        submission = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/sampleSubmission.csv')

        submission['Sales'] = y_pred

        submission.to_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data/submission.csv', index = False)

    def arima(self, q = 1, d = 0, p = 0, length = 192882, submit = True):
        '''
        Predicting *length* amount in the future based on previous observations all in one
        
        :param (q, d, p): order
        :param length: length of future observations
        '''
        
        y_prev = np.array(self.train_data['Sales'][:-length])

        # fit model
        model = arima.ARIMA(y_prev, order = (q, d, p))
        model_fit = model.fit()
        output = model_fit.predict(end=length-1)

        # Post processing negative values
        output = np.array(output)
        output[output < 0] = 0.2  # Why 0.2? Mean value, but lower

        if submit : self.submit(output)

        return self.root_mean_squared_log_error(np.array(self.train_data['Sales'][-length:]), output)

    def arima_traverse(self, leap, length = 192882):
        '''
        Will predict *leap* number of predictions at a time and add them to the training pool, rinse repeat
        Doesn't work very well and takes forever to run...
        
        :param leap: number of predictions at a time
        :param length: length of total predictions
        '''

        predictions = np.array([])
        training = np.array(self.train_data['Sales'])

        for x in range(math.floor(length/leap)):
            
            print(f'{x} / {math.floor(length/leap)}')
            
            # fit model
            model = arima.ARIMA(training, order = (1, 0, 0))  # Good orders are (1, 0, 0) and (2, 1, 1)
            model_fit = model.fit()
            output = model_fit.predict(end=leap-1)
            
            # Re-add predictions to training pool
            predictions = np.append(predictions, output)
            training = np.append(training, output)

            print(f'Prediction length: {len(predictions)}')
            
        
        # Remainder
        remainder = length - len(predictions)
        model = arima.ARIMA(training, order = (1, 0, 0))
        model_fit = model.fit()
        output = model_fit.predict(end=remainder-1)
        predictions = np.append(predictions, output)

        # Post processing negative values
        predictions[predictions < 0] = 0.2  # Why 0.2? Mean value, but lower
        
        # Submitting
        self.submit(predictions)

        return self.root_mean_squared_log_error(np.array(self.train_data['Sales'][-length:]), predictions)

    def root_mean_squared_log_error(self, y_true, y_pred):
        '''
        RMSE

        :param y_true: Goal 
        :param y_pred: Predictions
        '''

        sum = 0
        skip = 0
        for x, y in  zip(y_true[-len(y_pred):], y_pred):
            if (x <= 0 or y <= 0):
                skip += 1
            else:
                sum += ((math.log1p(1 + y) - math.log1p(1 + x)) ** 2)

        return math.sqrt(sum/len(y_pred-skip))

    def generate_features(self, df, sku_features, id_map):
        '''
        Feature generation from Introduction notebook.
        '''
        
        # Add metadata
        df = pd.merge(df, id_map, how='left', on='ts_id')
        df = pd.merge(df, sku_features, how='left', on='SKU')

        # Time features
        df['day_of_month'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.weekday
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['week'] = df['Date'].dt.week
        
        # Enlarge promo features
        # Since we know that promo is important
        
        df['ts_promo'] = df['ts_id'].astype(str) + df['isPromo'].astype(str)
        df['store_promo'] = df['Store'].astype(str) + df['isPromo'].astype(str)
        df['segment_promo'] = df['Segment'].astype(str) + df['isPromo'].astype(str)
        df['brand_promo'] = df['Brand'].astype(str) + df['isPromo'].astype(str)
        df['sku_promo'] = df['SKU'].astype(str) + df['isPromo'].astype(str)
        
        df['dom_promo'] = df['day_of_month'].astype(str) + df['isPromo'].astype(str)
        df['dow_promo'] = df['day_of_week'].astype(str) + df['isPromo'].astype(str)
        
        return df


if __name__ == "__main__":
    
    f = forecast_model()
    f.run()
