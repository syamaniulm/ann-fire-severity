# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 03:47:02 2022

@author: Syamani
"""

# Artificial Neural Network Regression

import math
import pandas as pd
import pickle
import dataframe_image as dfi
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Vegetation indices list

veg_indices = ['SAVI', 'NDVI', 'TSAVI', 'MSAVI', 'DVI', 'RVI', 'PVI', 'IPVI', 'WDVI', 'TNDVI', 'GNDVI', 'GEMI', 'ARVI', 'NDI45', 'MCARI', 'EVI', 'S2REP', 'IRECI', 'PSSRa', 'ARI', 'GLI', 'LCI', 'CVI', 'CRI550', 'CRI700']

# Reading data file

training_data = pd.read_csv('Ground_Samples/Training_Data.csv')

y_train = training_data.iloc[:,3]

validation_data = pd.read_csv('Ground_Samples/Validation_Data.csv')
y_test = validation_data.iloc[:,3]

# Batch regression analysis

nn_r2_1h = []
nn_mape_1h = []
nn_rmse_1h = []

nn_r2_2h = []
nn_mape_2h = []
nn_rmse_2h = []

nn_r2_3h = []
nn_mape_3h = []
nn_rmse_3h = []

nn_r2_4h = []
nn_mape_4h = []
nn_rmse_4h = []

nn_r2_5h = []
nn_mape_5h = []
nn_rmse_5h = []

nn_r2_6h = []
nn_mape_6h = []
nn_rmse_6h = []

for i in tqdm(range(25)):
    
    x_train = training_data.iloc[:,[i+4,29,30]]
    x_test = validation_data.iloc[:,[i+4,29,30]]
    
    # Neural network regression model with 1 hidden layer
    
    nn_reg_1h = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=1000)
    nn_reg_1h.fit(x_train, y_train)
    y_train_pred = nn_reg_1h.predict(x_train)
    y_test_pred = nn_reg_1h.predict(x_test)
    nn_r2_1h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_1h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_1h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))

    # Neural network regression model with 2 hidden layers
    
    nn_reg_2h = MLPRegressor(hidden_layer_sizes=(100,100,), random_state=1, max_iter=1000)
    nn_reg_2h.fit(x_train, y_train)
    y_train_pred = nn_reg_2h.predict(x_train)
    y_test_pred = nn_reg_2h.predict(x_test)
    nn_r2_2h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_2h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_2h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Neural network regression model with 3 hidden layers
    
    nn_reg_3h = MLPRegressor(hidden_layer_sizes=(100,100,100,), random_state=1, max_iter=1000)
    nn_reg_3h.fit(x_train, y_train)
    y_train_pred = nn_reg_3h.predict(x_train)
    y_test_pred = nn_reg_3h.predict(x_test)
    nn_r2_3h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_3h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_3h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Neural network regression model with 4 hidden layers
    
    nn_reg_4h = MLPRegressor(hidden_layer_sizes=(100,100,100,100,), random_state=1, max_iter=1000)
    nn_reg_4h.fit(x_train, y_train)
    y_train_pred = nn_reg_4h.predict(x_train)
    y_test_pred = nn_reg_4h.predict(x_test)
    nn_r2_4h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_4h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_4h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Neural network regression model with 5 hidden layers
    
    nn_reg_5h = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,), random_state=1, max_iter=1000)
    nn_reg_5h.fit(x_train, y_train)
    y_train_pred = nn_reg_5h.predict(x_train)
    y_test_pred = nn_reg_5h.predict(x_test)
    nn_r2_5h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_5h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_5h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Neural network regression model with 6 hidden layers
    
    nn_reg_6h = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100,), random_state=1, max_iter=1000)
    nn_reg_6h.fit(x_train, y_train)
    y_train_pred = nn_reg_6h.predict(x_train)
    y_test_pred = nn_reg_6h.predict(x_test)
    nn_r2_6h.append(round(r2_score(y_train, y_train_pred),4))
    nn_mape_6h.append(round(mape(y_test, y_test_pred)*100,2))
    nn_rmse_6h.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Saving neural network regression models
    
    fname_1h = './Models/' + veg_indices[i].lower() + '_nn_reg_1h_model.sav'
    fname_2h = './Models/' + veg_indices[i].lower() + '_nn_reg_2h_model.sav'
    fname_3h = './Models/' + veg_indices[i].lower() + '_nn_reg_3h_model.sav'
    fname_4h = './Models/' + veg_indices[i].lower() + '_nn_reg_4h_model.sav'
    fname_5h = './Models/' + veg_indices[i].lower() + '_nn_reg_5h_model.sav'
    fname_6h = './Models/' + veg_indices[i].lower() + '_nn_reg_6h_model.sav'
    
    pickle.dump(nn_reg_1h, open(fname_1h, 'wb'))
    pickle.dump(nn_reg_2h, open(fname_2h, 'wb'))
    pickle.dump(nn_reg_3h, open(fname_3h, 'wb'))
    pickle.dump(nn_reg_4h, open(fname_4h, 'wb'))
    pickle.dump(nn_reg_5h, open(fname_5h, 'wb'))
    pickle.dump(nn_reg_6h, open(fname_6h, 'wb'))

nn_reg_dict = {'R2 (1H)': nn_r2_1h, 'MAPE (1H)': nn_mape_1h, 'RMSE (1H)': nn_rmse_1h, 'R2 (2H)': nn_r2_2h, 'MAPE (2H)': nn_mape_2h, 'RMSE (2H)': nn_rmse_2h, 'R2 (3H)': nn_r2_3h, 'MAPE (3H)': nn_mape_3h, 'RMSE (3H)': nn_rmse_3h, 'R2 (4H)': nn_r2_4h, 'MAPE (4H)': nn_mape_4h, 'RMSE (4H)': nn_rmse_4h, 'R2 (5H)': nn_r2_5h, 'MAPE (5H)': nn_mape_5h, 'RMSE (5H)': nn_rmse_5h, 'R2 (6H)': nn_r2_6h, 'MAPE (6H)': nn_mape_6h, 'RMSE (6H)': nn_rmse_6h}
nn_reg_df = pd.DataFrame(nn_reg_dict, veg_indices, columns=['R2 (1H)', 'MAPE (1H)', 'RMSE (1H)', 'R2 (2H)', 'MAPE (2H)', 'RMSE (2H)', 'R2 (3H)', 'MAPE (3H)', 'RMSE (3H)', 'R2 (4H)', 'MAPE (4H)', 'RMSE (4H)', 'R2 (5H)', 'MAPE (5H)', 'RMSE (5H)', 'R2 (6H)', 'MAPE (6H)', 'RMSE (6H)'])

nn_reg_df_styled = nn_reg_df.style.background_gradient(cmap='YlOrRd')

nn_reg_df_styled

dfi.export(nn_reg_df_styled, 'nn_regression.png')

print('Processing completed...')