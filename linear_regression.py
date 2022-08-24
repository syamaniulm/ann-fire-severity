# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 03:47:08 2022

@author: Syamani
"""

# Multiple Linear Regression

import numpy as np
import math
import pandas as pd
import pickle
import dataframe_image as dfi
import warnings

from sklearn.linear_model import LinearRegression
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

l_intercept = []
l_coef = np.empty((25,3))
l_r2 = []
l_mape = []
l_rmse = []

for i in tqdm(range(25)):
    
    x_train = training_data.iloc[:,[i+4,29,30]]
    x_test = validation_data.iloc[:,[i+4,29,30]]
    
    # Linear regression model
    
    l_reg = LinearRegression().fit(x_train, y_train)
    l_intercept.append(l_reg.intercept_)
    l_coef[i] = l_reg.coef_
    y_train_pred = l_reg.predict(x_train)
    y_test_pred = l_reg.predict(x_test)
    l_r2.append(round(r2_score(y_train, y_train_pred),4))
    l_mape.append(round(mape(y_test, y_test_pred)*100,2))
    l_rmse.append(round(math.sqrt(mse(y_test, y_test_pred)*100),4))
    
    # Saving linear regression models
    
    fname = './Models/' + veg_indices[i].lower() + '_l_reg_model.sav'
    
    pickle.dump(l_reg, open(fname, 'wb'))
    
l_reg_dict = {'Intercept': l_intercept, 'VI Coef.': l_coef[:,0], 'NDMI Coef.': l_coef[:,1], 'PSRI Coef.': l_coef[:,2], 'R2': l_r2, 'MAPE': l_mape, 'RMSE': l_rmse}
l_reg_df = pd.DataFrame(l_reg_dict, veg_indices, columns=['Intercept', 'VI Coef.', 'NDMI Coef.', 'PSRI Coef.', 'R2', 'MAPE', 'RMSE'])

l_reg_df_styled = l_reg_df.style.background_gradient(cmap='YlOrRd', subset=['R2','MAPE', 'RMSE'])

l_reg_df_styled

dfi.export(l_reg_df_styled, 'l_regression.png')

print('Processing completed...')
