# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 05:28:45 2022

@author: Syamani
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings

from osgeo import gdal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Reading image files

print('Reading image files...')

lr_veg_image = gdal.Open('./Images/mcari.tif')
nn_veg_image = gdal.Open('./Images/ireci.tif')
ndm_image = gdal.Open('./Images/ndmi.tif')
psr_image = gdal.Open('./Images/psri.tif')
mask_image = gdal.Open('./Images/mask.tif')

# Reading image parameters

img_proj = lr_veg_image.GetProjection()
geotransform = lr_veg_image.GetGeoTransform()
img_height = lr_veg_image.RasterYSize
img_width = lr_veg_image.RasterXSize

# Reading image arrays

lr_veg_band = lr_veg_image.GetRasterBand(1)
nn_veg_band = nn_veg_image.GetRasterBand(1)
ndm_band = ndm_image.GetRasterBand(1)
psr_band = psr_image.GetRasterBand(1)
mask_band = mask_image.GetRasterBand(1)

lr_veg_index = lr_veg_band.ReadAsArray()
nn_veg_index = nn_veg_band.ReadAsArray()
ndm_index = ndm_band.ReadAsArray()
psr_index = psr_band.ReadAsArray()
mask = mask_band.ReadAsArray()

# Resolving nan and infinity pixel values

lr_veg_index[np.isnan(lr_veg_index)] = 0
lr_veg_index[np.isinf(lr_veg_index)] = 0

nn_veg_index[np.isnan(nn_veg_index)] = 0
nn_veg_index[np.isinf(nn_veg_index)] = 0

ndm_index[np.isnan(ndm_index)] = 0
ndm_index[np.isinf(ndm_index)] = 0

psr_index[np.isnan(psr_index)] = 0
psr_index[np.isinf(psr_index)] = 0

# Image tiling

print('Tiling images...')

tile_size = int(img_height/10)
row_num = int(img_height/tile_size)
column_num = int(img_width/tile_size)
tile_num = int(row_num*column_num)

lr_vi_image_row = np.empty((column_num,tile_size,tile_size))
nn_vi_image_row = np.empty((column_num,tile_size,tile_size))
ps_image_row = np.empty((column_num,tile_size,tile_size))
vm_image_row = np.empty((column_num,tile_size,tile_size))
lr_vi_image_tiles = np.empty((tile_num,tile_size,tile_size))
nn_vi_image_tiles = np.empty((tile_num,tile_size,tile_size))
ps_image_tiles = np.empty((tile_num,tile_size,tile_size))
vm_image_tiles = np.empty((tile_num,tile_size,tile_size))

for i in tqdm(range(row_num)):
    for j in range(column_num):
        lr_vi_image_row[j,:,:] = lr_veg_index[i*tile_size:i*tile_size+tile_size,j*tile_size:j*tile_size+tile_size]
        nn_vi_image_row[j,:,:] = nn_veg_index[i*tile_size:i*tile_size+tile_size,j*tile_size:j*tile_size+tile_size]
        vm_image_row[j,:,:] = ndm_index[i*tile_size:i*tile_size+tile_size,j*tile_size:j*tile_size+tile_size]
        ps_image_row[j,:,:] = psr_index[i*tile_size:i*tile_size+tile_size,j*tile_size:j*tile_size+tile_size]
    lr_vi_image_tiles[i*column_num:i*column_num+column_num] = lr_vi_image_row
    nn_vi_image_tiles[i*column_num:i*column_num+column_num] = nn_vi_image_row
    vm_image_tiles[i*column_num:i*column_num+column_num] = vm_image_row
    ps_image_tiles[i*column_num:i*column_num+column_num] = ps_image_row

del lr_vi_image_row, nn_vi_image_row, ps_image_row, vm_image_row

# Loading regression models

print('Loading regression models...')

l_reg = pickle.load(open('./Models/mcari_l_reg_model.sav', 'rb'))
nn_reg = pickle.load(open('./Models/ireci_nn_reg_6h_model.sav', 'rb'))

feature_11 = 'MCARI'
feature_12 = 'IRECI'
feature_2 = 'NDMI'
feature_3 = 'PSRI'

# Regression analysis

print('Regression analysis...')

dBAIS2_l_reg_output = np.empty((tile_num,tile_size,tile_size))
dBAIS2_nn_reg_output = np.empty((tile_num,tile_size,tile_size))

for tile in tqdm(range(tile_num)):
    x_train_1 = np.reshape(lr_vi_image_tiles[tile], (tile_size**2))
    x_train_2 = np.reshape(nn_vi_image_tiles[tile], (tile_size**2))
    x_train_3 = np.reshape(vm_image_tiles[tile], (tile_size**2))
    x_train_4 = np.reshape(ps_image_tiles[tile], (tile_size**2))

    lr_veg_index_dict = {feature_11: x_train_1, feature_2: x_train_3, feature_3: x_train_4}
    
    lr_veg_index_df = pd.DataFrame(lr_veg_index_dict)
    
    nn_veg_index_dict = {feature_12: x_train_2, feature_2: x_train_3, feature_3: x_train_4}
    
    nn_veg_index_df = pd.DataFrame(nn_veg_index_dict)

    l_reg_output = l_reg.predict(lr_veg_index_df)

    dBAIS2_l_reg_output[tile] = np.reshape(l_reg_output, (tile_size, tile_size))

    nn_reg_output = nn_reg.predict(nn_veg_index_df)

    dBAIS2_nn_reg_output[tile] = np.reshape(nn_reg_output, (tile_size, tile_size))

del lr_veg_index_dict, lr_veg_index_df, nn_veg_index_dict, nn_veg_index_df, lr_vi_image_tiles, nn_vi_image_tiles, vm_image_tiles, ps_image_tiles

print('Regression analysis completed...')

print('Merging output image tiles...')

dBAIS2_l_reg_predict = np.empty((img_height,img_width))
dBAIS2_nn_reg_predict = np.empty((img_height,img_width))

l_image_line = np.empty((column_num,tile_size,tile_size))
l_array_line = np.empty((tile_size,img_width))
nn_image_line = np.empty((column_num,tile_size,tile_size))
nn_array_line = np.empty((tile_size,img_width))

for i in tqdm(range(row_num)):
    l_image_line = dBAIS2_l_reg_output[i*column_num:(i+1)*column_num]
    nn_image_line = dBAIS2_nn_reg_output[i*column_num:(i+1)*column_num]
    for j in range(column_num):
        l_array_line[:,j*tile_size:j*tile_size+tile_size] = l_image_line[j]
        nn_array_line[:,j*tile_size:j*tile_size+tile_size] = nn_image_line[j]
    dBAIS2_l_reg_predict[i*tile_size:i*tile_size+tile_size,0:(i+1)*column_num*tile_size] = l_array_line
    dBAIS2_nn_reg_predict[i*tile_size:i*tile_size+tile_size,0:(i+1)*column_num*tile_size] = nn_array_line

del l_image_line, l_array_line, nn_image_line, nn_array_line, dBAIS2_l_reg_output, dBAIS2_nn_reg_output

dBAIS2_l_reg_predict = dBAIS2_l_reg_predict * mask
dBAIS2_nn_reg_predict = dBAIS2_nn_reg_predict * mask

print('Writing prediction images to files...')

# Setting neural network regression prediction output file path and file name

nn_reg_name = './Prediction_Output/nn_reg_predictive_dbais2.tif'
nn_reg_raster = gdal.GetDriverByName("GTiff").Create(nn_reg_name, img_width, img_height, 1, gdal.GDT_Float32)
nn_reg_raster.GetRasterBand(1).WriteArray(dBAIS2_nn_reg_predict)

# Updating neural network regression prediction output image georeference

nn_reg_raster.SetGeoTransform(geotransform)
nn_reg_raster.SetProjection(img_proj)
nn_reg_raster.FlushCache()

nn_reg_raster = None

# # Setting linear regression prediction output file path and file name

# l_reg_name = './Prediction_Output/l_reg_predictive_dbais2.tif'
# l_reg_raster = gdal.GetDriverByName("GTiff").Create(l_reg_name, img_width, img_height, 1, gdal.GDT_Float32)
# l_reg_raster.GetRasterBand(1).WriteArray(dBAIS2_l_reg_predict)

# # Updating linear regression prediction output image georeference

# l_reg_raster.SetGeoTransform(geotransform)
# l_reg_raster.SetProjection(img_proj)
# l_reg_raster.FlushCache()

# l_reg_raster = None

print('Writing image files completed...')

print('Classifying fire severity for visualization...')

fire_severity_classes = [0.1, 0.27, 0.44, 0.66]
usgs_fire_severity_class = ['U', 'L', 'Ml', 'Mh', 'H']

l_reg_fire_severity = np.digitize(dBAIS2_l_reg_predict, bins=fire_severity_classes)
nn_reg_fire_severity = np.digitize(dBAIS2_nn_reg_predict, bins=fire_severity_classes)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

ax[0].set_title('Linear Regression Predictive Fire Severity', fontsize=20)
ax[0].axis('off')
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.1)
cbar0 = plt.colorbar(ax[0].imshow(l_reg_fire_severity+1, cmap='jet'), cax=cax0)
cbar0
cbar0.ax.tick_params(labelsize=20)
cbar0.set_ticks([x for x in range(1, len(usgs_fire_severity_class)+1)])
tick_labels0 = usgs_fire_severity_class
cbar0.set_ticklabels(tick_labels0)

ax[1].set_title('Neural Net. Regression Predictive Fire Severity', fontsize=20)
ax[1].axis('off')
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
cbar1 = plt.colorbar(ax[1].imshow(nn_reg_fire_severity+1, cmap='jet'), cax=cax1)
cbar1
cbar1.ax.tick_params(labelsize=20)
cbar1.set_ticks([x for x in range(1, len(usgs_fire_severity_class)+1)])
tick_labels1 = usgs_fire_severity_class
cbar1.set_ticklabels(tick_labels1)

plt.show()