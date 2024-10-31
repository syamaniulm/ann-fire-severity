# Wildfire Severity Prediction Using ANN Regression
Wildfire severity prediction models using artificial neural network regression.

### Python prerequisites:
Python 3.9 or higher

### Package Requirements:

numpy <br/>
pandas <br/>
scikit-learn <br/>
math <br/>
gdal <br/>
matplotlib <br/>
pickle <br/>
tqdm <br/>

These codes are the output of a research entitled GeoAI for Disaster Mitigation: Fire Severity Prediction Models using Sentinel-2 and ANN Regression (https://ieeexplore.ieee.org/document/9993515). The research was conducted on the island of Borneo, specifically South Kalimantan, Indonesia. Therefore, the model was trained with the characteristics of vegetation and fires in this area. If you use this model in an area that has vegetation and fire characteristics similar to South Kalimantan, then you can directly use the pre-trained model. However, if you use the model in an area that has different vegetation and fire characteristics, then you are highly recommended to retrain the model in your study area.<br/>

### Fire Severity Prediction:

If you want to directly predict fire severity based on the pre-trained model, you can directly run fire_severity_prediction.py.

### Retraining Regression Models:

If you want to retrain the model, you can run the neural_network_regression.py file.

### Citation Guide (IEEE):

S. D. Ali et al., "GeoAI for Disaster Mitigation: Fire Severity Prediction Models using Sentinel-2 and ANN Regression," 2022 IEEE International Conference on Aerospace Electronics and Remote Sensing Technology (ICARES), Yogyakarta, Indonesia, 2022, pp. 1-7, doi: 10.1109/ICARES56907.2022.9993515.
