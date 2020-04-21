import gdal
import struct
from osgeo import osr
import os
import pyproj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('C:\\Users\\comp\\anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin')
sys.path.append('C:\\Users\\comp\\anaconda3\\Library\\bin')
sys.path.append('C:\\Users\\comp\\anaconda3\\Library\\bin\\graphviz')
sys.path.append('C:\\Users\\comp\\anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin\\graphviz')


xmin = -80
xmax = -40
ymin = -40
ymax = -10

fm = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\\MODIS_tif\\merged_MODIS_v8.tif'
fv = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\merged_south_america_5mo.tif'
samples = pd.read_csv('C:\\Users\\comp\\Documents\\MODIS_machine_learning\\training_data.csv')
samples.drop(columns=['Unnamed: 3'], inplace = True)
samples = samples[samples['X'] != 0.0]
samples.dropna(inplace = True)

modis = gdal.Open(fm)
m = modis.ReadAsArray()

viirs = gdal.Open(fv)
v = viirs.ReadAsArray()

samples['vi'] = 0.
samples['rad'] = 0.
samples['xpos'] = 0
samples['ypos'] = 0
for i,row in samples.iterrows():
    xpos = int(np.rint(((row['X'] - xmin) / (xmax -xmin)) * 9600))-1
    ypos = 7200 - int(np.rint(((row['Y'] - ymin) / (ymax -ymin)) * 7200))-1
    samples.loc[i,'xpos'] = xpos
    samples.loc[i,'ypos'] = ypos
    samples.loc[i,'vi'] = v[ypos][xpos]
    samples.loc[i,'rad'] = m[ypos][xpos]

samples['class'] = np.where(samples['urban'] == 0.0, 'n', 'u')

training = samples.sample(frac=0.5, random_state = 753)
validation = samples.drop(index = training.index)

# %% fit model
# extract 

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(min_samples_split = 10)
# Train the model on training data
rf.fit(training[['vi','rad']], training['class']);

# %% validate

validation['preds'] = rf.predict(validation[['vi','rad']])
validation['correct'] = np.where(validation['preds'] != validation['class'],'incorrect','correct')
validation.groupby(['correct'])['correct'].count()

# %% predict

# xtmax = training['xpos'].max()
# xtmin = training['xpos'].min()
# ytmax = training['ypos'].max()
# ytmin = training['ypos'].min()

# m_sub = m[ytmin:ytmax,xtmin:xtmax]
# v_sub = v[ytmin:ytmax,xtmin:xtmax]

# shp = np.shape(m_sub)

shp = np.shape(m)

m_rs = np.reshape(m,(shp[0]*shp[1],1))
v_rs = np.reshape(v,(shp[0]*shp[1],1))

to_predict = np.concatenate([v_rs,m_rs],axis=1)
to_predict_2 = np.nan_to_num(to_predict, nan=0)

# Use the forest's predict method on the test data
prediction = rf.predict(to_predict_2)

pred_rs = np.reshape(prediction,shp) 

pred_num = np.where(pred_rs == 'u',1,0)

driver = gdal.GetDriverByName("GTiff")
out = driver.Create("predicted_output.tif", 9600, 7200, 1, gdal.GDT_UInt16)
out.SetGeoTransform((-80,40/9600,0,-10,0,-30/7200))
wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
out.SetProjection(wkt)
out.GetRasterBand(1).WriteArray(pred_num)
out.FlushCache()

# %% validate the model

from sklearn.tree import export_graphviz

estimator = rf.estimators_

from sklearn.tree import export_graphviz
# Export as dot file

export_graphviz(estimator,
                feature_names=['1','2'],
                filled=True,
                rounded=True)

os.system('dot -Tpng tree.dot -o tree.png')

#fn=data.feature_names
#cn=data.target_names
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = ['rad','ndvi'], 
               class_names=['u','n'],
               filled = True);
plt.show()
fig.savefig('rf_individualtree.png')

import graphviz 

dot_data = tree.export_graphviz(rf.estimators_[0], out_file='tree.dot',
                                feature_names=['rad','ndvi'],
                                class_names=rf.classes_,
                                filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_10samp.png', '-Gdpi=600'])

# plt.imshow(pred_num)
# plt.show()

# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')