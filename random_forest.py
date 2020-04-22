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
    samples.loc[i,'vi'] = m[ypos][xpos]
    samples.loc[i,'rad'] = v[ypos][xpos]

samples['class'] = np.where(samples['urban'] == 0.0, 'n', 'u')

training = samples.sample(frac=0.5, random_state = 753)
validation = samples.drop(index = training.index)

# %% plot sample statistics

# pd.melt(samples, columns ='class')
# samples.pivot(columns='class')
samples['r_bin'] = pd.cut(samples['rad'], bins = np.arange(0,500,50))
samples['v_bin'] = pd.cut(samples['vi'], bins = np.arange(0,10000,1000))

sg = samples.groupby(['r_bin','class'])['class'].count()
sgu = sg.unstack()
sgu.reset_index(inplace = True)
sgu['mid'] = sgu['r_bin'].apply(lambda x : x.mid)

sg_v = samples.groupby(['v_bin','class'])['class'].count()
sgu_v = sg_v.unstack()
sgu_v.reset_index(inplace = True)
sgu_v['mid'] = sgu_v['v_bin'].apply(lambda x : x.mid)

#sgu.plot(kind='bar', stacked = True)

plt.bar(sgu['mid'],sgu['n'],label='non-urban')
plt.bar(sgu['mid'],sgu['u'],label='urban')
plt.yscale('log')
plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as py
#animals=['giraffes', 'orangutans', 'monkeys']

# fig = make_subplots(rows=1, cols=2)

# fig.add_trace(
#     go.Figure(data=[
#     go.Bar(name='Non-urban', x=np.arange(24), y=sgu['n']),
#     go.Bar(name='Urban', x=np.arange(24), y=sgu['u'])]),
#     row=1, col=1
# )

trace1 = go.Bar(x=np.arange(25,525,50), y=sgu['n'],showlegend=False, legendgroup='group2', marker_color='#31bb56')
trace2 = go.Bar(x=np.arange(25,525,50), y=sgu['u'],showlegend=False, legendgroup='group1', marker_color='#48a4ce')

trace3 = go.Bar(name='Non-Urban',x=np.arange(500,10500,1000), y=sgu_v['n'],legendgroup='group2', marker_color='#31bb56')
trace4 = go.Bar(name='Urban',x=np.arange(500,10500,1000), y=sgu_v['u'],legendgroup='group1', marker_color='#48a4ce')

fig = py.subplots.make_subplots(rows=1, cols=2, subplot_titles=("Radiance", "NDVI"))

fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3,1,2)
fig.append_trace(trace4,1,2)

fig.update_yaxes(type="log", row=1, col=1)

fig.update_layout(barmode='stack', margin={"r":0,"t":20,"l":0,"b":0})

fig.write_html("plot_rad_ndvi_cdn.html", include_plotlyjs = 'cdn')


fig['layout'].update(height=600, width=600)
#iplot(fig)

fig = go.Figure(data=[
    go.Bar(name='Non-urban', x=np.arange(24), y=sgu['n']),
    go.Bar(name='Urban', x=np.arange(24), y=sgu['u']),
    go.Bar(name='Non-urban', x=np.arange(20), y=sgu_v['n']),
    go.Bar(name='Urban', x=np.arange(20), y=sgu_v['u'])
])
# Change the bar mode
fig.update_layout(barmode='stack', margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_html("plot.html")

# %% fit model
from sklearn.ensemble import RandomForestClassifier

# initialize model
rf = RandomForestClassifier(min_samples_split=10, max_depth=5)#

# fir the model
rf.fit(training[['vi','rad']], training['class']);

# %% validate model
validation['preds'] = rf.predict(validation[['vi','rad']])
validation['test'] = np.where(validation['preds'] != validation['class'],'incorrect','correct')
validation.groupby(['test'])['test'].count()

# %% predict

# xtmax = training['xpos'].max()
# xtmin = training['xpos'].min()
# ytmax = training['ypos'].max()
# ytmin = training['ypos'].min()

# m_sub = m[ytmin:ytmax,xtmin:xtmax]
# v_sub = v[ytmin:ytmax,xtmin:xtmax]

# shp = np.shape(m_sub)

shp = np.shape(m)

rad_rs = np.reshape(v,(shp[0]*shp[1],1))
ndvi_rs = np.reshape(m,(shp[0]*shp[1],1))

to_predict = np.concatenate([ndvi_rs,rad_rs],axis=1)
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

# %% plot tree

# from sklearn.tree import export_graphviz

# estimator = rf.estimators_

# from sklearn.tree import export_graphviz
# # Export as dot file

# export_graphviz(estimator,
#                 feature_names=['1','2'],
#                 filled=True,
#                 rounded=True)

# os.system('dot -Tpng tree.dot -o tree.png')

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
from sklearn import tree

dot_data = tree.export_graphviz(rf.estimators_[0], out_file='tree.dot',
                                feature_names=['ndvi','rad'],
                                class_names=rf.classes_,
                                filled=True, rounded=True,
                                special_characters=True)  

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_minsamp_10.png', '-Gdpi=100'])

# plt.imshow(pred_num)
# plt.show()

# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')