import sys

sys.path.append('C:\\Users\\comp\\anaconda3\\Lib\\site-packages\\GDAL-2.3.3-py3.7-win-amd64.egg-info\\scripts')
sys.path.append('C:\\Users\\comp\\anaconda3\\Library\\bin')
#import gdalwarp as gw
import os

import subprocess


path = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\\MODIS_tif\\'


files = []
for filename in os.listdir(path):
    print(filename)
    if filename.endswith("max_vi.tif"):
        files.append(path + filename)
        
c = [r"C:\Users\comp\anaconda3\Library\bin\gdalwarp.exe",
     "-s_srs",
     "\"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs\"",
     "-t_srs",
     "\"EPSG:4326\""]+ files + ['merged_MODIS_v6.tif']
  
subprocess.check_call(c)

st = ''
for i in c:
    st += i + ' '
#gw.main(['','-o'] + files + ['merged_MODIS.tif'])

# working command line call:
# gdalwarp -s_srs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs" -t_srs "EPSG:4326" -te -80 -40 -40 -10 -ts 9600 7200