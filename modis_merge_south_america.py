import sys
sys.path.append('C:\\Users\\comp\\anaconda3\\Lib\\site-packages\\GDAL-2.3.3-py3.7-win-amd64.egg-info\\scripts')
import gdal_merge as gm
import os



path = 'DATA\\VIIRS_south_america_v2_tif\\'
files = []
for filename in os.listdir(path):
    print(filename)
    if filename.endswith("5mo.tif"):
        files.append(path + filename)
  
gm.main(['','-o', 'merged_south_america_5mo.tif'] + files)