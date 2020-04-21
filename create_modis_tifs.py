import gdal
import struct
from osgeo import osr
import os
import pyproj

path = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\\MODIS\\'
files = []
for filename in os.listdir(path):
    if filename.endswith(".hdf"):
        files.append(filename)

for f in files:
    print(f)
    hdf_file = gdal.Open(path+f)
    subDatasets = hdf_file.GetSubDatasets()
    
    # open the NDVI band
    dataset = gdal.Open(subDatasets[0][0])
    meta = dataset.GetMetadata_Dict()
    nx = int(meta['DATACOLUMNS'])
    ny = int(meta['DATAROWS'])
    
    gt = dataset.GetGeoTransform()
    p1 = dataset.GetProjection()
    
    out_name = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\\MODIS_tif\\%s.tif' % f[:-4]
    
    #p2 = pyproj.Proj("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    
    # write to geo-tiff image
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(out_name, nx, ny, 1, gdal.GDT_Float32)
    out.SetGeoTransform(gt)
    out.SetProjection(p1)
    x=dataset.ReadAsArray()
    out.GetRasterBand(1).WriteArray(x)
    out.FlushCache()

    
