import gdal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import struct
from bitstring import BitArray


fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 
            'Int32':'i', 'Float32':'f', 'Float64':'d'}

# read files
path = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\MODIS\\'

tiles = ["h10v10","h11v10","h11v11","h11v12","h12v10","h12v11","h12v12","h13v10","h13v11","h13v12", "h14v10", "h14v11",]
for t in tiles:
    files = []
    for filename in os.listdir(path):
        if filename.endswith(".hdf") and t in filename:
            files.append(filename)
    
    # number of scanlines to read across all files
    chunk = 400
    output = np.ones([2400, 2400])
    vi_m = np.ones([chunk,len(files),2400])
    for r in np.arange(0,2400, chunk):
        print('TILE: %s CHUNK: %d' % (t,r))
        for f in np.arange(0,len(files)):
            print(files[f])
            hdf_file = gdal.Open(path+files[f])
            subDatasets = hdf_file.GetSubDatasets()
            
            # open vi
            vi = gdal.Open(subDatasets[0][0])
            
            meta = vi.GetMetadata_Dict()
            
            
            vt = int(meta['VERTICALTILENUMBER'])
            ht = int(meta['HORIZONTALTILENUMBER'])
            
            vi_band = vi.GetRasterBand(1)
            
            vi_BandType = gdal.GetDataTypeName(vi_band.DataType)
            
            gt = vi.GetGeoTransform()
            p = vi.GetProjection()
            
            y = r
            
            
            # read in scanlines and store in vi_m matrix
            for c in np.arange(chunk):
                vi_scan = vi_band.ReadRaster(0,int(y + c),vi_band.XSize,1,vi_band.YSize,1,vi_band.DataType)

                vi_values = struct.unpack(fmttypes[vi_BandType] * vi_band.XSize, vi_scan)
                for i in np.arange(len(vi_values)):
                    vi_m[c][f][i] = vi_values[i]
                        
        # perform time averaging for each scanline
        for c in np.arange(chunk):
            output[r + c] = np.nanmean(vi_m[c], axis = 0)
            
    # write to tif 
    for i in np.arange(3):
        outname = 'C:\\Users\\comp\\Documents\\MODIS_urbanization\\DATA\MODIS_tif\\' + t + '_max_vi.tif'
        driver = gdal.GetDriverByName("GTiff")
        out = driver.Create(outname, 2400, 2400, 1, gdal.GDT_Float32)
        out.SetGeoTransform(gt)
        out.SetProjection(p)
        out.GetRasterBand(1).WriteArray(output)
        out.FlushCache()
            
            # break
        
            # for i in np.arange(0,len(scanline),2):
            #     try:
            #         cm[y][int(i/2)]=bin(scanline[i])[-2:]
            #     except:
            #         cm[y][int(i/2)] = 0
            
            # for i in np.arange(len(values)):
            #     mask = (values[i] >> 5) & 3
            #     cm[y][i] = mask
            #     if mask >= 2:
            #         num_rejected += 1
                    
                
                    
                    
        
        
        # li.append([num_rejected, date])
        
        #break

# driver = gdal.GetDriverByName("GTiff")
# out = driver.Create('cc_test_dnb.tif', 2400, 2400, 1, gdal.GDT_Float32)
# out.SetGeoTransform((west,10/2400,0,north,0,-10/2400))
# dataset.SetProjection("none")
# x=dataset.ReadAsArray()
# out.GetRasterBand(1).WriteArray(x)
# dataset.FlushCache()


# df = pd.DataFrame(li, columns = ['rejected', 'date'])
# df['date'] = pd.to_datetime(df['date'])

# plt.rcParams.update({'font.size': 14})
# fig, ax = plt.subplots(1, 1, figsize=(12, 5))
# locator = mdates.AutoDateLocator(minticks=20, maxticks=40)
# formatter = mdates.ConciseDateFormatter(locator)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)
# plt.ylabel('% Pixels Rejected')


# ax.plot(df['date'],df['rejected']/(2400*2400)*100)
# #plt.plot([df['date'].min(),df['date'].max()],[20,20],'--r',label='filter threshold')
# ax.legend(loc = 1)
# ax.set_xlim([np.datetime64('2019-06'), np.datetime64('2019-12')])
# ax.set_title('Rejected Cloud Pixels')

