## Amanda Roberts 2017/08/01

#This code utilizes machine learning to make a prediction of LAI values at ORNL 
#.tif files from the BRDF flight performed there are used to "train" the Random Forest Regressor 
#The regressor then gets data from the full site and predicts LAI


#Import packages needed
import numpy as np
import gdal, osr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
   

#array2raster converts an array to a geotiff
#newRasterfn - name of the output geotiff
#rasterOrigin - the coordinates of the starting corner of the desired raster
#pixelWidth - which direction the raster should be written (-1 to the left, 1 to the right)
#pixelHeight - which direction the raster should be written (-1 down/South, 1 up/North)
#array - the array to be turned into a raster
#epsg - 
def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, epsg):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
#Create string variables for the location of the .tif files    
uncertaintyLAI = 'S:/Users/aroberts/ORNL/Spectrometer/Deblur/DerivedIndices/LAI_Results_Normal/ORNL_std.tif'
meanNDWI = 'S:/Users/aroberts/ORNL/Spectrometer/Deblur/WaterIndices/NDWI_Results_Normal/ORNL_mean.tif'
meanNDVI = 'S:/Users/aroberts/ORNL/Spectrometer/Deblur/VegetationIndices/NDVI_Results_Normal/ORNL_mean.tif'
meanLAI = 'S:/Users/aroberts/ORNL/Spectrometer/Deblur/DerivedIndices/LAI_Results_Normal/ORNL_mean.tif'
meanAlbedo = 'S:/Users/aroberts/ORNL/Spectrometer/Deblur/DerivedIndices/Albedo_Results_Normal/ORNL_mean.tif'

#Open the files with GDAL
uncertaintyLAI_dataset = gdal.Open(uncertaintyLAI)
meanNDWI_dataset = gdal.Open(meanNDWI)
meanNDVI_dataset = gdal.Open(meanNDVI)
meanLAI_dataset = gdal.Open(meanLAI)
meanAlbedo_dataset = gdal.Open(meanAlbedo)

#Get the needed metadata from the file
cols_uncertaintyLAI = uncertaintyLAI_dataset.RasterXSize
rows_uncertaintyLAI = uncertaintyLAI_dataset.RasterYSize

#Get the raster band object
uncertaintyLAI_raster = uncertaintyLAI_dataset.GetRasterBand(1)
meanNDWI_raster = meanNDWI_dataset.GetRasterBand(1)
meanNDVI_raster = meanNDVI_dataset.GetRasterBand(1)
meanLAI_raster = meanLAI_dataset.GetRasterBand(1)
meanAlbedo_raster = meanAlbedo_dataset.GetRasterBand(1)

#Get no data value
noDataVal = uncertaintyLAI_raster.GetNoDataValue()

#Make an array from the raster
uncertaintyLAI_array = uncertaintyLAI_raster.ReadAsArray(0, 0, cols_uncertaintyLAI, rows_uncertaintyLAI).astype(np.float)
meanNDWI_array = meanNDWI_raster.ReadAsArray(0, 0, cols_uncertaintyLAI, rows_uncertaintyLAI).astype(np.float)
meanNDVI_array = meanNDVI_raster.ReadAsArray(0, 0, cols_uncertaintyLAI, rows_uncertaintyLAI).astype(np.float)
meanLAI_array = meanLAI_raster.ReadAsArray(0, 0, cols_uncertaintyLAI, rows_uncertaintyLAI).astype(np.float)
meanAlbedo_array = meanAlbedo_raster.ReadAsArray(0, 0, cols_uncertaintyLAI, rows_uncertaintyLAI).astype(np.float)


#Get the shape
shape = uncertaintyLAI_array.shape

#Get how many rows are in the array
rows_for_training = int(shape[0])

#Make training data
predictor_data = uncertaintyLAI_array[:rows_for_training]
training_data_1 = meanNDWI_array[:rows_for_training]
training_data_2 = meanNDVI_array[:rows_for_training]
training_data_3 = meanLAI_array[:rows_for_training]
training_data_7 = meanAlbedo_array[:rows_for_training]

#Turn into vectors, which are needed for processing
predictor_data_vec = np.reshape(predictor_data, (rows_for_training * shape[1], 1))
training_data_1_vec = np.reshape(training_data_1, (rows_for_training * shape[1], 1))
training_data_2_vec = np.reshape(training_data_2, (rows_for_training * shape[1], 1))
training_data_3_vec = np.reshape(training_data_3, (rows_for_training * shape[1], 1))
training_data_7_vec = np.reshape(training_data_7, (rows_for_training * shape[1], 1))

#Set the values that are unrealistically high and 0 (indicitive of values too high for the equation used to derive LAI) to the 99th percentile
training_data_3_vec[training_data_3_vec == 0] = np.percentile(meanLAI_array[meanLAI_array != -9999], 99)
training_data_3_vec[training_data_3_vec > np.percentile(meanLAI_array[meanLAI_array != -9999], 99)] = \
np.percentile(meanLAI_array[meanLAI_array != -9999], 99)


#Get all the values that aren't -9999
goodValues_predictor = np.where(predictor_data_vec != noDataVal)
goodValues_training1 = np.where(training_data_1_vec != noDataVal)
goodValues_training2 = np.where(training_data_2_vec != noDataVal)
goodValues_training3 = np.where(training_data_3_vec != noDataVal)
goodValues_training7 = np.where(training_data_7_vec != noDataVal)

#Print out how many good values it has
print('Predictor good values ', np.count_nonzero(predictor_data_vec != noDataVal))
print('Training 1 good values: ', np.count_nonzero(training_data_1_vec != noDataVal))
print('Training 2 good values: ', np.count_nonzero(training_data_2_vec != noDataVal))
print('Training 3 good values: ', np.count_nonzero(training_data_3_vec != noDataVal))
print('Training 7 good values: ', np.count_nonzero(training_data_7_vec != noDataVal))

#Create a variable that records where the good values overlap
intersect = np.intersect1d(goodValues_predictor[0], goodValues_training1[0])
intersect = np.intersect1d(intersect, goodValues_training2[0])
intersect = np.intersect1d(intersect, goodValues_training3[0])
intersect = np.intersect1d(intersect, goodValues_training7[0])

print(np.count_nonzero(intersect))

#Combine all the overlapping data into one array
all_training_data = np.concatenate([training_data_1_vec[intersect], training_data_2_vec[intersect], \
                                    training_data_3_vec[intersect],training_data_7_vec[intersect]], axis=1)
predictor_data_vec = predictor_data_vec[intersect]    


#Define parameters for Random Forest Regressor; the depth is the max number of branches the forest can have
max_depth = 10

#Define regressor rules
regr_rf = RandomForestRegressor(max_depth = max_depth, random_state=2)

#Fit the data to regressor variables
regr_rf.fit(all_training_data, predictor_data_vec.flatten())

##Start getting information for the whole site files

#String variable that stores the location of the tif file
meanFullSiteNDWI = 'S:/Users/aroberts/ORNL/FullMosiac/Spectrometer/NDWI.tif'
meanFullSiteNDVI = 'S:/Users/aroberts/ORNL/FullMosiac/Spectrometer/NDVI.tif'
meanFullSiteLAI = 'S:/Users/aroberts/ORNL/FullMosiac/Spectrometer/LAI/LAI.tif'
meanFullSiteAlbedo = 'S:/Users/aroberts/ORNL/FullMosiac/Spectrometer/ALBD.tif'

#Open the files with GDAL
meanFullSiteNDWI_dataset = gdal.Open(meanFullSiteNDWI)
meanFullSiteNDVI_dataset = gdal.Open(meanFullSiteNDVI)
meanFullSiteLAI_dataset = gdal.Open(meanFullSiteLAI)
meanFullSiteAlbedo_dataset = gdal.Open(meanFullSiteAlbedo)

#Get the raster from the file
meanFullSiteNDWI_raster = meanFullSiteNDWI_dataset.GetRasterBand(1)
meanFullSiteNDVI_raster = meanFullSiteNDVI_dataset.GetRasterBand(1)
meanFullSiteLAI_raster = meanFullSiteLAI_dataset.GetRasterBand(1)
meanFullSiteAlbedo_raster = meanFullSiteAlbedo_dataset.GetRasterBand(1)

#Get the needed metadata
cols_fullSite = meanFullSiteNDWI_dataset.RasterXSize
rows_fullSite = meanFullSiteNDWI_dataset.RasterYSize

#Create an array from the raster
meanFullSiteNDWI_array = meanFullSiteNDWI_raster.ReadAsArray(0, 0, cols_fullSite, rows_fullSite).astype(np.float)
meanFullSiteNDVI_array = meanFullSiteNDVI_raster.ReadAsArray(0, 0, cols_fullSite, rows_fullSite).astype(np.float)
meanFullSiteLAI_array = meanFullSiteLAI_raster.ReadAsArray(0, 0, cols_fullSite, rows_fullSite).astype(np.float)
meanFullSiteAlbedo_array = meanFullSiteAlbedo_raster.ReadAsArray(0, 0, cols_fullSite, rows_fullSite).astype(np.float)

#Get the shape of the array
shapeFull = meanFullSiteNDWI_array.shape

#Get the number of rows in the array
fullSize = int(shapeFull[0])

#Make validation data
#validation_predictor = uncertaintyLAI_array[rows_for_training+1:] 
validation_data_1 = meanFullSiteNDWI_array[:fullSize]
validation_data_2 = meanFullSiteNDVI_array[:fullSize]
validation_data_3 = meanFullSiteLAI_array[:fullSize]
validation_data_7 = meanFullSiteAlbedo_array[:fullSize]

#Turn into vectors
#validation_predictor_vec = np.reshape(validation_predictor, ((shape[0] - rows_for_training - 1) * shape[1], 1))
validation_data_1_vec = np.reshape(validation_data_1, ((shapeFull[0]) * shapeFull[1], 1))
validation_data_2_vec = np.reshape(validation_data_2, ((shapeFull[0]) * shapeFull[1], 1))
validation_data_3_vec = np.reshape(validation_data_3, ((shapeFull[0]) * shapeFull[1], 1))
validation_data_7_vec = np.reshape(validation_data_7, ((shapeFull[0]) * shapeFull[1], 1))

#Get good data points
#goodValues_validation_predictor = np.where(np.isnan(validation_predictor_vec != noDataVal)
goodValues_validation1 = np.where(np.isfinite(validation_data_1_vec))
goodValues_validation2 = np.where(np.isfinite(validation_data_2_vec))
goodValues_validation3 = np.where(np.isfinite(validation_data_3_vec))
goodValues_validation7 = np.where(np.isfinite(validation_data_7_vec))

#print('validation predictor good values ', np.count_nonzero(goodValues_validation_predictor != noDataVal))
#print('validation 1 good values ', np.count_nonzero(np.isfinite(validation_data_1_vec))
#print('validation 2 good values ', np.count_nonzero(np.isfinite(validation_data_2_vec))
#print('validation 3 good values ', np.count_nonzero(np.isfinite(validation_data_3_vec))
#print('validation 7 good values ', np.count_nonzero(np.isfinite(validation_data_7_vec))

#Get the locations of all 
intersectVal = np.intersect1d(goodValues_validation1[0], goodValues_validation7[0])
intersectVal = np.intersect1d(intersectVal, goodValues_validation2[0])
intersectVal = np.intersect1d(intersectVal, goodValues_validation3[0])

#Concatinate all the data into one array
all_validation_data = np.concatenate([validation_data_1_vec[intersectVal], validation_data_2_vec[intersectVal], \
                                      validation_data_3_vec[intersectVal], validation_data_7_vec[intersectVal]],axis=1)
#validation_predictor_vec = validation_predictor_vec[intersectVal]

#Get a vector of predicted values for the whole site
pred_uncertaintyLAI_vec = regr_rf.predict(all_validation_data)
#difference_uncertaintyLAI_vec = validation_predictor_vec.flatten() - pred_uncertaintyLAI_vec

#Create a histogram of the predicted values
plt.hist(pred_uncertaintyLAI_vec,bins=250,histtype='step')
#plt.hist(difference_uncertaintyLAI_vec,bins=250,histtype='step')

#Create a new array of -9999 and replace them with the predicted values where the data existed before
outArray = np.zeros(len(validation_data_3_vec))
outArray = outArray - 9999
for counter in range (0, len(intersectVal)):
    outArray[intersectVal[counter]] = pred_uncertaintyLAI_vec[counter]
    
#outDifArray = np.zeros(len(validation_data_3_vec))
#outDifArray = outDifArray - 9999
#for counter in range (0, len(intersectVal)):
#    outDifArray[intersectVal[counter]] = difference_uncertaintyLAI_vec[counter]

#Make the array the right size
outArray = np.reshape(outArray, ((shapeFull[0] - rows_for_training - 1), shape[1]))
#outDifArray = np.reshape(outDifArray, ((shape[0] - rows_for_training - 1), shape[1]))

#Get he needed metadata
mapinfo = meanLAI_dataset.GetGeoTransform()
xMin = mapinfo[0]
yMax = mapinfo[3]

#Make geotiff files
array2raster('predict_LAIuncertainty_20170724_NDVI_LAI_Alb_NDWI.tif', (xMin, yMax), 1, -1, outArray, 26916) #predictions
#array2raster('predict_LAIuncertainty_difference_20170724_NDVI_lai_alb_ndwi_weight_lai_1.tif', (xMin, yMax), 1, -1, outDifArray, 26916) #difference

#Print out how important each feature was in the regressor
importances = regr_rf.feature_importances_
print(importances)

#Get the r2 value
r2_predictor = regr_rf.score(all_training_data, predictor_data_vec.flatten())