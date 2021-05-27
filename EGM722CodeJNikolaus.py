############## Set up and importing packages/modules#############################

#check that the correct version of Python is installed (Python 3.8)
import sys
print(sys.version)

#import modules needed to convert CSV into shp, plot shp, draw buffer
import pandas as pd
import geopandas as gpd
from cartopy.feature import ShapelyFeature
from shapely.geometry import Point
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio import plot

######################### csv ###########################

#Import and read CSV
df_Hun = pd.read_csv(r'INSERT FILEPATH FOR CSV HERE')
print (df_Hun)

#Geopandas to dataframe

df_Hun.crs = 'epsg:4326'

geometry=[Point(xy) for xy in zip(df_Hun["Long"], df_Hun["Lat"])]

geo_df = gpd.GeoDataFrame(df_Hun,
                          crs = 'epsg:4326',
                          geometry = geometry)
geo_df.head()
print (geo_df)

#create a shapefile to open in GIS software
geo_df.to_file('Condition.shp')

#create buffer around points
geo_df.crs = 'epsg:4326'
geo_df = geo_df.to_crs(epsg=3395)
buffer = geo_df.buffer(250, cap_style=1)

#creating map and adding figure

points = gpd.read_file('C:INSERT CONDITION.SHP FILE PATH HERE')

myFig = plt.figure(figsize=(8, 8)) # size of figure created 8 x 8
myCRS = ccrs.UTM(33) # Universal Transverse Mercator reference system to transform data
ax = plt.axes (projection=ccrs.Mercator()) #plots data by creating an axes object in figure based on Mercator projection

Condition_points = ShapelyFeature(points['geometry'], myCRS, edgecolor='black', facecolor='darkblue')
xmin, ymin, xmax, ymax = points.total_bounds
ax.add_feature(Condition_points)
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)



# generate matplotlib handles to create a legend of the features we put in our map.
def generate_handles(labels, colors, edge='k', alpha=1):
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

# create a scale bar of length 20 km in the upper right corner of the map
def scale_bar(ax, location=(0.92, 0.95)):
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]

    tmc = ccrs.TransverseMercator(sbllx, sblly)
    x0, x1, y0, y1 = ax.get_extent(tmc)
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    plt.plot([sbx, sbx - 20000], [sby, sby], color='k', linewidth=9, transform=tmc)
    plt.plot([sbx, sbx - 10000], [sby, sby], color='k', linewidth=6, transform=tmc)
    plt.plot([sbx-10000, sbx - 20000], [sby, sby], color='w', linewidth=6, transform=tmc)

    plt.text(sbx, sby-4500, '20 km', transform=tmc, fontsize=8)
    plt.text(sbx-12500, sby-4500, '10 km', transform=tmc, fontsize=8)
    plt.text(sbx-24500, sby-4500, '0 km', transform=tmc, fontsize=8)


############# calculate Urban area ##############

#import shapefile of urban area 2001
Urban_2001 = gpd.read_file (r'INSERT SHAPEFILE UrbanExtent2001 HERE')
print (Urban_2001.head())

#Select 5 rows in shapefile
selection = Urban_2001[0:4]

#iterate over rows and print area of polygon
for index, row in selection.iterrows():
    poly_area = row['geometry'].area
    print("Urban area 2001 {index} is: {area:.3f}".format(index=index, area=poly_area))

#Shapfile 2001: create new column named 'area' and assign area of polygons to it, showing max, min and mean area

Urban_2001['area'] = Urban_2001.area
print(Urban_1991['area'].head())

#repoject data to crs to get meter as coordinates
Urban_2001.crs = 'epsg:4326'
Urban_2001 = Urban_2001.to_crs(epsg=3395)

# Maximum area
max_area = Urban_2001['area'].max()
# Minimum area
min_area = Urban_2001['area'].min()
# Mean area
mean_area = Urban_2001['area'].mean()

print("Max area: {max}\nMin area: {min}\nMean area: {mean}".format(max=round(max_area, 2), min=round(min_area, 2), mean=round(mean_area, 2)))

#import shapefile of urban area 2020
Urban_2020 = gpd.read_file (r'INSERT FILEPATH FOR UrbanExtent2020 SHAPEFILE HERE')
print (Urban_2020.head())

#Select 5 rows in shapefile
selection1 = Urban_2020[0:4]

#iterate over rows and print area of polygon
for index, row in selection1.iterrows():
    poly_area = row['geometry'].area
    print("Urban area 2020 {index} is: {area:.3f}".format(index=index, area=poly_area))

#Shapfile 2020: create new column named 'area' and assign area of polygons to it, showing max, min and mean area
Urban_2020['area'] = Urban_2020.area
print(Urban_2020['area'].head())

#repoject data to crs to get meter as coordinates
Urban_2020.crs = 'epsg:4326'
Urban_2020 = Urban_2020.to_crs(epsg=3395)

# Maximum area
max_area = Urban_2020['area'].max()
# Minimum area
min_area = Urban_2020['area'].min()
# Mean area
mean_area = Urban_2020['area'].mean()

print("Max area: {max}\nMin area: {min}\nMean area: {mean}".format(max=round(max_area, 2), min=round(min_area, 2), mean=round(mean_area, 2)))



########################### Landsat5 ############################

# add Landsat5 image 2001 and display bands
datasetLT5 = rio.open (r'INSERT FILEPATH FOR LANDSAT LT05 HERE.tif')

print('{} opened in {} mode'.format(datasetLT5.name,datasetLT5.mode))
print('image has {} band(s)'.format(datasetLT5.count))
print('image size (width, height): {} x {}'.format(datasetLT5.width, datasetLT5.height))
print('band 1 dataype is {}'.format(datasetLT5.dtypes[0]))

#check georeferencing information - should crrorspond to EPSG:32633 = WGS84 UTM Zone 33N
print (datasetLT5.bounds)
#check coordinate reference system (CRS)
print (datasetLT5.crs)

#Load Landsat5 2001
img = datasetLT5.read()
print(img.shape)
print(img[5])

#Only display image in window 60km around central point Landsat5 2001
centeri, centerj = datasetLT5.height // 2, datasetLT5.width // 2
centerx, centery = datasetLT5.transform * (centerj, centeri)
print(datasetLT5.index(centerx, centery))
print((centeri, centerj) == datasetLT5.index(centerx, centery))

top, lft = datasetLT5.index(centerx-6000, centery+6000)
bot, rgt = datasetLT5.index(centerx+6000, centery-6000)

subset = datasetLT5.read(window=((top, bot), (lft, rgt)))


#Display Landsat5, create new acropy CRS object, create Mmatplolib Axes object
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection=myCRS))

#Display Landsat5 as an image on a 2D regular raster using Near Infared Band (Band 4)
ax.imshow(img[3], cmap='gray', vmin=200, vmax=5000)


########################### Landsat8 ############################
# add Landsat8 image 2021 and display bands
datasetLC8 = rio.open (r'INSERT FILEPAT FOR lc08_hun.tif HERE')
print('{} opened in {} mode'.format(datasetLC8.name,datasetLC8.mode))
print('image has {} band(s)'.format(datasetLC8.count))
print('image size (width, height): {} x {}'.format(datasetLC8.width, datasetLC8.height))
print('band 1 dataype is {}'.format(datasetLC8.dtypes[0]))

#check georeferencing information
print (datasetLC8.bounds)

#check coordinate reference system (CRS) - should crrorspond to EPSG:32633 = WGS84 UTM Zone 33N
print (datasetLC8.crs)

#Load Landsat8 2021
img1 = datasetLC8.read()
print(img1.shape)
print(img1[6])

#Only display image in window 60km around central point Landsat8
centeri, centerj = datasetLC8.height // 2, datasetLC8.width // 2 # note that centeri corresponds to the row, and centerj the column
centerx, centery = datasetLC8.transform * (centerj, centeri) # note the reversal here, from i,j to j,i
print(datasetLC8.index(centerx, centery))
print((centeri, centerj) == datasetLC8.index(centerx, centery)) # check that these are the same

top, lft = datasetLC8.index(centerx-6000, centery+6000)
bot, rgt = datasetLC8.index(centerx+6000, centery-6000)

subset = datasetLC8.read(window=((top, bot), (lft, rgt))) # format is (top, bottom), (left, right)

datasetLC8.close() # remember to close the dataset now that we're done with it.

#Display Landsat8, create new acropy CRS object, create Mmatplolib Axes object
myCRS1 = ccrs.UTM(33)
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection=myCRS))

#Display Landsat8 as an image on a 2D regular raster and Near Infared Band (Band 5)
ax.imshow(img1[4], cmap='gray', vmin=200, vmax=5000)


def percentile_stretch(image, pmin=0., pmax=100.):
    '''
        C:INSERT FILEPAT FOR lc08_hun HERE

    '''
    # check pmin < pmax,  are between 0, 100
    if not 0 <= pmin < pmax <= 100:
        raise ValueError('0 <= pmin < pmax <= 100')
    # set image as 2D
    if not image.ndim == 2:
        raise ValueError('Image can only have two dimensions (row, column)')

    minval = np.percentile(image, pmin)
    maxval = np.percentile(image, pmax)

#Image strech
    stretched = (image - minval) / (maxval - minval)  #
    stretched[image < minval] = 0
    stretched[image > maxval] = 1

    return stretched


def img_display(image, ax, bands, stretch_args=None, **imshow_args):
    '''
    C:INSER FILEPAT FOR lc08_hun HERE
    '''
    dispimg = image.copy().astype(np.float32)

    for b in range(image.shape[0]):  # loop over each band, stretching using percentile_stretch()
        if stretch_args is None:  # if stretch_args is None, use the default values for percentile_stretch
            dispimg[b] = percentile_stretch(image[b])
        else:
            dispimg[b] = percentile_stretch(image[b], **stretch_args)

    # transpose the image to re-order the indices
    dispimg = dispimg.transpose([1, 2, 0])

    #  Display image
    handle = ax.imshow(dispimg[:, :, bands], **imshow_args)

    return handle, ax

my_kwargs = {'extent': [xmin, xmax, ymin, ymax],
             'transform': myCRS}

my_stretch = {'pmin': 0.1, 'pmax': 99.9}

h, ax = img_display(img, ax, [2, 1, 0], stretch_args=my_stretch, **my_kwargs)

#export Landsat8 image
datasetLC8 = rio.open ('C:INSERT FILEPATH TO STORE NEW LANDSAT8 IMAGE HERE,'w',driver='Gtiff',
                          width=datasetLC8.width,height = datasetLC8.height,
                          count=1,
                          crs=datasetLC8.crs,
                          transform=datasetLC8.transform,
                          dtype='float64')

datasetLC8.close()


############## callculate NDVI of Landsat5 ########################
#import relevant bands
band3 = rio.open (r'INSERT FILEPATH FOR LT05_L2SP_186040_20011230_20200905_02_T1_SR_B3.tif LANDSAT5 BAND 3 HERE') #red
band4 = rio.open(r'INSERT FILEPATH FOR LT05_L2SP_186040_20011230_20200905_02_T1_SR_B4.tif LANDSAT 5 BAND 4 HERE ') #nir

#plot band
plot.show(band3)

#raster values as matrix array
band3.read(1)

#multiple band representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot.show(band3, ax=ax1, cmap='Blues') #red
plot.show(band4, ax=ax2, cmap='Blues') #nir
fig.tight_layout()

#generate nir and red objects as arrays in float64 format
red = band3.read(1).astype('float64')
nir = band4.read(1).astype('float64')

print (nir)

#ndvi calculation, empty cells or nodata cells are reported as 0
ndvi=np.where(
    (nir+red)==0.,
    0,
    (nir-red)/(nir+red))

#export ndvi image
ndviImage = rio.open ('INSERT FILEPATH WERE ndvi2001Image.tiff WILL BE STORED','w',driver='Gtiff',
                          width=band3.width,height = band3.height,
                          count=1,
                          crs=band3.crs,
                          transform=band3.transform,
                          dtype='float64')
ndviImage.write(ndvi,1)
ndviImage.close()

#plot ndvi
ndviImage = rio.open('INSERT FILEPATH WERE NEW IMAGE ndvi2001Image.tiff IS STORED')
fig = plt.figure(figsize=(18,12))
plot.show(ndvi)



################################# calculate NDVI Landsat8 ###############################################

#import relevant bands
band4 = rio.open('INSERT FILE PATH LC08_L2SP_186040_20201202_20210312_02_T1_SR_B4.TIF OF LANDSAT8 BAND 4 HERE ') #red
band5 = rio.open('C:INSERT FILE PATH LC08_L2SP_186040_20201202_20210312_02_T1_SR_B5.TIF OF LANDSAT8 BAND 5 HERE ') #nir


#plot band
plot.show(band4)

#raster values as matrix array
band4.read(1)

#multiple band representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot.show(band4, ax=ax1, cmap='Blues') #red
plot.show(band5, ax=ax2, cmap='Blues') #nir
fig.tight_layout()

#generate nir and red objects as arrays in float64 format
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')

print (nir)

#ndvi calculation, empty cells or nodata cells are reported as 0
ndvi1=np.where(
    (nir+red)==0.,
    0,
    (nir-red)/(nir+red))


#export ndvi image
ndviImage1 = rio.open ('INSERT FILEPATH WERE ndvi2020Image.tiff WILL BE STORED'','w',driver='Gtiff',
                          width=band4.width,height = band4.height,
                          count=1,
                          crs=band4.crs,
                          transform=band4.transform,
                          dtype='float64')
ndviImage1.write(ndvi1,1)
ndviImage1.close()

#plot ndvi
ndviImage1 = rio.open('INSERT FILEPATH WERE NEW IMAGE ndvi2020Image.tiff IS STORED')
fig = plt.figure(figsize=(18,12))
plot.show(ndvi1)








