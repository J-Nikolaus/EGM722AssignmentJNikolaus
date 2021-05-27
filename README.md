# EGM722AssignmentJNikolaus
 
**Introduction **

Many archeological sites today are severely threatened by landscape changes, oftentimes instigated by anthropogenic factors. Geospatial analysis can help to detect and monitor those landscape changes and the subsequent threat and damage they may present to archaeological sites. The use of Python programming for GIS and remote sensing applications can significantly speed up the process of monitoring sites in many different locations. Methods like this are even more important and relevant in areas that are difficult or dangerous to access. This code can help heritage professionals to monitor the condition of known sites, to create buffer zones around the sites and to calculate changes in agricultural and urban expansion. To demonstrate the functionality of the code, the case study focuses on the area in and around the oasis town of Hun in Libya, where sites have already been documented and threat and damage assessment have been conducted. 

**Dependencies, Packages and Installation**
This code uses anaconda to install packages and modules. Conda will need to be downloaded and installed on the computer.
This code is written for Python 3.8.10. To check the correct version is installed the following code can be run:
Import sys
Print(sys.version)
The code for this assignment is stored on GitHub: 
https://github.com/J-Nikolaus/EGM722AssignmentJNikolaus

The envirnment.yml holds the relevant dependencies for Anaconda: 
Dependencies:
  - python=3.8.8
  - geopandas=0.8.1
  - cartopy=0.18.0
  - rasterio=1.1.1
  - numpy=1.20
  - matplotlib=3.4.2
Anaconda is a package management system that helps to manage and to deploy packages in Python. Anaconda includes ‘Conda’ the open source environment and package management system, as well as Anaconda Navigator, a desktop program through which conda packages can be installed and environments can be managed. For this assignment, Anaconda was used to create the environment of this project, and to subsequently install and manage the packages that will be deployed in the code.
Anaconda Navigator has to be installed to create the yml. Environment or to run the installation commands for the individual packages below. (https://docs.anaconda.com/).

**Data provided**
-   	(CSV_Sites_Hun) CSV file that holds information about site locations and the condition of sites
-   	(UrbanExtent1991; UrbanExtent2020) Shapefile of outline of Hun town in 1991 and 2020 
-   	
**Data not provided**
-   	The landsat images used in this study are available from the USGS server https://earthexplorer.usgs.gov/
-   	2020: Landsat 8
ID: LC08_L2SP_186040_20201202_20210312_02_T1
Date Acquired: 2020/12/02
Path: 186
Row: 040
 
2001: Landsat 5
ID: LT05_L2SP_186040_20011230_20200905_02_T1
Date Acquired: 2001/12/30
Path: 186
Row: 040
