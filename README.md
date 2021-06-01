# Oil detection with neural networks

## 1. Data

### 1.1 Network input data

The `rasterio` will allow the user to open raster files which consists of 3 files per image : `.hdr`, `.img` and `.bsd` :

- one file `.img`

It is the image written in a binary format

- one file `.hdr`

It is the header containing the metadata of the image :
(documentation [here](https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html#:~:text=The%20ENVI%20header%20file%20contains,hdr.))


#### 1.2 Labels

There are 4 000 images manually segmented 

To highlight which regions of the image belong to which category, CEFREM members have drawn polygons on the image with qgis and indicated at which category pixels inside them belong. We have access to their vertices with pixel coordinates or gps coordinates with the  `.shp` or `.shx` files. We can find the caategory of each polygon in the  `.dbf` file.

Ces fichiers peuvent être ouverts dans python à l'aide du package ... **#TODO**

So, it is necessary to determine with python to which category belong which pixels thanks to the polygons.

**#TODO : vérifier que je peux ouvrir les fichiers**

#### 1.3 Notes

- The network may have difficulties to distinguish:
    - the 2 types of oil discharge (spill or seep)
    - plankton can create strings that look like oil discharge. However, they can be differentiated by their edges: the transition of luminosity on the image between the sea and an oil discharge will be sharper than with the plankton. That is why diminishing the resolution of the image can be a problem
    - We will eventually add more categories as plankton by thresholding the image.
- The confidence level for each polygon is also available in the `.dbs` under the `ìndice` column. It can be interesting to integrate it in future models.
- The "raw" (before any preprocessing of **this** project) images are 25 000 px width. It might become a bottleneck for the training (to be tested)
   - Potential solutions :
      - Cut the image into patches (500 px widthpar for instance, not less): ⚠️ be cautious not to cut too many oil discharge features
      - Reduce the resolution: ⚠️ potential problems with the plankton

## 2 Objectives

### 2.1 General objectives
- Create a segmentation map of the image

There are 3 categories (classes) possible (at the initial state of the project)
* Oil discharge type 1: oil spill: caused by boats
* Oil discharge type 2 : seep: natural discharge
* The rest

- Be able to tell the coordinates of each input image and their resolution (for eventual debugging purposes (plankton))
- TODO later: Keep the mean of the sea part to the same level for each image

### 2.2 Image patches classification

We take the image and split it into smaller regions (named patches). The network then has to tell which class is present on this image. As several classes may be on the same patch, we can predict the probability that each class is on the image. Thus, we will have the following output vector :

<!-- $$
\begin{bmatrix}
           P_{\in\; classe\;1}(patch) \\ \vdots \\ P_{\in\; classe\;m}(patch)
         \end{bmatrix}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0D%0A%20%20%20%20%20%20%20%20%20%20%20P_%7B%5Cin%5C%3B%20classe%5C%3B1%7D(patch)%20%5C%5C%20%5Cvdots%20%5C%5C%20P_%7B%5Cin%5C%3B%20classe%5C%3Bm%7D(patch)%0D%0A%20%20%20%20%20%20%20%20%20%5Cend%7Bbmatrix%7D%0D"></div>

It will be necessary to determine what method to use to create the patches. **#TODO**

### 2.3 Image segmentation

This time, we take the image and we want that the network outputs the category of each pixel. To reach this goal we will start with a premade network, available at [this link] (https://github.com/bonlime/keras-deeplab-v3-plus)

## 3. Development environment

### 3.1 Python packages used
Python 3.7 : mandatory for windows users : allows to use rasterio

|Package|Utilisation|
|:---:|:---:|
|Pytorch (torch)|Neural network|
|rasterio and GDAL|To read raster files (cf [#3](https://github.com/Rob174/detection_nappe_hydrocarbures_inria_cefrem/issues/3))|

## 4. Réunions

- jeudi réunion zoom

- mardi 01/06 au CEFREM

Liste de questions : 
1. Script sample pour ouvrir les rasters -> cf stackoverflow
2. Sample de d'image .hdr pour savoir à quoi correspond chaque label -> ok
3. Confirmation : annotation = polygones qui permettront à terme de déterminer pour chaque pixel de l'image si il apprtient à telle ou telle catégorie -> ok
4. Quels sont les catégories possibles (les 2 types de rejets de pétroles notamment) ? -> nappes d'hydrocarbures (seep (naturelles) spill (artificielles)), ou pas
5. Quels fichiers contiennent les annotations ? -> shp et éventuellement shx mais à voir ce qu'il faut à python
6. Pistes sur comment les ouvrir ?
7. Est-ce que 1 zone d'1 image peut avoir plusieurs annotations différentes ? -> non
8. Parlé de fichiers annotation corrompus : est-ce que c'est bon maintenant et quels fichiers faut-il prendre ? -> tt bon ; De l'ordre du To de données 
9. Confirmation du planning : 1. Classification de patchs ; 2. Segmentation d'images complètes -> ok

Question transversale à répondre : quelle est l'utilité de relu et des fonctions d'activation ?
