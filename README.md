# Détection de nappes d'hydrocarbures à l'aide de réseaux de neurones

## 1. Etapes du stages

### 1.1 Récupération des données

#### 1.1.1 Les images d'entrée du réseau

On pourra ouvrir le couple fichier `.hdr` et `.img` avec la librairie `rasterio` (voir script démo **#TOASK : script demo)

Les images originales se présentent sous forme d'images raster se présentant sous forme de 2 fichiers : 

- un fichier `.img`

C'est l'image en format binaire

- un fichier `.hdr`

C'est le header contenant les métadonnées de l'image : 
(documentation [ici](https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html#:~:text=The%20ENVI%20header%20file%20contains,hdr.))
* map info : origine, orientation, et taille d'un pixel , potentiellement
  - Projection name
  - Reference (tie point) pixel x location (in file coordinates)
  - Reference (tie point) pixel y location (in file coordinates)
  - Pixel easting
  - Pixel northing
  - x pixel size
  - y pixel size
  - Projection zone (UTM only)
  - North or South (UTM only)
  - Datum
  - Units

* la matrice indique : la résolution en px vers coordonnées gps, les coordonnées de base....
* header offset : nombre d'octets représentant le header dans le fichier `.img`
.... **#TOASK**

#### 1.1.2 Les labels

Il s'agit de .... images segmentées manuellement

Pour indiquer quelles zones de l'images appartiennent à quelle catégorie, les chercheurs du CEFREM? ont tracer des polygones sur l'image et indiqué à quel catégorie les pixels à l'intérieur de polygone appartiennent. On fourni alors les coordonnées en px et en coordonnées "gps" des points du polygone dans un fichier ... **#TOASK**

Ces fichiers peuvent être ouverts dans python à l'aide du package ... **#TOASK**

Il est alors nécessaire de déterminer en python quels pixels appartiennent à quelles classes d'après ces polygones. **#TODO**

**#TODO : vérifier que je peux ouvrir les fichiers**

### 1.2 Objectifs

- Segmenter l'image, indiquer pour chaque pixel à quel catégorie il appartient

Il y a .... catégories (que l'on nommera classes) possibles
* Type de pétrole 1 : pétrole provenant de bâteaux **#TOASK : nom précis**
* Type de pétrole 2 : pétrole naturel **#TOASK : nom précis**
* Bâteaux ? **#TOASK**
* Le reste

- Faire correspondre chaque zone à des coordonnées "gps" **# TOASK : nom précis des coordonnées**

### 1.2 Classification de patchs d'images

On prend l'image, on la sépare en plus petites zones (que l'on nommera patchs) et on doit indiquer pour chaque zone si une certaine classe est présente. Comme plusieurs classes peuvent être présentent sur un même patch, on pourra par exemple prédire la probabilité que chaque classe soit présente. Ainsi, pour chaque classe on prédira la probabilité que cette classe soit présente ou non. On aura alors un vecteur de sortie de la forme :

<!-- $$
\begin{bmatrix}
           P_{\in\; classe\;1}(patch) \\ \vdots \\ P_{\in\; classe\;m}(patch)
         \end{bmatrix}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D%0D%0A%20%20%20%20%20%20%20%20%20%20%20P_%7B%5Cin%5C%3B%20classe%5C%3B1%7D(patch)%20%5C%5C%20%5Cvdots%20%5C%5C%20P_%7B%5Cin%5C%3B%20classe%5C%3Bm%7D(patch)%0D%0A%20%20%20%20%20%20%20%20%20%5Cend%7Bbmatrix%7D%0D"></div>

Il sera également nécessaire de déterminer comment est-ce que l'on réalisera les patchs. **#TODO**

### 1.3 Segmentation d'une image 

On prend cette fois la totalité de l'image et on veut que le réseau indique pour chaque pixel à quelle classe il appartient. Pour cela on repartira du réseau **#TOASK**

## 2. Environnement de développement

### 2.1 Packages python utilisés

|Package|Utilisation|
|:---:|:---:|
|Pytorch (torch)|Réalisation et utilisation des réseaux de neurones|
|rasterio|Lecture des images|
