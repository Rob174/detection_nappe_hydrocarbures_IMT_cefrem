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
* la matrice indique : la résolution en px vers coordonnées gps, les coordonnées de base....
* header offset : nombre d'octets représentant le header dans le fichier `.img`


#### 1.1.2 Les labels

Il s'agit de 1 000 images segmentées manuellement

Pour indiquer quelles zones de l'images appartiennent à quelle catégorie, les chercheurs du CEFREM ont tracer des polygones sur l'image et indiqué à quel catégorie les pixels à l'intérieur de polygone appartiennent. On fourni alors les coordonnées en px et en coordonnées "gps" des points du polygone dans un fichier `.shp` ou `.shx` de qgis (avec QGIS). Les catégories de chaque forme sont stockées dans les fichiers `.dbf`

Ces fichiers peuvent être ouverts dans python à l'aide du package ... **#TOASK**

Il est alors nécessaire de déterminer en python quels pixels appartiennent à quelles classes d'après ces polygones. **#TODO**

**#TODO : vérifier que je peux ouvrir les fichiers**

#### 1.1.3 Notes

- Le réseau pourra éventuellement avoir des difficultés pour distinguer :
    - les 2 types de nappes d'hydrocarbures
    - des filements de plancton de nappes d'hydrocarbures (la différence de changement d'intensité peut tout de même permettre de distinguer ces 2 types de catégories : :warning: diminution de résolution de l'image)
    - Il sera possible d'ajouter d'autres catégories en appliquant éventuellement un seuillage aux images (par ex pour le plancton
- On pourra dans un second temps également utilisé le niveau de confiance de chaque classification de polygone en utilisant la colone `ìndice` du fichier `.dbs`
- Les images font environ 25 000 px de côté, cela posera peut-être des problème de mémoire (à tester)
   - Solutions envisageables :
      - Découper l'image en patchs (500px par ex ok mais pas moins) : ⚠️ à ne pas trop couper de motifs
      - Dégrader la résolution : ⚠️ + haut

### 1.2 Objectifs

- Segmenter l'image, indiquer pour chaque pixel à quel catégorie il appartient

Il y a 3 catégories (que l'on nommera classes) possibles (à cet état initial, qui ont été annotées)
* Type de pétrole 1 : spill pétrole provenant de bâteaux
* Type de pétrole 2 : seep pétrole naturel 
* Le reste

- Ne pas perdre les informations gps et de résolution (la 2e pr debugguer) 
- Il sera également envisageable de ramener la moyenne des pixels de la mer sur toutes les images à une même valeur

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

## 3. Réunions

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
