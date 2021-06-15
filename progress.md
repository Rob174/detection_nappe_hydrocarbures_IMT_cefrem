# Progress

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem

## 07-06-2021

✔️ Cache transfert

✔️Patch creation algorithm 
- ✔️ tests

✔️ Object to save data 

✔️ Classification dataset with fixed pixel grid
- ✔️Order to process patches
- ✔️DatasetFactory 

## 08-06-2021
- ✔️ Reduce patch size 
- ✔️ support resolution information backup
- ✔️ Create Dataloader object
- ✔️ Support for Resnet18, VGG16 (EfficientNetv4)
- ✔️ Resolution statistics with patch size of 1000 px and output size of 255
- ✔️ Resolution statistics with patch size of 500 px and output size of 255


## 09-06-2021

- :triangular_flag_on_post: warp affine transformation before making the grid
  - ✔️ add transformation matrix to the cache informations
  - ✔️ apply the transformation
- ✔️ Main script
  - ✔️ resnet trainable
  - ✔️ progressbar training
  - ✔️ validation
  - ✔️ save results

## 10-06-2021

- ✔️ First training
- ✔️ Coloration map
- ✔️ Vizualization system

## 11-06-2021
- :email: Mail inverse_transform matrix
- ✔️ Interactive plot
- ✔️ save model
- ✔️ show result on rgb map
- Confusion matrix

## 14-06-2021
- :triangular_flag_on_post: experimentations with the transform matrix
   --> does not allow to properly rotate the image : just a translation and a flip + scaling in this matrix
- ✔️ RGB overlay debugging
- ✔️ Tests to determine the method to extract seeps and spills length statistics
- ✔️ Extract seeps and spills length statistics

## 15-06-2021
- ✔️ Write RGB overlay
- ✔️ Training Crossentropy
- ✔️ Training MSE
- Filter trainings by parameters

## TODO

- get the position of the image/patch
- Confusion matrix
- Redo the compressed file annotations to take into account the time codes of the annotations and avoid overlappings
- Classify only by telling if there is something or not on the image
- Classify only by using 2 probabilities (seep or spill) --> if vector output (0,0) --> other
- Add a legend to the rgb overlay
- Exclude all patches with a margin (select the margin value (uniq, float) and if more than x pixel with this value --> exclude)
- Rotation augmentation
