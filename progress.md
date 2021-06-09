# Progress

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish

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

- ⏲️ warp affine transformation before making the grid
  - ✔️ add transformation matrix to the cache informations
  - ⏲️ apply the transformation
- ⏲️ Main script
  - ✔️ resnet trainable
  - ✔️ progressbar training
  - ⏲️ validation
  - ⏲️ save results


## TODO

- auto backup of the parameters thanks to the attr_ prefix
- main script using the objects
- get the position of the image/patch
