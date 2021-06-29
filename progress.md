# Progress

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem ; 🐛 bug ; 〰️ ok does the job but maybe to improve

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
- ✔️ Filter trainings by parameters 
- ✔️ Redo the compressed file annotations to take into account the time codes of the annotations and avoid overlappings
   - ✔️ correct the extract raster to hdf5 file (for reproduction purpose only)
   - ✔️ make a correction script (to remake the annotation file)

## 16-06-2021
- ✔️ Exclude all patches with a margin (select the margin value (uniq, float) and if more than x pixel with this value --> exclude)
- 〰️ Add a legend to the rgb overlay (improve the link between annotations and channels)
- ✔️ Simplify the analysis backend code with pandas

## 18-06-2021
- ✔️ allow interactive buttons
- ✔️ update possibilities
- ✔️ Classify only by telling if there is something or not on the image
- bug fixes
- reflexion about optimization possibilities [#25](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/25)

Working version : [dd43dc4](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/commit/dd43dc4e8fd941a7738dc0a238647b0923182c8b)

- ✔️ Classify only by using 2 probabilities (seep or spill) --> if vector output (0,0) --> other

## 21-06-06
- ✔️dashboard bug fixes
- ✔️ training on 2 classes or 1 class
- ✔️ profiling methods
- ✔️ balance the dataset
- ✔️ shapes statistics update

## 22-06-06
- ✔️ Important bug fix ! Training not using correct batches --> compiling version [a39e48](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/a39e48ff933e16c063e15d3c371f78308a8bdbb5)
- 🔨 Adding debug option to save training reference and output
- 🔨 get the position of the image/patch
- ✔️ vizualize property of trainings with regex filter
- ✔️ interactive visualizer of dataframe with property of trainings
- ✔️ integrated and html documentation of the code

## 23-06-06
As no parameters produces better performances, we will investigate:
- deeper models:
  - ✔️ vgg16
- 🚩 test concurrent training with terminal launch
  - Training time explodes from ~2h to ~15h for 2 trainings in parallel (maybe caused by memory problem or concurrent access to hdf5 file)
- ✔️ Standardize data
  - ✔️ Compute statistics (mean and standard deviation) of the global dataset
  - ✔️ Use these statistics to apply standardization
- ✔️ correct filter metadata problem commit [f02d1c5f37](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/f02d1c5f374ef1097dfffb22332bc535664b1056)
  - ✔️ simplification
  - ✔️ works
  - 

## 24-06-06
- ⏲️ playing with learning rate
- ✔️ with / without algo balance
- ✔️ with / without algo margins
- 🔨 Augmentations
  - ✔️ mirrors
  - ✔️ rotation with step of 15°
- 🔨 patch augmentation vs image augmentation
- ✔️ Doc
- ✔️ Debug overlay

## 25-06-06
- ✔️ Launching a debugging session
- ✔️ ⚠️⚠️ Major bug fix on reject system (balance and margin exclusion) [f728e82d](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/commit/f728e82ddc4962d4d15f4c9bfd449c27a5ce428f)
- ✔️ Resize augmentation
- ✔️ ⚠️⚠️ Major bug fix for **multi epoch** reject system (balance and margin exclusion) [e4ab323
](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/e4ab3238bde1c383b4d197c147ca64eeb76fa00f)
- tests:
  - ✔️ balance margins 1000 1 epoch
  - ✔️ nobalance margins 100000000 1 epoch
  - ✔️ nobalance margins 1000 1 epoch
  - ✔️ balance margins 1000 10 epoch
## 28-06-06
- ✔️ classificationpatch balance margins 1000 10 epoch augmentations_patch mirrors,rotation,resize_4_1.5 (more than 1 day of training)
- 🔨 filter dataset algorithm clusters [#27](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/27)

## 29-06-06
- Checking if bug in rgb_overlay
  - ok, loss checked and computed manually, gives coherent results but the model predicts always the same value for the training above (0.32438424 0.60291094 0.659301  )
  - [potential causes](https://stackoverflow.com/questions/41881220/keras-predict-always-output-same-value-in-multi-classification): class imbalance. 
  - Potential solutions
    - Raise training time --> pb: here 100 epochs -> more than 1 day of training
      - Reduce time consumption of hdf5 cache cf [#27](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/27)
    - Adjust loss to give  more importance to seep and spills positive detection
- **classificationpatch1** balance margins 1000 10 epoch augmentations_patch mirrors,rotation,resize_4_1.5
- balance margins 1000 10 epoch augmentations_img mirrors,rotation,resize_1_0.1
- balance margins 1000 10 epoch augmentations_img mirrors,rotation,resize_1_0.1 augmentations_patch mirrors,rotation,resize_4_1.5

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem ; 🐛 bug ; 〰️ ok does the job but maybe to improve

## TODO

Priorities 1️⃣: high priority ; 9️⃣low priority

- Training to launch:
  - unfreeze all layers of the pretrained model
