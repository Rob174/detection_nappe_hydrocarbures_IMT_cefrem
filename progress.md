# Progress

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem ; 🐛 bug ; 〰️ ok does the job but maybe to improve ; 🛑 pause ; 🛰️ release

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

## 21-06-2021
- ✔️dashboard bug fixes
- ✔️ training on 2 classes or 1 class
- ✔️ profiling methods
- ✔️ balance the dataset
- ✔️ shapes statistics update

## 22-06-2021
- ✔️ Important bug fix ! Training not using correct batches --> compiling version [a39e48](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/a39e48ff933e16c063e15d3c371f78308a8bdbb5)
- 🔨 Adding debug option to save training reference and output
- 🔨 get the position of the image/patch
- ✔️ vizualize property of trainings with regex filter
- ✔️ interactive visualizer of dataframe with property of trainings
- ✔️ integrated and html documentation of the code

## 23-06-2021
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

## 24-06-2021
- ⏲️ playing with learning rate
- ✔️ with / without algo balance
- ✔️ with / without algo margins
- 🔨 Augmentations
  - ✔️ mirrors
  - ✔️ rotation with step of 15°
- 🔨 patch augmentation vs image augmentation
- ✔️ Doc
- ✔️ Debug overlay

## 25-06-2021
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
## 28-06-2021
- ✔️ classificationpatch balance margins 1000 10 epoch augmentations_patch mirrors,rotation,resize_4_1.5 (more than 1 day of training)
- 🛑🔨(partially) filter dataset algorithm clusters [#27](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/27)

## 29-06-2021
- ✔️ Checking if bug in rgb_overlay
  - ok, loss checked and computed manually, gives coherent results but the model predicts always the same value for the training above (0.32438424 0.60291094 0.659301  )
  - [Possible cause](https://stackoverflow.com/questions/41881220/keras-predict-always-output-same-value-in-multi-classification): class imbalance. 
  - Possible solutions
    - Raise training time --> pb: here 100 epochs -> more than 1 day of training
      - Reduce time consumption of hdf5 cache cf [#27](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/27)
    - Adjust loss to give  more importance to seep and spills positive detection
- 🛰️ Before working to implement and generate the new hdf5 file (with balanced augmented dataset of fixed patches ready to use) release [v1.0](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/v1.0) added ``git checkout v1.0`` to switch to this branch

## 30-06-2021
- ✔️ Optimize rotation algorithm for big images with inverse transformation
  - ✔️ proof of concept
  - ✔️ integrate it to existing code (resize, rotation)
## 01-07-2021
- ✔️ Optimize rotation algorithm for big images with inverse transformation
  - ✔️ integrate it to existing code (resize, rotation)
  - ✔️ changing from getitem protocol to iter protocol (more adapted as we want to ignore some items)
    - ✔️ supporting split
## 02-07-2021

- ✔️ Optimize rotation algorithm for big images with inverse transformation
    - ✔️ progressbar problem (unknown size of the dataset -> no overall progress status possible)
    - ✔️ debug transformation warpaffine shiftings (for posterity ⚠️⚠️ order of dimensions in opencv ⚠️⚠️ not rows,cols but cols,rows)
    - ✔️ profiling to check bottlenecks
- ⏲️ Create a new hdf5 file with balanced augmented dataset of fixed patches ready to use

## 05-07-2021
- Diapo 

## 06-07-2021
- Diapo
- In parallel we clone another time the repo to continue to work on the code
- ✔️ Use composition not inheritance
- ✔️ Get a patch for diapo with not only other category
- ✔️ unfreeze all layers of the pretrained model

## 07-07-2021 08-07-2021
- Diapo...
- ✔️ Adapting training attributes extraction code to take into account preprocessing used to compute the cache
- ✔️ Allowing to try multiple fonction to retrieve training attribute extraction

## 09-07-2021
- ✔️ Spliting in objects training options (progressbar, model backup)
- ✔️ Adding abstract classes to improve typing hints possibilities
- ✔️ Adding EarlyStopping
- ✔️ Adding missing enums

## 10-07-2021
- ✔️ Profiling code
- ✔️ Update attributes mapping for analysis

# 11-07-2021
- ✔️ Make statistics of classes in cache 
- ✔️ Make mean std of images in cache
- ✔️ Comparing wwith sql request on qgis -> to further explore

# 12-07-2021
- ✔️🐛 Correction class index
- ✔️🐛 Debug class name ↔️ class value : solved: coherent predictions: the model put all patches in the predominant category (seep for the moment cf issue 30)
  - ✔️ Computing 70 epoch cache training
- 🔨 Create an other cache with only background (other class)
  - ✔️ Generating new hdf5 cache
  - 🔨 Adding interface to add determined amount of "other" patches
# 13-07-2021
- ⏲️ Add annotations dataset with points

# 15-07-2021
- ✔️ Script to correct dataset with transformation matrix
- ✔️ debugging annotations
- ✔️ setting up tests

# 16-07-2021
- ✔️ cache generation algorithm checked [aee9eab](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/tree/aee9eab67b673dfe01176e02168483e7954d1b51)
- ✔️ add tests for cache
- ✔️ correct empty annotations
- ✔️recomputing stats pixels
- ✔️recomputing stats classes
- ✔️recomputing stats pixels for other cache
- ✔️compositing stats for other seepSpill balance control [#33](https://github.com/Rob174/detection_nappe_hydrocarbures_IMT_cefrem/issues/33)
- ⏲️ Documentation
- ✔️ training cache with other (interval 1) 70 epochs
  - no significant results : same prediction for each patch
  - curves coherent with prediction (does not learn after 1st epoch flobally)
- train with lr 100 times smaller

# 19-07-2021
- ✔️ progressively add other patches
- ✔️ Debugging metadata mapping 
- ✔️ Confusion matrix
  - ✔️ Backend code 
  - ✔️ Tests
  - ✔️ Frontend code

# 20-07-2021

- ⏳ model problem debugging
- ⏲️ doc

✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem ; 🐛 bug ; 〰️ ok does the job but maybe to improve ; 🛑 pause ; 🛰️ release

## TODO

Priorities 1️⃣: high priority ; 9️⃣low priority
- 2️⃣ Update the documentation
- 9️⃣ Find a way not to write all the dict of preprocessings/metrics at each save step
- 9️⃣ Hyperparameters optimization
- Question : convert predicted annotated image back to raster ?
