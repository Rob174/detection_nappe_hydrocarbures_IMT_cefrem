"""
Module that gather scripts to create the attr_dataset hdf5 / json files

- open_files.py: open raster function
- extract_to_hdf5.py: main script to create the images_informations_preprocessed.json, images_preprocessed.hdf5 and  annotations_labels_preprocessed.hdf5 files
- correct_overlap_annotations.py: script to fix the first code version of the extract_to_hdf5 script and take the time annotation of the label into account
- extract_stat_for_standardization.py: script to compute mean and std of the pixel of the already created hdf5 image file
"""