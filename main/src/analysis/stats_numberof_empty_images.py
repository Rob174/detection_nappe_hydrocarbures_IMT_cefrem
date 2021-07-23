from h5py import File
import numpy as np

if __name__ == '__main__':
    num_empty = 0
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\annotations_labels_preprocessed.hdf5","r") as cache_annot:
        for image in cache_annot.values():
            image = np.array(image,dtype=np.float32)
            uniq_values = np.unique(image)
            if len(uniq_values) == 1 and uniq_values[0] == 0:
                num_empty += 1
        print(num_empty)