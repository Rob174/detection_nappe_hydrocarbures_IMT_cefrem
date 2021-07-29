"""Convert a hdf5 file with images of shape (3,256,256) to shape (256,256) by taking the first channel only"""


if __name__ == '__main__':
    from h5py import File
    import numpy as np
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_cache_other\filtered_cache_other_annotations.hdf5","r") as cache_origin:
        print(len(cache_origin))
        with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_cache_other\filtered_cache_other_annotations_corrected.hdf5","w") as cache_write:
            for i,(k,v) in enumerate(cache_origin.items()):
                if i % 1000 == 0:
                    print(i,end="\r")
                array = np.array(v,dtype=np.float32)[0,:,:]
                cache_write.create_dataset(k,shape=array.shape,dtype='f',data=array)
