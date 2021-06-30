
from scipy.ndimage import affine_transform
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_test\test_shape_characteristics.jpg")
    # Approach 1
    rows,cols,_ = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((rows/2,cols/2),-45,1)
    translate = np.array([[1,0,rows/2],
                      [0,1,cols/2],
                      [0,0,1]])
    rotation_centered_matrix = np.concatenate((cv2.getRotationMatrix2D((0,0),-45,1),[[0,0,1]]))
    # Center the desired patch
    coord_upper_left = np.array([10000.,10000.]).T
    size = 1000
    shift_patch = np.identity(3)
    shift_patch[:-1,-1] = coord_upper_left+size/2
    transformation_matrix = translate.dot(rotation_centered_matrix).dot(shift_patch)[:-1,:]
    result = cv2.warpAffine(img,transformation_matrix,dsize=(size,)*2)
    import matplotlib.pyplot as plt
    plt.imshow(result,cmap="gray")
    plt.show()