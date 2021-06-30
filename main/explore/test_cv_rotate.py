"""Script that compares two algorithm to extract a rotate patch of a big image
Supposition: On my laptop with ~16GB RAM I cannot store 2 times an array of dimensions ( 14600,14600,3)

Input image:
The rotated image of a white image with a small rectangle (that can be fitted in memory multiple times (> 5))

1st algorithm:
Approach 1:
create a transformation array composing a rotation of 45° (translation,rotation,and translation cf issue) with another translation
and then ask only to compute the result image with a window size of the size of the patch

Approach 2:
rotate the image
slice the patch

Difference:
In approach 2 (on the contrary of approach 1) opencv is required to create a second full size version of the input image
 which is impossible (cf supposition)

Test procedure:
- Comment one of the approach section and uncomment the other
- If the approach 1 is executed no memory must be reported
- If the approach 2 is executed a memory must be reported
"""
from scipy.ndimage import affine_transform
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = np.zeros((14600,14600,3),dtype=np.float32)
    rows,cols,_ = img.shape
    shift = 2500
    size = 100
    img[shift:shift+size,shift:shift+size,:] = 255
    transformation_matrix = cv2.getRotationMatrix2D((rows/2,cols/2),-45,1)
    img = cv2.warpAffine(img,transformation_matrix,dsize=(cols,rows))

    # Approach 1
    translate = np.array([[1,0,rows/2],
                      [0,1,cols/2],
                      [0,0,1]])
    rotation_centered_matrix = np.concatenate((cv2.getRotationMatrix2D((0,0),45,1),[[0,0,1]]))
    # Center the desired patch
    coord_upper_left = np.array([-shift,-shift]).T
    shift_patch = np.identity(3)
    shift_patch[:-1,-1] = coord_upper_left
    transformation_matrix = translate.dot(rotation_centered_matrix).dot(shift_patch)[:-1,:]
    transformation_matrix_without_transl = cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 1)#.dot(shift_patch)[:-1,:]
    rotation_matr = np.concatenate((cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 1),[[0,0,1]]),axis=0) #2500,2500
    transformation_matrix_without_transl1 = shift_patch.dot(rotation_matr)[:-1,:]
    # result = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # result1 = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # result2 = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # result3 = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # result4 = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # result5 = cv2.warpAffine(img,transformation_matrix_without_transl1,dsize=(size,size))
    # # # Approach 2
    transformation_matrix = cv2.getRotationMatrix2D((rows/2,cols/2),45,1)
    result = cv2.warpAffine(img,transformation_matrix,dsize=(cols,rows))
    result1 = cv2.warpAffine(img,transformation_matrix,dsize=(cols,rows))
    result2 = cv2.warpAffine(img,transformation_matrix,dsize=(cols,rows))
    print()
