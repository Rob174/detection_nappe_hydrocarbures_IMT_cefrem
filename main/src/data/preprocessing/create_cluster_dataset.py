from h5py import File
import numpy as np
import cv2
import colorsys
from PIL import Image,ImageDraw
import matplotlib

from main.FolderInfos import FolderInfos
from main.src.data.preprocessing.correct_overlap_annotations import get_annotations,get_annotations_points
from scipy.optimize import minimize, LinearConstraint

distance = lambda point1, point2: np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def min_distance(x1,y1,x2,y2,x3,y3,x4,y4):
    A = np.array([[2*((x1-x2)**2+(y1-y2)**2),                    2*((x1-x2)*(x3-x4)+(y1-y2)*(y3-y4))],
                    [2*((x1-x2)*(x3-x4)+(y1-y2)*(y3-y4)),          2*((x3-x4)**2+(y3-y4)**2)          ]])

    B = np.array([[-2*((x1-x2)*(x2-x4)+(y1-y2)*(y2+y4))],
                    [-2*((x3-x4)*(x2-x4)+(y3-y4)*(y2+y4))]])

    # We want find the smallest distance between two point of two edges defined by their extremal points
    # One point satisfies the property that x = λx1 + (1-λ)x2 ; (idem with y) with x1,x2 (resp y1,y2) the x-(resp y) coordinates
    # of the extrimity of this vertex, with λ ∈ [0,1]
    # So we want to minimize
    #    ___________________________________________________________________________________________________________________
    #   ╱                                                     2                                                        2
    # ╲╱ ((λ1 ⋅ x1 + (1 - λ1) ⋅ x2) - (λ2 ⋅ x3 + (1 - λ2) ⋅ x4))  + ((λ1 ⋅ y1 + (1 - λ1) ⋅ y2) - (λ2 ⋅ y3 + (1 - λ2) ⋅ y4))
    # it is a non linear minimization problem with constraints. We will use trust-constr algorithm from scipy
    distance_btw_two_points_on_edges = lambda  lambdas: distance([lambdas[0]*x1+(1-lambdas[0])*x2,lambdas[0]*y1+(1-lambdas[0])*y2],
                                                                [lambdas[1]*x3+(1-lambdas[1])*x4,lambdas[1]*y3+(1-lambdas[1])*y4])
    grad = np.array
    res = minimize(distance_btw_two_points_on_edges, [0.1, 0.1], method='trust-constr', options={'disp': False},tol=0.4, # tolerance of 0.4 as we have a px distance
                   constraints=LinearConstraint(A=np.identity(2), lb=np.zeros((2,))+1e-3, ub=np.ones((2,)))) # λ ∈ [0,1]
    if res.success is False:
        raise Exception("Pb with minimization method: try to change the method used in the minimize function")
    return distance_btw_two_points_on_edges(res.x)
class Stats_
def do():

    annotations,name_to_annotations = get_annotations_points(*get_annotations())
    list_cluster_dims = []
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\annotations_labels_preprocessed.hdf5","r") as cache:
        for i, [name, img] in enumerate(cache.items()):
            print("----------------------------------------------")
            img = np.copy(np.array(img, dtype=np.uint8))
            classe = 1
            lshapes = [annotations[i]["points"] for i in name_to_annotations[name]]
            if len(lshapes) == 0:
                return None
            lclusters = []
            dist_clustering_max = 250
            for points in lshapes: # For each shape
                if len(lclusters) == 0:
                    lclusters.append([points])
                else:
                    added_to_existing_cluster = False
                    for i,cluster_list in enumerate(lclusters): # We iterate over cluster and search if this shape is nearby a cluster
                        for shape_cluster_points in cluster_list:
                            # We check if one min distance between two vertrices is lower than the threshold
                            for vertex_pt1, vertex_pt2 in zip(shape_cluster_points, shape_cluster_points[1:] + [shape_cluster_points[0]]):
                                for point1, point2 in zip(points, points[1:] + [points[0]]):
                                    try:
                                        if min_distance(*vertex_pt1, *vertex_pt2,
                                                        *point1, *point2) < dist_clustering_max:
                                            lclusters[i].append(points)
                                            added_to_existing_cluster = True
                                            break
                                    except Exception as e:
                                        print(e)
                                        raise Exception()

                                if added_to_existing_cluster is True:
                                    break
                            if added_to_existing_cluster is True:
                                break
                        if added_to_existing_cluster is True:
                            break
                    if added_to_existing_cluster is False:
                        lclusters.append([points])
            # clean_img = np.zeros((*img.shape,3),dtype=np.uint8)
            # segmentation_map = Image.fromarray(clean_img)
            # draw = ImageDraw.ImageDraw(segmentation_map)
            # get_color = lambda x:matplotlib.colors.to_hex([*colorsys.hsv_to_rgb(x/len(lclusters),1,1)])
            print(f"{len(lclusters)} clusters found")
            for i,cluster in enumerate(lclusters):
                print(f"\t- {len(cluster)} shapes : ",cluster)
                cv2.minAreaRect(np.array())
                # for shape_points in cluster:
                #     color = get_color(i)
                #     draw.polygon(shape_points, fill=color)

            #     plt.figure()
            #     plt.imshow(np.asarray(segmentation_map),cmap="gray")
            #     plt.figure()
            #     plt.imshow(img.astype(np.uint8),cmap="gray")
            #     plt.show()
            # break
    #     n = len(cache.keys())
    #     for i,[name,img] in enumerate(cache.items()):
    #         classes_to_analyse = [1,2]
    #         print(f"{i/n*100:.2f}%")
    #         img = np.copy(np.array(img,dtype=np.float32))
    #         for classe in classes_to_analyse:
    #             contours,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #             if len(contours) == 0:
    #                 continue
    #             lshapes = []
    #             # Get list of points = contours of each shape
    #             for points,hier in zip(contours,hierarchy[0]):
    #                 if hier[-1] != -1:
    #                     continue
    #                 lshapes.append(points)
    #             lclusters = []
    #             dist_clustering_max = 250
    #             for points in lshapes:
    #                 if len(lclusters) == 0:
    #                     lclusters.append([points])
    #                 else:
    #                     for i,cluster_list in enumerate(lclusters):
    #                         for points_clusteri in cluster_list:
if __name__ == "__main__":
    do()
