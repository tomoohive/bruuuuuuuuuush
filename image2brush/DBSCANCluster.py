import numpy as np
import cv2

from sklearn.cluster import DBSCAN

from .ColorGenerator import rand_color_map
from .config import Settings

def dbscanClusteringShapeDetectionHSV(input_image, splits_clusters, h_weight=1, s_weight=1):
    OUTPUT_DIRECTORY = Settings().read_value("output_dir")
    input_shape = input_image.shape
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(input_image_hsv)
    texture_data = np.empty((0,4), int)
    coordinate_data = np.empty((0,2), int)

    for x, split_clusters in enumerate(splits_clusters):
        texture_datum = np.empty((0,4), int)
        coordinate_datum = np.empty((0,2), int)
        for y, split_cluster in enumerate(split_clusters):
            if split_cluster == 0:
                texture_dot = np.array([[x*2, y*2, h[x][y]*h_weight, s[x][y]*s_weight]])
                texture_datum = np.append(texture_datum, texture_dot, axis=0)
                coordinate = np.array([[x, y]])
                coordinate_datum = np.append(coordinate_datum, coordinate, axis=0)
        texture_data = np.concatenate((texture_data, texture_datum), axis=0)
        coordinate_data = np.concatenate((coordinate_data, coordinate_datum), axis=0)

    dbscan = DBSCAN(eps=40, min_samples=1).fit(texture_data)
    clusters = dbscan.labels_
    n = np.max(clusters)
    print(n)

    cluster_data = {}
    for coordinate, cluster in zip(coordinate_data, clusters):
        if str(cluster) not in cluster_data:
            cluster_data[str(cluster)] = []
        cluster_data[str(cluster)].append([int(coordinate[0]), int(coordinate[1])])

    RGB = rand_color_map(0, 255, n)
    
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 4), dtype=int)
    for coordinate, color_map in zip(coordinate_data, clusters):
        rgb = RGB[color_map-1]
        cluster_layer[coordinate[0]][coordinate[1]] = np.array([rgb[0], rgb[1], rgb[2], 255])
    cv2.imwrite(OUTPUT_DIRECTORY + '/mask.png', cluster_layer[:,:,:3])
    cv2.imwrite(OUTPUT_DIRECTORY + '/layer.png', cluster_layer)

    return cluster_data