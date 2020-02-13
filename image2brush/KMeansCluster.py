import numpy as np
import cv2

from sklearn.cluster import KMeans

from .ColorGenerator import rand_color_map
from .config import Settings

def kMeansClusteringShapeDetection(input_image, splits_clusters, h_weight=1, s_weight=1):
    n = 50
    input_shape = input_image.shape
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(input_image_hsv)
    nh = (h / 180).astype(float)
    ns = (h / 255).astype(float)
    nv = (v / 255).astype(float)
    texture_data = np.empty((0,4), int)
    coordinate_data = np.empty((0,2), int)

    for x, split_clusters in enumerate(splits_clusters):
        texture_datum = np.empty((0,4), int)
        coordinate_datum = np.empty((0,2), int)
        for y, split_cluster in enumerate(split_clusters):
            if split_cluster == 0:
                texture_dot = np.array([[x, y, h[x][y]*h_weight, s[x][y]*s_weight]])
                texture_datum = np.append(texture_datum, texture_dot, axis=0)
                coordinate = np.array([[x, y]])
                coordinate_datum = np.append(coordinate_datum, coordinate, axis=0)
        texture_data = np.concatenate((texture_data, texture_datum), axis=0)
        coordinate_data = np.concatenate((coordinate_data, coordinate_datum), axis=0)

    kmeans = KMeans(n_clusters= n)
    kmeans.fit(texture_data)
    kmeans_clusters = kmeans.predict(texture_data)

    cluster_data = {}
    for coordinate, cluster in zip(coordinate_data, kmeans_clusters):
        if str(cluster) not in cluster_data:
            cluster_data[str(cluster)] = []
        cluster_data[str(cluster)].append([int(coordinate[0]), int(coordinate[1])])

    RGB = rand_color_map(0, 255, n)
    
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 4), dtype=int)
    for coordinate, color_map in zip(coordinate_data, kmeans_clusters):
        rgb = RGB[color_map - 1]
        cluster_layer[coordinate[0]][coordinate[1]] = np.array([rgb[0], rgb[1], rgb[2], 255])
    cv2.imwrite(Settings().read_value("output_dir") + '/mask.png', cluster_layer[:,:,:3])
    cv2.imwrite(Settings().read_value("output_dir") + '/layer.png', cluster_layer)

    return cluster_data

def kMeansClusteringCoordinate(input_image, splits_clusters):
    n = 50
    input_shape = input_image.shape
    texture_data = np.empty((0,2), int)
    coordinate_data = np.empty((0,2), int)

    for x, split_clusters in enumerate(splits_clusters):
        texture_datum = np.empty((0,2), int)
        coordinate_datum = np.empty((0,2), int)
        for y, split_cluster in enumerate(split_clusters):
            if split_cluster == 0:
                texture_dot = np.array([[x, y]])
                texture_datum = np.append(texture_datum, texture_dot, axis=0)
                coordinate = np.array([[x, y]])
                coordinate_datum = np.append(coordinate_datum, coordinate, axis=0)
        texture_data = np.concatenate((texture_data, texture_datum), axis=0)
        coordinate_data = np.concatenate((coordinate_data, coordinate_datum), axis=0)

    kmeans = KMeans(n_clusters= n)
    kmeans.fit(texture_data)
    kmeans_clusters = kmeans.predict(texture_data)

    cluster_data = {}
    for coordinate, cluster in zip(coordinate_data, kmeans_clusters):
        if str(cluster) not in cluster_data:
            cluster_data[str(cluster)] = []
        cluster_data[str(cluster)].append([int(coordinate[0]), int(coordinate[1])])

    RGB = rand_color_map(0, 255, n)
    
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 4), dtype=int)
    for coordinate, color_map in zip(coordinate_data, kmeans_clusters):
        rgb = RGB[color_map - 1]
        cluster_layer[coordinate[0]][coordinate[1]] = np.array([rgb[0], rgb[1], rgb[2], 255])
    cv2.imwrite(Settings().read_value("output_dir") + '/mask.png', cluster_layer[:,:,:3])
    cv2.imwrite(Settings().read_value("output_dir") + '/layer.png', cluster_layer)

    return cluster_data

def kMeansClusteringLAB(input_image, l_weight=1, a_weight=1, b_weight=1):
    input_shape = input_image.shape
    lab_data = np.empty((0,3), int)
    input_image_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(input_image_lab)
    
    for index_x in range(input_image_lab.shape[0]):
        lab_datum = np.empty((0,3), int)
        for index_y in range(input_image_lab.shape[1]):
            lab = np.array([[l[index_x][index_y]*l_weight, a[index_x][index_y]*a_weight, b[index_x][index_y]*b_weight]])
            lab_datum = np.append(lab_datum, lab, axis=0)
        lab_data = np.concatenate((lab_data, lab_datum), axis=0)

    kmeans = KMeans(n_clusters=2, init='k-means++', n_jobs=2).fit(lab_data)
    kmeans.fit(lab_data)
    kmeans_clusters = kmeans.predict(lab_data)

    clusters = np.split(kmeans_clusters, int(len(kmeans_clusters)/input_shape[1]))
    return clusters

def kMeansClusteringHSV(input_image):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            if len(l[idx: idx + n]) == n:
                yield l[idx: idx + n]
            else:
                pass

    input_shape = input_image.shape
    hsv_data = np.empty((0,3), int)
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(input_image_hsv)

    print('generate array')
    for index_x in range(input_image_hsv.shape[0]):
        hsv_datum = np.empty((0,3), int)
        for index_y in range(input_image_hsv.shape[1]):
            hsv = np.array([[h[index_x][index_y], s[index_x][index_y], v[index_x][index_y]]])
            hsv_datum = np.append(hsv_datum, hsv, axis=0)
        hsv_data = np.concatenate((hsv_data, hsv_datum), axis=0)

    kmeans = KMeans(n_clusters=2, n_jobs=2).fit(hsv_data)
    kmeans.fit(hsv_data)
    kmeans_clusters = kmeans.predict(hsv_data)

    RGB = [0,255]

    splits_clusters = split_list(kmeans_clusters, input_shape[1])
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 1), dtype=int)
    for x, split_clusters in enumerate(splits_clusters):
        for y, split_cluster in enumerate(split_clusters):
            cluster_layer[x][y] = np.array(RGB[split_cluster])

    cv2.imwrite('result/cluster.png', cluster_layer)
    return cluster_layer

def kMeansClusteringYUV(input_image):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            if len(l[idx: idx + n]) == n:
                yield l[idx: idx + n]
            else:
                pass

    input_shape = input_image.shape
    yuv_data = np.empty((0,3), int)
    input_image_yuv = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(input_image_yuv)

    print('generate array')
    for index_x in range(input_image_yuv.shape[0]):
        yuv_datum = np.empty((0,3), int)
        for index_y in range(input_image_yuv.shape[1]):
            yuv = np.array([[y[index_x][index_y], u[index_x][index_y], v[index_x][index_y]]])
            yuv_datum = np.append(yuv_datum, yuv, axis=0)
        yuv_data = np.concatenate((yuv_data, yuv_datum), axis=0)

    kmeans = KMeans(n_clusters=2, n_jobs=2).fit(yuv_data)
    kmeans.fit(yuv_data)
    kmeans_clusters = kmeans.predict(yuv_data)

    RGB = rand_color_map(0, 255, 2)

    splits_clusters = split_list(kmeans_clusters, input_shape[1])
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 3), dtype=int)
    for x, split_clusters in enumerate(splits_clusters):
        for y, split_cluster in enumerate(split_clusters):
            rgb = RGB[split_cluster]
            cluster_layer[x][y] = np.array([rgb[0], rgb[1], rgb[2]])

    cv2.imwrite('result/cluster.png', cluster_layer)

def kMeansClusteringHLS(input_image):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            if len(l[idx: idx + n]) == n:
                yield l[idx: idx + n]
            else:
                pass

    input_shape = input_image.shape
    hls_data = np.empty((0,3), int)
    input_image_hls = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(input_image_hls)

    print('generate array')
    for index_x in range(input_image_hls.shape[0]):
        hls_datum = np.empty((0,3), int)
        for index_y in range(input_image_hls.shape[1]):
            hls = np.array([[h[index_x][index_y], l[index_x][index_y], s[index_x][index_y]]])
            hls_datum = np.append(hls_datum, hls, axis=0)
        hls_data = np.concatenate((hls_data, hls_datum), axis=0)

    kmeans = KMeans(n_clusters=2, n_jobs=2).fit(hls_data)
    kmeans.fit(hls_data)
    kmeans_clusters = kmeans.predict(hls_data)

    RGB = rand_color_map(0, 255, 2)

    splits_clusters = split_list(kmeans_clusters, input_shape[1])
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 3), dtype=int)
    for x, split_clusters in enumerate(splits_clusters):
        for y, split_cluster in enumerate(split_clusters):
            rgb = RGB[split_cluster]
            cluster_layer[x][y] = np.array([rgb[0], rgb[1], rgb[2]])

    cv2.imwrite('result/cluster.png', cluster_layer)

def kMeansClusteringXYZ(input_image):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            if len(l[idx: idx + n]) == n:
                yield l[idx: idx + n]
            else:
                pass

    input_shape = input_image.shape
    xyz_data = np.empty((0,3), int)
    input_image_xyz = cv2.cvtColor(input_image, cv2.COLOR_BGR2XYZ)
    x, y, z = cv2.split(input_image_xyz)

    print('generate array')
    for index_x in range(input_image_xyz.shape[0]):
        xyz_datum = np.empty((0,3), int)
        for index_y in range(input_image_xyz.shape[1]):
            xyz = np.array([[x[index_x][index_y], y[index_x][index_y], z[index_x][index_y]]])
            xyz_datum = np.append(xyz_datum, xyz, axis=0)
        xyz_data = np.concatenate((xyz_data, xyz_datum), axis=0)

    kmeans = KMeans(n_clusters=2, n_jobs=2).fit(xyz_data)
    kmeans.fit(xyz_data)
    kmeans_clusters = kmeans.predict(xyz_data)

    RGB = rand_color_map(0, 255, 2)

    splits_clusters = split_list(kmeans_clusters, input_shape[1])
    cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 3), dtype=int)
    for x, split_clusters in enumerate(splits_clusters):
        for y, split_cluster in enumerate(split_clusters):
            rgb = RGB[split_cluster]
            cluster_layer[x][y] = np.array([rgb[0], rgb[1], rgb[2]])

    cv2.imwrite('result/cluster.png', cluster_layer)