import numpy as np
import cv2

from .config import Settings

def extractMaskBoundayAndBrushData(input_path, cluster_data):
    MASK_CROP_DATA_DIRECTORY = Settings().get_mask_crop_data_directory()
    BRUSH_CROP_DATA_DIRECTORY = Settings().get_brush_crop_data_directory()

    r, g, b = cv2.split(cv2.imread(input_path))
    image_shape = r.shape
    for index, cluster_index in enumerate(cluster_data):
        cluster_datum = np.array(cluster_data[cluster_index])
        xy_max = np.max(cluster_datum, axis=0)
        xy_min = np.min(cluster_datum, axis=0)
        mask_layer = np.zeros(shape=(image_shape[0], image_shape[1], 3), dtype=int)
        brush_layer = np.zeros(shape=(image_shape[0], image_shape[1], 4), dtype=int)
        for coordinate in cluster_data[cluster_index]:
            x, y =  coordinate[0], coordinate[1]
            mask_layer[x][y] = np.array([255, 255, 255])
            brush_layer[x][y] = np.array([r[x][y], g[x][y], b[x][y], 255])
        try:
            cv2.imwrite(MASK_CROP_DATA_DIRECTORY + '/mask_crop' + str(index) + '.png', mask_layer[int(xy_min[0]):int(xy_max[0]), int(xy_min[1]):int(xy_max[1])])
            cv2.imwrite(BRUSH_CROP_DATA_DIRECTORY + '/brush_crop' + str(index) + '.png', brush_layer[int(xy_min[0]):int(xy_max[0]), int(xy_min[1]):int(xy_max[1])])
        except:
            pass