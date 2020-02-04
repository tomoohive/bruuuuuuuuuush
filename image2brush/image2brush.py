import argparse
import cv2

from .ImageProcessing import *
from .KMeansCluster import *
from .CropTextures import *

def applyCannyEdgeAndMorphologicalFilter(image_path):
    canny_edge_image = applyCannyEdgeDetectionFromImagePath(image_path)
    binalize_image = applyBinalizeFilterFromImagePath(canny_edge_image)
    return applyMorphologicalTransform(binalize_image)

def extractBrushesUsingCoordinateAndKMeansClusering(image_path):
    binalize_image = applyCannyEdgeAndMorphologicalFilter(image_path)
    cluster_data = kMeansClusteringCoordinate(input_image = binalize_image)
    extractMaskBoundayAndBrushData(image_path, cluster_data)

def extractBrushesUsingLABAndKMeansClusering(image_path):
    blur_image = applyGaussianFilterFromImagePath(image_path, 5)
    kMeansClusteringLAB(blur_image)

def extractBrushesUsingHSVAndKMeansClusering(image_path):
    blur_image = applyGaussianFilterFromImagePath(image_path, 15)
    cluster_layer = kMeansClusteringHSV(blur_image)
    cluster_data = kMeansClusteringCoordinate(input_image = cluster_layer)
    extractMaskBoundayAndBrushData(image_path, cluster_data)

def extractBrushesUsingYUVAndKMeansClusering(image_path):
    blur_image = applyGaussianFilterFromImagePath(image_path, 15)
    kMeansClusteringYUV(blur_image)
    
def extractBrushesUsingHLSAndKMeansClusering(image_path):
    blur_image = applyGaussianFilterFromImagePath(image_path, 15)
    kMeansClusteringHLS(blur_image)

def extractBrushesUsingXYZAndKMeansClusering(image_path):
    blur_image = applyGaussianFilterFromImagePath(image_path, 15)
    kMeansClusteringXYZ(blur_image)

def extractBrushesUsingBilateralLABAndKMeansClustering(image_path):
    blur_image = applyBilateralFilterFromImagePath(image_path)
    clusters = kMeansClusteringLAB(blur_image)
    cluster_data = kMeansClusteringShapeDetection(blur_image, clusters)
    extractMaskBoundayAndBrushData(image_path, cluster_data)

def main():
    parser = argparse.ArgumentParser(description = 'get a brush from image')
    parser.add_argument('--input', type=str, default='./input.png')
    parser.add_argument('--output', type=str, default='./canny_sigma.png')
    args = parser.parse_args()

    extractBrushesUsingBilateralLABAndKMeansClustering(args.input)
    # extractBrushesUsingCoordinateAndKMeansClusering(args.input)


if __name__ == '__main__':
    main()