import argparse
import json
import dataclasses
import enum
import numpy

from abc import abstractmethod

from .ImageProcessing import *
from .KMeansCluster import *
from .CropTextures import *

@dataclasses.dataclass
class Filtering:
    image_path: str

    @abstractmethod
    def apply_filtering(self):
        pass

class GaussianFilter(Filtering):
    def apply_filtering(self):
        blur_image = applyGaussianFilterFromImagePath(self.image_path, 5)
        return blur_image

class BilateralFilter(Filtering):
    def apply_filtering(self):
        blur_image = applyBilateralFilterFromImagePath(self.image_path)
        return blur_image

class FilteringType(enum.Enum):
    gaussian = 'GaussianFilter'
    bilateral = 'BilateralFilter'

    @classmethod
    def get_filtering_instance(cls, filtering_type, input_image):
        return eval(cls[filtering_type].value)(input_image)


@dataclasses.dataclass
class DivideArea:
    blur_image: numpy.ndarray

    @abstractmethod
    def apply_clustering(self):
        pass

@dataclasses.dataclass
class MorphologicalTransformDivideArea(DivideArea):
    def apply_clustering(self):
        canny_edge_image = applyCannyEdgeDetectionFromImage(self.blur_image)
        binalize_image = applyBinalizeFilterFromImage(canny_edge_image)
        return morphologicalTransformClustering(binalize_image)

@dataclasses.dataclass
class KMeansClusteringDivideArea(DivideArea):
    L_weight: int
    A_weight: int
    B_weight: int

    def apply_clustering(self):
        return kMeansClusteringLAB(self.blur_image, self.L_weight, self.A_weight, self.B_weight)

class DivideType(enum.Enum):
    morphological = 'MorphologicalTransformDivideArea'
    k_means = 'KMeansClusteringDivideArea'

    @classmethod
    def get_divide_instance(cls, first_step, blur_image):
        if len(first_step) == 1:
            return eval(cls[first_step['method']].value)(blur_image)
        else:
            L_weight = first_step['L_weight']
            A_weight = first_step['A_weight']
            B_weight = first_step['B_weight']
            return eval(cls[first_step['method']].value)(blur_image, L_weight, A_weight, B_weight)

@dataclasses.dataclass
class ClusterBrush:
    input_image: numpy.array
    cluster_data: numpy.array

    @abstractmethod
    def apply_clustering(self):
        pass

@dataclasses.dataclass
class KMeansClusteringBrushCoordinate(ClusterBrush):
    def apply_clustering(self):
        return kMeansClusteringCoordinate(self.input_image, self.cluster_data)

@dataclasses.dataclass
class KMeansClusteringBrushCoordinateHSV(ClusterBrush):
    H_weight: int
    S_weight: int

    def apply_clustering(self):
        return kMeansClusteringShapeDetection(self.input_image, self.cluster_data, self.H_weight, self.S_weight)

class ClusterBrushType(enum.Enum):
    coordinate = 'KMeansClusteringBrushCoordinate'
    coordinateHS = 'KMeansClusteringBrushCoordinateHSV'

    @classmethod
    def get_cluster_brush_instance(cls, second_step, input_image, cluster_data):
        if len(second_step) == 1:
            return eval(cls[second_step['method']].value)(input_image, cluster_data)
        else:
            H_weight = second_step['H_weight']
            S_weight = second_step['S_weight']
            return eval(cls[second_step['method']].value)(input_image, cluster_data, H_weight, S_weight)

# def applyCannyEdgeAndMorphologicalFilter(image_path):
#     canny_edge_image = applyCannyEdgeDetectionFromImagePath(image_path)
#     binalize_image = applyBinalizeFilterFromImagePath(canny_edge_image)
#     return applyMorphologicalTransform(binalize_image)

# def extractBrushesUsingCoordinateAndKMeansClusering(image_path):
#     binalize_image = applyCannyEdgeAndMorphologicalFilter(image_path)
#     cluster_data = kMeansClusteringCoordinate(input_image = binalize_image)
#     extractMaskBoundayAndBrushData(image_path, cluster_data)

# def extractBrushesUsingLABAndKMeansClusering(image_path):
#     blur_image = applyGaussianFilterFromImagePath(image_path, 5)
#     kMeansClusteringLAB(blur_image)

# def extractBrushesUsingHSVAndKMeansClusering(image_path):
#     blur_image = applyGaussianFilterFromImagePath(image_path, 15)
#     cluster_layer = kMeansClusteringHSV(blur_image)
#     cluster_data = kMeansClusteringCoordinate(input_image = cluster_layer)
#     extractMaskBoundayAndBrushData(image_path, cluster_data)

# def extractBrushesUsingYUVAndKMeansClusering(image_path):
#     blur_image = applyGaussianFilterFromImagePath(image_path, 15)
#     kMeansClusteringYUV(blur_image)
    
# def extractBrushesUsingHLSAndKMeansClusering(image_path):
#     blur_image = applyGaussianFilterFromImagePath(image_path, 15)
#     kMeansClusteringHLS(blur_image)

# def extractBrushesUsingXYZAndKMeansClusering(image_path):
#     blur_image = applyGaussianFilterFromImagePath(image_path, 15)
#     kMeansClusteringXYZ(blur_image)

# def extractBrushesUsingBilateralLABAndKMeansClustering(image_path):
#     blur_image = applyBilateralFilterFromImagePath(image_path)
#     clusters = kMeansClusteringLAB(blur_image)
#     cluster_data = kMeansClusteringShapeDetection(blur_image, clusters)
#     extractMaskBoundayAndBrushData(image_path, cluster_data)

def loadJSON(file_path):
    f = open(file_path, "r")
    json_data = json.load(f)
    return json_data

def loadCommandLineVariable():
    parser = argparse.ArgumentParser(description='This is a script of getting brushes from image.')
    parser.add_argument('--setting', type=str, required=True)
    return parser.parse_args()

def main():
    args = loadCommandLineVariable()
    setting_data = loadJSON(args.setting)

    filteringType = FilteringType.get_filtering_instance(setting_data['filtering'], setting_data['input_image'])
    blur_image = filteringType.apply_filtering()

    firstStep = DivideType.get_divide_instance(setting_data['first_step'], blur_image)
    clusters = firstStep.apply_clustering()

    secondStep = ClusterBrushType.get_cluster_brush_instance(setting_data['second_step'], blur_image, clusters)
    brush_clusters = secondStep.apply_clustering()



if __name__ == '__main__':
    main()