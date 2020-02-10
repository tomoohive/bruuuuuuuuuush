import argparse
import json
import dataclasses
import enum

from .ImageProcessing import *

@dataclasses.dataclass
class Filtering:
    image: str

    def apply_filtering(self):
        pass

class GaussianFilter(Filtering):
    def apply_filtering(self):
        blur_image = applyGaussianFilterFromImagePath(self.image, 5)
        return blur_image

class BilateralFilter(Filtering):
    def apply_filtering(self):
        blur_image = blur_image = applyBilateralFilterFromImagePath(self.image)

class FilteringType(enum.Enum):
    gaussian = 'GaussianFilter'
    bilateral = 'BilateralFilter'

    @classmethod
    def get_filtering_instance(cls, filtering_type, input_image):
        return eval(cls[filtering_type].value)(input_image)


@dataclasses.dataclass
class DivideArea:
    image: np.ndarray

    def apply_clustering(self):
        pass

class MorphologicalTransformDivideArea(DivideArea):
    def apply_clustering(self):
        canny_edge_image = applyCannyEdgeDetectionFromImage(image_path)
        binalize_image = applyBinalizeFilterFromImage(canny_edge_image)
        return applyMorphologicalTransform(binalize_image)

class KMeansClusteringDivideArea(DivideArea):
    def apply_clustering(self):
        kMeansClusteringLAB(blur_image)

# from .KMeansCluster import *
# from .CropTextures import *

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
    blur_image = FilteringType.get_filtering_instance(setting_data['filtering'], setting_data['input_image'])



if __name__ == '__main__':
    main()