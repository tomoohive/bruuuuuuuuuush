import os
import shutil

OUTPUT_DIRECTORY = './output'
MASK_CROP_DATA_DIRECTORY = OUTPUT_DIRECTORY + '/mask_crop_data'
BRUSH_CROP_DATA_DIRECTORY = OUTPUT_DIRECTORY + '/brush_crop_data'
FEATHERING_DATA_DIRECTORY = OUTPUT_DIRECTORY + '/feathering'

def makeDirectories():
    os.makedirs(OUTPUT_DIRECTORY)
    os.makedirs(MASK_CROP_DATA_DIRECTORY)
    os.makedirs(BRUSH_CROP_DATA_DIRECTORY)
    os.makedirs(FEATHERING_DATA_DIRECTORY)

def defineDirectories(output_path):
    OUTPUT_DIRECTORY = output_path
    
    if os.path.isdir(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)
    makeDirectories()
