import os
import shutil

from .config import Settings

def makeDirectories():
    os.makedirs(Settings().read_value("output_dir"))
    os.makedirs(Settings().get_mask_crop_data_directory())
    os.makedirs(Settings().get_brush_crop_data_directory())
    os.makedirs(Settings().get_feathering_directory())

def defineDirectories(output_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    makeDirectories()
