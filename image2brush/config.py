import os
import json
import argparse

class Settings:

    args = None

    def __init__(self):
        
        # Only load args once
        if self.args is None:
            parser = argparse.ArgumentParser(description='This is a script of getting brushes from image.')
            parser.add_argument('--setting', type=str, required=True)
            Settings.args = parser.parse_args()
        
        f = open(self.args.setting, "r")
        self._json = json.loads(f.read())
    
    def read_value(self, key):
        return self._json[key]

    def get_feathering_directory(self):
        return self.read_value("output_dir") + "/feathering"

    def get_mask_crop_data_directory(self):
        return self.read_value("output_dir") + "/mask_crop_data"

    def get_brush_crop_data_directory(self):
        return self.read_value("output_dir") + "/brush_crop_data"