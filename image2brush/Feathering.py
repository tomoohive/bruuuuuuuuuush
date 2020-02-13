import os

from PIL import Image, ImageDraw, ImageFilter

from .config import Settings

def applyFeatheringProcessing(kernel_size):
    BRUSH_CROP_DATA_DIRECTORY = Settings().get_brush_crop_data_directory()
    MASK_CROP_DATA_DIRECTORY = Settings().get_mask_crop_data_directory()
    FEATHERING_DATA_DIRECTORY = Settings().get_feathering_directory()

    brush_list = os.listdir(BRUSH_CROP_DATA_DIRECTORY)
    mask_list = os.listdir(MASK_CROP_DATA_DIRECTORY)
    brush_list.sort()
    mask_list.sort()

    for index, (brush, mask) in enumerate(zip(brush_list, mask_list)):
        im_rgba = Image.open(BRUSH_CROP_DATA_DIRECTORY + '/' + brush)
        im_back = Image.new('RGBA', im_rgba.size, (0, 0, 0, 0))
        im_a = Image.open(MASK_CROP_DATA_DIRECTORY + '/' + mask).convert('L')
        im_a_blur = im_a.filter(ImageFilter.GaussianBlur(kernel_size))

        im = Image.composite(im_rgba, im_back, im_a_blur)
        im.save(FEATHERING_DATA_DIRECTORY + '/result'+ str(index) +'.png')