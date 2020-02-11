# image2brush: Generating damaged brushes from input images

### Requirements
- python3.8 (because using dataclass)
- openCV (pip: opencv-python)
- numpy
- Pillow
- scikit-learn

### How to Use
1. Download scripts and move this directory
```
git clone https://github.com/tomoohive/bruuuuuuuuuush.git
cd bruuuuuuuuuush
```

2. Set up parameters to change JSON file "image2brush.config.json"
```json
{
    "input_image": "./input.png",
    "filtering": "bilateral",
    "first_step": {
        "method": "morphological"
    },
    "_first_step": {
        "method": "k_means",
        "L_weight": 1,
        "A_weight": 5,
        "B_weight": 5
    },
    "second_step": {
        "method": "coordinate"
    },
    "_second_step": {
        "method": "coordinateHS",
        "H_weight": 3,
        "S_weight": 3
    },
    "output_dir": "./output",
    "feathering": 3
}
```
"input_image": You need to bring input image and put on suitable place.
"filtering": You can choose "gaussian" or "biateral" before applying k-means clustering.
"first_step": You can choose "morphological" or "k_means". If you select "k_means", you have to decide weight for LAB color space parameters(type: int).
"second_step": You can choose "coordinate" or "coordinateHS". "coordinate" produces using coordinate data for k-means clustering. Also, "coordinateHS" produces using coordinate data and H and S parameters from HSV color space.
"output_dir": You can set up the output directory on arbitary place. However, if you select an exist directory, that directory will be deleted automatically.
"feathering": Finally, you can set up strength of feathering for cropped brushes.

3. Excute python file with config JSON file
```
python3 -m image2brush --setting image2brush.config.json
```
By using --setting option, you can select a voluntary setting JSON file.

4. Finish

### Results
"mask_crop_data": results of mask data
"brush_crop_data": results of brush data
"feathering": resuls of applying feathering processing for cropped brushes