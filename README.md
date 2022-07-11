# preProcessing
This repository contains two main scripts for the preProcessing of the Whole Slide Images (WSIs) as an initial step for histopathological deep learning.

0. Install openslide on Fedora via: ```dnf install openslide-tools```.
1. Set up python environment with ```pip install -r requirements.txt```.
2. extractTiles-ws : This script is used to tessellate the WSIs. The main required inputs for this function:

Input Variable name | Description
--- | --- 
-s | Path to the WSI folder | 
-o | Path to the output folder where tiles are saved
--skipws | Skip tessellation of WSI if annotation is missing. Default value is False.
-px | Size of image patches to analyze, in pixels
-um | Size of image patches to analyze, in microns.
--num_threads | Number of threads to use when tessellating.
--augment | Augment extracted tiles with flipping/rotating.
--ov | The Size of overlappig for extracted tiles. It can be values between 0 and 1.

3. Normalize : This script is used to normalize the extracted tiles using Macenko method. The main required inputs for this function:

Input Variable name | Description
--- | --- 
-inputPath | Path to the BLOCKS folder, where the tiles are saved | 
-outputPath | Path to the output folder, to save the normalized tiles
--sampleImagePath | Path to one sample tile, which it's color distribution will used as a template for all the tiles.

In this script, we are using the Macenko normalization method from https://github.com/wanghao14/Stain_Normalization.git repository.
