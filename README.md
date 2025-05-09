# 02516_IDLCV_Project_02

## Project Description

This project aims at implementing encoder-decoder CNNs and UNets for semantic image segmentation of medical images (skin lesion and retinal blood vessels). Additionally, differnt  performance metrics shall be implemented and an ablation study shall be conducted.

![poster slide](/docs/poster_slide.jpg)

## Data

Two datasets are in focus for this project. They are both stored in the `/dtu/datasets1/02516` directory on the DTU HPC.

**PH2:** This dataset contains the images for the skin lesion task. In total there are XX training images, XX validation images and XX test images. All have ground truth masks.

![skin lesion](/docs/IMD406.bmp)

**DRIVE:** This dataset contains the images for the retinal blood vessel task. In total there are XX training images, XX validation images and XX test images. All have ground truth masks.

![retinal blood vessel](/docs/38_training.png)

## Run training on the HPC

To start training using batch jobs first modify the `jobscript.sh` file you want to use. Please note that every contributor is recommended to create his own `jobscript_[NAME].sh` file according to his preferences.

Then execute:
```bash
bsub -app c02516_1g.10gb < jobscript_[NAME].sh
```

To monitor the progress execute:
```bash
bstat
```

To abort the run:
```bash
bkill <job-id>
```
