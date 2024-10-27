# 02516_IDLCV_Project_02

## Project Description

This project aims at implementing encoder-decoder CNNs and UNets for semantic image segmentation of medical images (skin lesion and retinal blood vessels). Additionally, differnt  performance metrics shall be implemented and an ablation study shall be conducted.

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

## ToDo V2
- (\ ( ͠° ͟ل͜ ͡°) /) T R A I N (\ ( ͠° ͟ل͜ ͡°) /) - Nandor & Zeljko

- DRIVE Dataset: how to deal with images? - Filip
  - Should be same input size for both datasets (512x512) 
  - Data Augmentation: random rotate, random flip

- Save dataset & transforms in results - Lukas
- Make the label binary in `make_dataset.py` - Filip
- Finish visualizer `load_model` and `plot_prediction` - Alex
- Ablation study (models with different loss functions) - Lukas

Next meeting Thursday 14:00 (24.10.2025)

## ToDo V3
- Weak labels (clicks with different sampling strategies) - Filip
- Point level loss function - Alex
- Ablation study (number of clicks, sampling strategy for clicks) - Lukas
- Random Crop transformation on train img / sequentially crop on test - Zeljko & Filip

Next meeting Sunday 11:00 (27.10.25)

## ToDo V4
- plotting function for weak labels - Alex
- put results to poster - Lukas, Filip & Zeljko
- Ablation study (number of clicks, sampling strategy for clicks) - Lukas
- RandomResizeCrop for DRIVE dataset - Zeljko

## Submission Deadline

Submission on Tuesday 29th October at 18:00

## Poster

The current version of the poster may be found [here](https://dtudk.sharepoint.com/:p:/r/sites/IntroDLCV2024/Delte%20dokumenter/General/Poster_Project_02.pptx?d=w755d00ab60ef469797666547bc7aeb02&csf=1&web=1&e=tGUnzq).

