#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516

### ------------- specify gpu request----------------
#BSUB -gpu "num=1"

### ------------- specify job name ----------------
#BSUB -J project_2_drive_unet

### ------------- specify number of cores ----------------
#BSUB -n 8
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o /zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/hpc_logs/OUTPUT_FILE%J.out
#BSUB -e /zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/hpc_logs/OUTPUT_FILE%J.err

HOME_DIR=/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02
LOG_FILE="$HOME_DIR/hpc_logs/output_drive_unet.txt"

source /zhome/25/a/202562/.venv_deep_learning/bin/activate
python3 /zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/src/main.py --dataset drive --models unet > "$LOG_FILE" 2>&1