#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516

### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ----------------
#BSUB -J project_2

### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o /zhome/c4/e/203768/02516_IDLCV_Project_02/hpc_logs/OUTPUT_FILE%J.out
#BSUB -e /zhome/c4/e/203768/02516_IDLCV_Project_02/hpc_logs/OUTPUT_FILE%J.err

HOME_DIR=/zhome/f8/d/203610/git/02516_IDLCV_Project_02

source $HOME_DIR/dlcv/bin/activate
python $HOME_DIR/git/02516_IDLCV_Project_02/src/main.py
