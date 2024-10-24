import sys
import torch
torch.cuda.empty_cache()
sys.stdout = open('/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/hpc_logs/output.txt', 'w')
print('test')
from experiments.test_experiment_drive import test_experiment_drive
def main():
    test_experiment_drive(epochs=100)

if __name__ == "__main__":
    main()