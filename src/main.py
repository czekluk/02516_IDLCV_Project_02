import sys
import torch
torch.cuda.empty_cache()
from experiments.test_experiment import test_experiment
def main():
    test_experiment()

if __name__ == "__main__":
    main()