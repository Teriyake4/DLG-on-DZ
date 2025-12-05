import os
import numpy as np

experimentListPath = os.path.join(".", "training/", "experiments.txt")
print(os.path.abspath(experimentListPath))

os.makedirs(os.path.dirname(experimentListPath), exist_ok=True)

experiments = ""

for dataset in ["mnist", "cifar10"]:
    for lr in [0.1]:
        for batch_size in [128, 256, 512]:
            for alpha in np.arange(0.5, 0.9, 0.1).tolist() + [0.9, 0.95, 0.99, 0.999, 1]:
                for mu in [9.23e-10, 6.92e-10, 3.85e-10, 1.54e-10, 1e-5, 1e-7, 1e-9, 1e-15, 1e-17, 1e-30]:
                    for p in [0]:
                        experiments += f"python experiments/sparse_gradient_training.py --dataset {dataset} --lr {lr} --batch_size {batch_size} --alpha {alpha} --mu {mu} --p {p}\n"

with open(experimentListPath, "w") as f:
    f.write(experiments)