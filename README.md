# README.md

For the full report, see [report.md](report.md)

## Scripting/Running Code

The code is spread across 5 Python files. To run the code, first check if Python is installed with `python --version`. Untested if older Python versions would work (3.10+ should reasonably work) but the environment was scripted in Python 3.13.11. You can download Python [here](https://www.python.org/downloads/).

Ensure the [Walmart dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) is downloaded and within the same directory/folder as this *report.md*.

Next check if `conda` is installed using `conda --version`. If not, install it through [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). Run the following once `conda` is setup:

```
conda env create -f environment.yml -n walmart_data
conda activate walmart_data
```

Code chunks are commented out in *main.py* with labels to run particular tasks. Uncomment the desired code tasks and run the file in the directory/folder of this *report.md* with 

```
python main.py
```

Currently, all functions are made to save to the directory/folder specified by `OUTPUT_FOLDER` and `STORE_FOLDER`. You are welcome to change any code, especially the configuration variables (in all capitalizated lettering) at the very top of *main.py*. The names of configuration variables explain their purpose. Note for replication purposes, everything reported in here is configured to `RNG_SEED = 314` unless otherwise specified. As a warning, there may be errors in each file depending on your IDE/environment but those are type checking errors and do not affect runtime so feel free to ignore those. What will be published to the repository <u>will</u> work as intended.