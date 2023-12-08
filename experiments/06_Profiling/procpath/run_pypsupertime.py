import argparse
from pypsupertime import Psupertime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs=1, required=True)
    args = parser.parse_args()

    p=Psupertime(n_batches=10, 
                 n_jobs=1,
                 regularization_params={"n_folds": 5, "n_params": 40},
                 preprocessing_params={"log": False, "scale": False, "normalize": False},
                 estimator_params={"early_stopping_batches":True, "penalty": "l1"})
    p.run(args.file[0], "Ordinal_Time_Labels")

