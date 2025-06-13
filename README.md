# EPFL CS-439 Spring 2025 Final Project

Implementation and Analysis of Flora (https://arxiv.org/pdf/2402.03293) integrated into Adam.

## Install the packages from requirements.txt

```
pip install -r requirements.txt
```

## Generate the data using the bash scripts:

### To generate the data of Convergence Speed & Stability experiment, run:

```
bash scripts/lr.sh
bash scripts/variants.sh
```
---

### To generate the data of Hyperparameter Sensitivity experiment, run:

#### For Mean Test Accuracy vs Learning Rate, run: 
```
bash scripts/lr.sh
```

#### For Mean Test Accuracy vs Rank, run: 
```
bash scripts/r.sh
```

#### For Mean Test Accuracy vs Kappa, run:
```
bash scripts/kappa.sh
```

---
## For Plotting the Results:

Please run the plot.ipynb notebook
