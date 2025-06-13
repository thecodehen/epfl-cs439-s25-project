# EPFL CS-439 Spring 2025 Final Project

Implementation and Analysis of [Flora](https://arxiv.org/pdf/2402.03293) integrated into Adam optimizer.

## Project Structure

```
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
├── plot.ipynb                     # Plotting and visualization notebook
├── experiments/                   # Experiment runner scripts
│   ├── run_adam.py               
│   ├── run_flora.py              
│   ├── run_floraaf.py            
│   └── run_floradr.py            
├── scripts/                      # Shell scripts for batch experiments
│   ├── lr.sh                     
│   ├── r.sh                      
│   ├── kappa.sh                  
│   ├── variant.sh                
│   ├── random_distribution.sh    
├── src/                          # Source code modules
│   ├── adam.py                   
│   ├── flora.py                  
│   ├── floraAdamFactoredMAS.py   
│   └── floraDecreasingRanks.py   
│   └── models.py   
│   └── training.py   
│   └── utils.py   
└── pdf/                          # Generated plots and analysis
    ├── optimizer_comparison_final.pdf
    ├── convergence_analysis.pdf
    ├── learning_rate.pdf
    ├── rank.pdf
    ├── kappa.pdf
    └── rand_projections.pdf
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Experiments

This project includes several types of experiments to analyze the Flora optimizer:

### 1. Convergence Speed & Stability

Compare Flora variants against standard Adam:

```bash
bash scripts/lr.sh        # Get data for standard Adam and Flora
bash scripts/variant.sh   # For flora variants
```

### 2. Hyperparameter Sensitivity Analysis

#### Learning Rate Sensitivity
```bash
bash scripts/lr.sh
```

#### Rank Sensitivity
```bash
bash scripts/r.sh
```

#### Kappa (Seed Refresh Frequency) Sensitivity
```bash
bash scripts/kappa.sh
```

### 3. Random Distribution Analysis

Test different random projection distributions (normal, uniform, discrete, discrete_3):

```bash
bash scripts/random_distribution.sh
```

This script evaluates how different random distribution types affect Flora's performance across various ranks (8, 256) with and without gradient clipping.


## Results and Visualization

After running experiments, use the [`plot.ipynb`](plot.ipynb) notebook to generate visualizations and analysis plots. Results are automatically saved in the `results/` directory organized by optimizer type and hyperparameters.