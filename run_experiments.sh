#!/bin/bash

for seed in {0..4}; do
  python -m experiments.compare_rand_projections --rand_distribution normal   --rank 16 --seed $seed
  python -m experiments.compare_rand_projections --rand_distribution uniform  --rank 16 --seed $seed
  python -m experiments.compare_rand_projections --rand_distribution discrete --rank 16 --seed $seed
done
