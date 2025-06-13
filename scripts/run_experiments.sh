#!/bin/bash

python -m experiments.run_adam --hidden_size 256
python -m experiments.run_adam --hidden_size 256 --clip_norm

for seed in {0..4}; do
  python -m experiments.run_flora --rand_distribution normal     --rank 8 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution uniform    --rank 8 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution discrete   --rank 8 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution discrete_3 --rank 8 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution normal     --rank 8 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution uniform    --rank 8 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution discrete   --rank 8 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution discrete_3 --rank 8 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution normal     --rank 256 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution uniform    --rank 256 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution discrete   --rank 256 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution discrete_3 --rank 256 --hidden_size 256 --seed $seed
  python -m experiments.run_flora --rand_distribution normal     --rank 256 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution uniform    --rank 256 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution discrete   --rank 256 --hidden_size 256 --seed $seed --clip_norm
  python -m experiments.run_flora --rand_distribution discrete_3 --rank 256 --hidden_size 256 --seed $seed --clip_norm
done
