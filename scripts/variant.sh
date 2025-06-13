for seed in 42 123 888; do
    python -m experiments.run_floraaf --seed $seed --epochs 3000 --lr 0.0005 --rank 16 --hidden_size 256
    python -m experiments.run_floradr --seed $seed --epochs 3000 --lr 0.0005 --rankFrom 16 --rankTo 2 --hidden_size 256 
done