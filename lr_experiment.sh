for seed in 42 123 888; do
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.05 --epochs 3000
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.01 --epochs 3000
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.005 --epochs 3000
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.001 --epochs 3000
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.0005 --epochs 3000
    python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr 0.0001 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.05 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.01 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.005 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.001 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.0005 --epochs 3000
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.0001 --epochs 3000
done