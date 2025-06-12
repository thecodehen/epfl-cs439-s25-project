for seed in 42 123 888; do
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.05
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.01
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.005
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.001
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.0005
    python -m experiments.run_flora --rand_distribution normal --rank 8 --hidden_size 256 --seed $seed --lr 0.0001
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.05
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.01
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.005
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.001
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.0005
    python -m experiments.run_adam --hidden_size 256 --seed $seed --lr 0.0001
done