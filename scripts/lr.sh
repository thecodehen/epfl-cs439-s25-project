for seed in 42 123 888; do
    for lr in 0.05 0.01 0.005 0.001 0.0005 0.0001 do
        python -m experiments.run_flora --rand_distribution normal --rank 16 --hidden_size 256 --seed $seed --lr $lr --epochs 3000
        python -m experiments.run_adam --hidden_size 256 --seed $seed --lr $lr --epochs 3000
    done
done