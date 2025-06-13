for seed in 42 123 88; do
    for rank in 2 4 8 16 32 64; do
        python -m experiments.run_flora --rand_distribution normal --rank $rank --hidden_size 256 --seed $seed --lr 0.0005 --epochs 3000
    done
done