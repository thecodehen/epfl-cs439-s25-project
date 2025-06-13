for seed in 42 123 88; do
    for kappa in 10 50 100 200 500 1000; do
        python -m experiments.run_flora --rand_distribution normal --rank 16 --kappa $kappa --hidden_size 256 --seed $seed --lr 0.0005 --epochs 3000
    done
done