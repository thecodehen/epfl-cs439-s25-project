import argparse
from datetime import datetime
import pathlib

import torch
import torch.nn as nn

import numpy as np

from src.adam import CustomAdam
from src.models import SimpleNeuralNetwork
from src.training import (
    train_model,
    evaluate_model,
)
from src.utils import get_mnist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000)
    parser.add_argument('--lr', type=float, default=1.e-5)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--clip_norm', action='store_true')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    input_size = 28 * 28
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SimpleNeuralNetwork(input_size, args.hidden_size, args.num_layers, num_classes, args.seed).to(device)

    # Training step
    loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = CustomAdam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    train_dataset, validation_dataset, test_dataset = get_mnist(seed=args.seed)

    train_losses, validation_losses, validation_accs, _ = train_model(model, args.epochs, train_dataset, validation_dataset, loss_fn, optimizer, args.clip_norm, device=device)
    _, _, acc = evaluate_model(model, test_dataset, device=device)

    print('Adam ', 'lr: ', args.lr, ' seed: ', args.seed, 'Accuracy: ', acc * 100, '%')

    # save the models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = pathlib.Path('results') / 'adam' / f"lr_{args.lr}-hidden_size_{args.hidden_size}-clip_norm_{args.clip_norm}-seed_{args.seed}-{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"model.pth")
    np.save(save_path / f"train_losses.npy", train_losses)
    np.save(save_path / f"validation_losses.npy", validation_losses)
    np.save(save_path / f"validation_accs.npy", validation_accs)

if __name__ == '__main__':
    main()
