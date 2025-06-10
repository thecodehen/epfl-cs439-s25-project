import argparse
from datetime import datetime
import pathlib

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from flora import FloraAdam
from models import SimpleNeuralNetwork
from utils import get_mnist

def train_one_epoch(model, optimizer, loss_fn, batch_inputs, batch_labels, epoch_index, device=None):
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(batch_inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, batch_labels)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss.item()

def train_model(model, epochs, train_dataset, validation_dataset, loss_fn, optimizer, device=None):
    train_inputs = train_dataset['inputs'].to(device)
    train_labels = train_dataset['labels'].to(device)
    validation_inputs = validation_dataset['inputs'].to(device)
    validation_labels = validation_dataset['labels'].to(device)


    train_losses = []
    validation_losses = []
    for epoch in tqdm(range(epochs)):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_inputs, train_labels, epoch, device=device)
        train_losses.append(train_loss)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            # Make predictions for this batch
            voutputs = model(validation_inputs)
            validation_loss = loss_fn(voutputs, validation_labels)
            validation_losses.append(validation_loss.item())

        # Track best performance, and save the model's state
        # if validation_loss < best_vloss:
        #     best_vloss = validation_loss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch)
        #     torch.save(model.state_dict(), model_path)

        epoch += 1

    return train_losses, validation_losses

def evaluate_model(model, test_dataset, device=None):
    inputs = test_dataset['inputs'].to(device)

    with torch.no_grad():
        outputs = model(inputs).argmax(dim=1)

    all_labels = test_dataset['labels']
    all_preds = outputs.cpu()

    assert len(all_labels) == len(all_preds)

    acc = accuracy_score(all_labels, all_preds)

    return all_preds, all_labels, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000)
    parser.add_argument('--lr', type=float, default=1.e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--rand_distribution', type=str, default='normal')
    parser.add_argument('--output_path', type=str, default='results')

    args = parser.parse_args()

    input_size = 28 * 28
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SimpleNeuralNetwork(input_size, args.hidden_size, args.num_layers, num_classes).to(device)

    # Training step
    loss_fn= nn.CrossEntropyLoss().to(device)

    optimizer = FloraAdam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        rank=args.rank,
        kappa=100,
        seed=args.seed,
        rand_distribution=args.rand_distribution,
    )

    train_dataset, validation_dataset, test_dataset = get_mnist()

    train_losses, validation_losses = train_model(model, args.epochs, train_dataset, validation_dataset, loss_fn, optimizer, device=device)
    _, _, acc = evaluate_model(model, test_dataset, device=device)

    print('Accuracy: ', acc * 100, '%')

    # save the models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = pathlib.Path('results') / f"{args.rand_distribution}-rank_{args.rank}-seed_{args.seed}-{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"model.pth")
    np.save(save_path / f"train_losses.npy", train_losses)
    np.save(save_path / f"validation_losses.npy", validation_losses)

if __name__ == '__main__':
    main()