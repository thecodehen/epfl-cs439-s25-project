from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, optimizer, loss_fn, batch_inputs, batch_labels, clip_norm: bool):
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(batch_inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, batch_labels)
    loss.backward()

    if clip_norm:
        total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    else:
        total_grad_norm = nn.utils.get_total_norm(model.parameters())

    # Adjust learning weights
    optimizer.step()

    return loss.item(), total_grad_norm.item()

def train_model(model, epochs, train_dataset, validation_dataset, loss_fn, optimizer, clip_norm, device=None):
    train_inputs = train_dataset['inputs'].to(device)
    train_labels = train_dataset['labels'].to(device)
    validation_inputs = validation_dataset['inputs'].to(device)
    validation_labels = validation_dataset['labels'].to(device)


    train_losses = []
    validation_losses = []
    validation_accs = []
    total_grad_norms = []
    for epoch in tqdm(range(epochs)):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, total_grad_norm = train_one_epoch(model, optimizer, loss_fn, train_inputs, train_labels, clip_norm)
        train_losses.append(train_loss)
        total_grad_norms.append(total_grad_norm)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            # Make predictions for this batch
            voutputs = model(validation_inputs)
            all_preds = voutputs.argmax(dim=1)
            validation_acc = accuracy_score(validation_labels.cpu(), all_preds.cpu())
            validation_accs.append(validation_acc)
            validation_loss = loss_fn(voutputs, validation_labels)
            validation_losses.append(validation_loss.item())

        epoch += 1

    return train_losses, validation_losses, validation_accs, total_grad_norms

def evaluate_model(model, test_dataset, device=None):
    inputs = test_dataset['inputs'].to(device)

    with torch.no_grad():
        outputs = model(inputs).argmax(dim=1)

    all_labels = test_dataset['labels']
    all_preds = outputs.cpu()

    assert len(all_labels) == len(all_preds)

    acc = accuracy_score(all_labels, all_preds)

    return all_preds, all_labels, acc
