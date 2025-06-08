from typing import Optional, Sequence, Union

from datasets import load_dataset
import torch
from torchvision import transforms

def stable_randn(
        shape: Union[int, Sequence[int]],
        seed: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    rn = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
    return rn


def next_seed(seed: int, adv: int = 0xF) -> int:
    """
    This is a naive helper function to generate a new seed from the given seed.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device
    ).tolist()[-1]


def split_seed(seed: int) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randint(0, torch.iinfo(torch.int64).max, (2,), generator=generator, device=generator.device).tolist()
    )

def get_mnist(num_proc: int=16):
    def transform_data(examples):
        # Flatten the images and convert to float32
        images = [transforms.ToTensor()(image).flatten() for image in examples['image']]
        labels = examples['label']
        return {'inputs': images, 'labels': labels}

    full_dataset = load_dataset('ylecun/mnist')
    full_dataset = full_dataset.map(
        transform_data,
        remove_columns=['image', 'label'],
        num_proc=num_proc,
        batched=True,
    )
    full_dataset.set_format(type='torch', columns=['inputs', 'labels'])

    train_valid = full_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_valid['train']
    validation_dataset = train_valid['test']
    test_dataset = full_dataset['test']

    return train_dataset, validation_dataset, test_dataset
