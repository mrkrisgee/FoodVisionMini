"""
Contains various utility functions for PyTorch.
Setting up DataLoaders for image classification data,
setting global seeds, model training and saving.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                        test_dir: str,
                        transform: transforms.Compose,
                        batch_size: int,
                        num_workers: int = NUM_WORKERS):
    """ 
    Creates training and testing Dataloaders.

    Takes in a training and testing directory path turns
    them into PyTorch Datasets and then into PyTorch Dataloaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names =
            create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform = transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader= DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True        
        )

    test_dataloader= DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True        
        )

    return train_dataloader, test_dataloader, class_names

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args: seed(int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)