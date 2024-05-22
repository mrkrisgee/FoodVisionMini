"""
Contains various utility functions for PyTorch.
Setting up DataLoaders for image classification data,
setting global seeds, device-agnostic code and
saving trained models.
"""
import os
import torch

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

def set_device(device: str="cuda"):
    """
    Sets up device agnostic code.
    
    Ensures data is calculated on the same device.

    The code will use the GPU when there is one available,
    otherwise it will fallback on the CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device being used: {device}")
    return device

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_nme: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)