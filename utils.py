"""
Contains various utility functions for PyTorch.
Setting up DataLoaders for image classification data,
setting global seeds, device-agnostic code and
saving trained models.
"""
import os
import torch

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


NUM_WORKERS = int(os.cpu_count()/2)

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
               model_name: str):
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

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def delete_unnecessary_files():
    """
    Removes any extra sub-directories that
    Tensorboard creates by default.
    """
    import shutil
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    parent = "runs"

    for item in os.listdir(parent):
        item_path = os.path.join(parent, item)
        # Check if the item is a directory and if itsnname is not the target name
        if os.path.isdir(item_path) and item != timestamp:
            shutil.rmtree(item_path)
            # print(f" Unecessary folder deleted: {item_path}")

    # print(f"\nOperation complete.")
    print(f"Unecessary folders deleted.")