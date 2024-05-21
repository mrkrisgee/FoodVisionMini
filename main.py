import torch
import torchvision
import pathlib

from torch import nn
from torchvision import transforms

from subset_gen import get_subset

# Get torchininfo. Install if it doesn't exist
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    from torchinfo import summary

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")

## Get data
# Download and transform data to be inline with ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225])

# Create a transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Setup data directory
data_dir = pathlib.Path("data")

# Check if directory exist
if not data_dir.exists():
    # Create the directory if it does not exist
    print(f"Creating {data_dir} directory...")
    data_dir.mkdir(parents=True, exist_ok=True)

# Determine whether to download data based on if directory is empy
download =  not any(data_dir.iterdir())

if download:
    print(f"Directory is empty. Downloading Food101 Dataset in {data_dir}")
else: 
    print("Dataset exists. Skipping download...")

# Download and transform Food101 data - [SOURCE: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/]
train_data = torchvision.datasets.Food101(root=data_dir,
                                            split="train",
                                            transform=transform,
                                            download=download)

test_data = torchvision.datasets.Food101(root=data_dir,
                                            split="test",
                                            transform=transform,
                                            download=download)

# Remove the original archive file if downloaded
if download:
    for item in data_dir.glob("*tar.gz"):
        try:
            print("Removing original archive")
            item.unlink()
        except:
            pass

## Create smaller datasets and copy to subset directories - 20% of data
label_splits = get_subset(target_classes=["tacos", "waffles", "sushi"],
                          amount=0.2)

