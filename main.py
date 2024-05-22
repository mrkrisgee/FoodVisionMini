import torch
import torchvision
import pathlib
import utils
import model_builder

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
device = utils.set_device()

## Get data
print(f"\n[INFO] Retrieving Food101 DataSet")
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

# Determine whether to download data based on if directory is empty
download =  not any(data_dir.iterdir())

if download:
    print(f"\n Directory is empty. Downloading Food101 Dataset in {data_dir}")
else: 
    print("\n Dataset exists. Skipping download...")

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
print(f"\n[INFO] Creating data subsets")
train_dir, test_dir = get_subset(target_classes=["tacos", "waffles", "sushi"],
                          amount=0.2)

## Create Datasets and DataLoaders
print(f"\n[INFO] Creating Datasets and DataLoaders")
train_dataloader, test_dataloader, class_names = utils.create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        transform=transform,
                                                                        batch_size=32)

## Setup the model
# Get number of output channels (one for each class)
OUT_FEATURES = len(class_names)

## TinyVGG Model
# tiny_vgg = model_builder.TinyVGG(hidden_units=10, output_shape=OUT_FEATURES).to(device)
# tiny_vgg.name = "TinyVGG"
# print(f"\n[INFO] Created new {tiny_vgg.name} model.")

## Effnetb0 Model
# effnetb0 = model_builder.create_effnetb0(OUT_FEATURES)

##Effnetb2 Model
# effnetb2 = model_builder.create_effnetb2(OUT_FEATURES)

## Effnetv2_s Model
# effnetv2_s = model_builder.create_effnetv2_s(OUT_FEATURES)

## ResNet101 Model
# resnet101 = model_builder.create_resnet101(OUT_FEATURES)

# 2. Get a summary of Models from torchvision.models (uncomment for full output)
# summary(model=resnet101, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ) 

## TRAIN MODEL

