"""
Contains code for custom built and PyTorch architectures
to instantiate various image classification models.
"""
import torch
import torchvision
import torchvision.models as models

from torch import nn
from utils import set_seeds, set_device

class TinyVGG(nn.Module):
    """
    Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    device = set_device()

    def __init__(self, hidden_units: int, output_shape: int,  input_shape: int=3) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                    stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
            stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 56 * 56, out_features=output_shape)  # Adjusting for the size after two max pools
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

def create_effnetb0(OUT_FEATURES: int):
    """
    efficientnet_b0 architecture - base model
    See the original architecture here:
    https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
    """
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    dropout = 0.2

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"\n[INFO] Created new {model.name} model.")
    return model

def create_effnetb2(OUT_FEATURES: int):
    """
    efficientnet_b2 architecture - base model
    See the original architecture here:
    https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html
    """
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    dropout = 0.3

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1408, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnet_b2"
    print(f"\n[INFO] Creating {model.name} feature extractor model...")
    return model

def create_effnetv2_s(OUT_FEATURES: int):
    """
    efficientnet_v2_s architecture - base model
    See the original architecture here:
    https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html
    """
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights).to(device)
    dropout = 0.2

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetv2_s"
    print(f"\n[INFO] Creating {model.name} feature extractor model...")
    return model

def create_resnet101(OUT_FEATURES: int):
    """
    resnet101 architecture - base model
    See the original architecture here:
    https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#resnet101
    """
    weights = models.ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the fc head
    model.fc = nn.Linear(in_features=2048, out_features=OUT_FEATURES
    ).to(device)

    # 5. Give the model a name
    model.name = "resnet101"
    print(f"\n[INFO] Creating {model.name} feature extractor model...")
    return model    

