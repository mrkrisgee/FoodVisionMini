import main
import model_builder, utils

import torch
import torchvision
import torchvision.models as models

from torch import nn
from torchinfo import summary

def get_model_summary(model: torch.nn.Module):
    """
    Contains code to instantiate and
    then output a summary of the 
    model architectures behind the various
    classification models added in
    the model_builder.py file.

    Args:
    model: "tinyvgg", "effnetb0", "effnetb2", "effnetv2_s", "resnet101"

    Current classification models include:
    ##  Custom Tiny VGG model replication
        based on the CNN Explainer website:
        https://poloclub.github.io/cnn-explainer/
    
    ##  efficientnet_b0 architecture - base model
        See the original architecture here:
        https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html

    ##  efficientnet_b2 architecture - base model
        See the original architecture here:
        https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html

    ##  efficientnet_v2_s architecture - base model
        See the original architecture here:
        https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html

    ##  resnet101 architecture - base model
        See the original architecture here:
        https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#resnet101

    Example use:
    get_model_summary("effnetb0")

    """
    OUT_FEATURES = main.main(fetch_attr=True)

    # Set up device agnostic code.
    device = utils.set_device()

    # Instantiate custom TinyVGG model
    if model == "tinyvgg":
        model = model_builder.TinyVGG(hidden_units=10, output_shape=OUT_FEATURES).to(device)
        model.name = "TinyVGG"
        print(f"\n[INFO] Created new {model.name} model.")
    # Instantiate efficientnet_b0 model
    elif model == "effnetb0":
        model = model_builder.create_effnetb0(OUT_FEATURES)
    # Instantiate efficientnet_b2 model
    elif model == "effnetb2":
        model = model_builder.create_effnetb2(OUT_FEATURES)
    # Instantiate efficientnet_v2_s model
    elif model == "effnetv2_s":
        model = model_builder.create_effnetv2_s(OUT_FEATURES)
    # Instantiate resnet101 model
    elif model == "resnet101":
        model = model_builder.create_resnet101(OUT_FEATURES)

    # 2. Get a summary of Models
    summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

get_model_summary("tinyvgg")
