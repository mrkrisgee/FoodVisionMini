import torch
import torchvision
import pathlib
import utils
import model_builder
import engine

from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

from subset_gen import get_subset


def main():
    # Get torchininfo. Install if it doesn't exist
    try:
        from torchinfo import summary
    except:
        print("[INFO] Couldn't find torchinfo... installing it.")
        from torchinfo import summary

    # Setup device agnostic code
    device = utils.set_device()
    print(f"Device being used: {device}")

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

    ## Init experiment parameters 
    # 1.  A list of the number of epochs we'd like to test
    num_epochs = [5]

    # 2. Create models list (new model needed for each experiment) - A list of the models we'd like to test
    models = ["effnetb0"]

    # 3. Create dataloaders dictionary- A dictionary of the different training DataLoaders (if any)
    train_dataloaders ={"data": train_dataloader}

    ## Init experiment iterations
    # 1. Set random seeds
    utils.set_seeds()

    # 2. Keep track of experiment numbers
    experiment_number = 0

    # 3. Loop through each DataLoader
    for dataloader_name, train_dataloader in train_dataloaders.items():

        # 4. Loop through each number of epochs
        for epochs in num_epochs:

            # 5. Loop through each model name and create a new model based on the name
            for model_name in models:

                # 6. Information print out
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")  

                # 7. Select the model
                if model_name == "tinyvgg":
                    model = model_builder.TinyVGG(hidden_units=10, output_shape=OUT_FEATURES).to(device)
                    model.name = "TinyVGG"
                    print(f"\n[INFO] Created new {model.name} model.")
                elif model_name == "effnetb0":
                    model = model_builder.create_effnetb0(OUT_FEATURES)
                elif model_name == "effnetb2":
                    model = model_builder.create_effnetb2(OUT_FEATURES)
                elif model_name == "effnetv2_s":
                    model = model_builder.create_effnetv2_s(OUT_FEATURES)
                elif model_name == "resnet101":
                    model = model_builder.create_resnet101(OUT_FEATURES)

                # 8. Define loss function and optimizer
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Start the timer
                start_time = timer()

                # 9. Train target model and track experiments
                engine.train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=5,
                            device=device,
                            writer=utils.create_writer(experiment_name=dataloader_name,
                                                model_name=model_name,
                                                extra=f"{epochs}_epochs"))

                # End the timer
                end_time = timer()
                print(f"Total training time: {end_time-start_time:.3f} seconds")
                
                # 10. Save the model to file
                save_file_path = f"This is my first test_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                save_model(model=effnetb0,
                        target_dir="models",
                        model_name=save_filepath)
                print("-"*50 + "\n")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Use the 'spawn' start method
    main()

