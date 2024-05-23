"""
Contains functionality for generating data subsets from 
larger datasets.

Originally sourced from github.com/mrdbourke -
[https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/04_custom_data_creation.ipynb]
"""
import random
import pathlib
import shutil

# Setup data paths and splits
data_dir = pathlib.Path("data")
data_path = data_dir / "food-101" / "images"
data_splits = ["train", "test"]

def get_subset(image_path=data_path,
               target_classes=["pizza", "steak", "sushi"],
               amount=0.2, # Change amount of data to get (e.g. 0.1 = random 10%, 0.2 = random 20%)
               seed=42):
    """Splits data into subsets consisting of a percentage
    amount of target classes. If subset exists, the process
    is skipped.

    Args:
        image_path (str): Where the data is located
        data_splits (List, str): Directories to split the data into
        target_classes (List, str): Classes of directories to create subset from
        amount (float): Percentage amount of data to get
        seed (int): Seed variation

    Returns:
        A List of paths with randomly sampled data from the given dataset.
    """
    path_check_bool = True
    target_dir_name = pathlib.Path(f"data/subset_{str(int(amount*100))}_percent")

    # Create boolean to check if subset already exists
    for split in data_splits:
        for target_class in target_classes:
            path_check = target_dir_name / split / target_class

            if not target_dir_name.exists():
                path_check_bool = False
            else:
                try:
                    for item in path_check.iterdir():
                        if item.is_dir() and item.name != target_class:
                            path_check_bool = False
                except:
                    path_check_bool = False
              
    random.seed(seed)
    label_splits = {}

    # Check if subset already created
    if not path_check_bool:
        # Get labels
        for data_split in data_splits:
            print(f"\n Creating image split for: {data_split}...")
            label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"
            with open(label_path, "r") as f:
                labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

            # Get random subset of target classes image ID's
            number_to_sample = round(amount * len(labels))
            print(f" Getting random subset of {number_to_sample} images for {data_split}...")
            sampled_images = random.sample(labels, k=number_to_sample)
            
            # Apply full paths
            image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
            label_splits[data_split] = image_paths

        ## Move training and testing images to dedicated folders        
        # Setup the directories
        target_dir = pathlib.Path(target_dir_name)

        # Create the target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n Creating directory: '{target_dir_name}'")

        # Copy to dedicated directories
        for image_split in label_splits.keys():
            for image_path in label_splits[str(image_split)]:
                dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
                if not dest_dir.parent.is_dir():
                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest_dir)
    else:
        print(f"\n Subset already exists in {target_dir_name}. Skipping...")

    # Save trainining and testing directory paths
    train_dir = target_dir_name / "train"
    test_dir = target_dir_name / "test"

    return train_dir, test_dir

    