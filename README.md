# FoodVisionMini

A series of experiments designed to grant a deeper understanding of Convolutional Neural Network (CNN) architectures.

This project aims to train various models using the PyTorch framework on a subset of the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)* to identify which model performs best. That model will then be used in the [FoodVision repo](https://github.com/mrkrisgee/FoodVision) to train the whole Data Set and perform predictions on images of food.

## Steps performed in the code
1. Downloads the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*.
2. Creates a smaller image data set from the original set and splits them into train and test data. This way, the models can train on smaller chunks of data for optimal speed. - *In our case, our subset consists of 20% of the data from the <ins>tacos</ins>, <ins>waffles</ins>, <ins>sushi</ins> food classes.*
3. Creates Train and Test DataLoaders to better handle large datasets in batches and apply data augmentation and shuffling.
4. Sets up experiment iterations by:
    - Instantiating training models.
    - Defining loss functions and optimizers.
    - Training the models.
    - Tracking each experiment with Tensorboard.
    - Saving out the models.

## Models used in experimentation
For the sake of these experiments, the data was trained by several classification models.
* A custom-built classification model replicating the **`TinyVGG`** architecture as explained on *[The CNN Explainer](https://poloclub.github.io/cnn-explainer/)* website.
* The [**`efficientnet_b0`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#efficientnet-b0) classification model and weights.
* The [**`efficientnet_b2`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#efficientnet-b2) classification model and weights.
* The [**`efficientnetv2_s`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#efficientnet-v2-s) classification model and weights.
* The [**`resnet101`**](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#resnet101) classification model and weights.

  
## Results
