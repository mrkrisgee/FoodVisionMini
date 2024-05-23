# FoodVisionMini

A series of experiments designed to grant a deeper understanding of Convolutional Neural Network (CNN) architectures.

This project aims to train various models using the PyTorch framework on a subset of the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)* to identify which model performs best. That model will then be used in the [FoodVision repo](https://github.com/mrkrisgee/FoodVision) to train the whole Data Set and perform predictions on images of food.

## Steps performed in the code
1. Downloads the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*.
2. Creates a smaller image data set from the original set and splits them into train and test data. This way, the models can train on smaller chunks of data for optimal speed. - *In our case, our subset consists of 20% of the data from the <ins>tacos</ins>, <ins>waffles</ins>, <ins>sushi</ins> food classes.*
3. Creates *Train* and *Test* DataLoaders to better handle large datasets in batches and apply data augmentation and shuffling.
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

The architecture of these models can be examined in the [model_architectures](https://github.com/mrkrisgee/FoodVisionMini/tree/main/model_architectures) directory.
  
## Results
A few stat graphs visualised in TensorBoard:

### Accuracy
![Accuracy](https://github.com/mrkrisgee/FoodVisionMini/blob/main/results/Accuracy.png?raw=true)

### Loss
![Loss](https://github.com/mrkrisgee/FoodVisionMini/blob/main/results/Loss_curves.png)

### Metric outputs on last epoch and total training times for each model: 
- **TinyVGG**: Epoch: 15 | train_loss: 0.0401 | train_acc: 0.9875 | test_loss: 3.5182 | test_acc: 0.3676 <ins>Total training time:</ins> 378.383 seconds
- **efficientnet_b0**: Epoch: 15 | train_loss: 0.3901 | train_acc: 0.8500 | test_loss: 0.3570 | test_acc: 0.8631 <ins>Total training time:</ins> 407.153 seconds
- **efficientnet_b2**: Epoch: 15 | train_loss: 0.3912 | train_acc: 0.8562 | test_loss: 0.3956 | test_acc: 0.8847 <ins>Total training time:</ins> 441.000 seconds
- **efficientnetv2_s**: Epoch: 15 | train_loss: 0.4136 | train_acc: 0.8458 | test_loss: 0.4243 | test_acc: 0.8381 <ins>Total training time:</ins> 458.606 seconds
- **resnet101**: Epoch: 15 | train_loss: 0.2616 | train_acc: 0.9292 | test_loss: 0.4049 | test_acc: 0.8756 <ins>Total training time:</ins> 442.883 seconds   
  
By examining the curves, we can conclude that the TinyVGG model is overfitting, so that experiment can be automatically be deemed as failed.
By comparing the other four experiments, the **total training time** does not vary largely between them.
It can be concluded, however, that the accuracy predictions on the test data during the **efficientnet_b2** experiment, seem to be the most performative and this classification model will be used to train the larger Food101 Data Set.

The results of these experiments can be examined in the [results](https://github.com/mrkrisgee/FoodVisionMini/tree/main/results) directory
