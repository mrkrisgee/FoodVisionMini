# FoodVisionMini

FoodVisionMini is a series of experiments aimed at gaining a deeper understanding of Convolutional Neural Network (CNN) architectures. This project trains various models using the PyTorch framework on a subset of the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)* to identify the best-performing model. The selected model will then be used in the [FoodVision](https://github.com/mrkrisgee/FoodVision) repository to train on the entire dataset and perform image predictions.

## Steps performed in the code
1. **Download the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*.**
   - Obtain the original dataset.
     
2. **Create a Smaller Subset**:
   - Generate a smaller dataset from the original, consisting of 20% of the images from the <ins>tacos</ins>, <ins>waffles</ins>, and <ins>sushi</ins> classes.
     
3. **Prepare DataLoaders:**
   - Implement Train and Test DataLoaders to manage large datasets in batches.
   - Apply data augmentation and shuffling.

4. **Set Up Experiment Iterations:**
   - Instantiate training models.
   - Define loss functions and optimizers.
   - Train the models.
   - Track experiments using TensorBoard.
   - Save the trained models.

## Models used in experimentation

The following classification models were used for training on the subset data:

* A custom-built classification model replicating the **`TinyVGG`** architecture as explained on *[The CNN Explainer](https://poloclub.github.io/cnn-explainer/)* website.
* The [**`EfficientNet-B0`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#efficientnet-b0) classification model with weights.
* The [**`EfficientNet-B2`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#efficientnet-b2) classification model with weights.
* The [**`EfficientNetV2-S`**](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#efficientnet-v2-s) classification model with weights.
* The [**`ResNet101`**](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#resnet101) classification model with weights.

The architecture of these models can be examined in the [`model_architectures`](https://github.com/mrkrisgee/FoodVisionMini/tree/main/model_architectures) directory.
  
## Results

A few stat graphs visualized in TensorBoard:

### Accuracy
![Accuracy](https://github.com/mrkrisgee/FoodVisionMini/blob/main/results/Accuracy.png?raw=true)

### Loss
![Loss](https://github.com/mrkrisgee/FoodVisionMini/blob/main/results/Loss_curves.png)

### Metric Outputs on Last Epoch and Total Training Times for Each Model:

- **TinyVGG**: Epoch: 15 | Train Loss: 0.0401 | Train Accuracy: 0.9875 | Test Loss: 3.5182 | Test Accuracy: 0.3676 | Total Training Time: 378.383 seconds
- **EfficientNet-B0**: Epoch: 15 | Train Loss: 0.3901 | Train Accuracy: 0.8500 | Test Loss: 0.3570 | Test Accuracy: 0.8631 | Total Training Time: 407.153 seconds
- **EfficientNet-B2**: Epoch: 15 | Train Loss: 0.3912 | Train Accuracy: 0.8562 | Test Loss: 0.3956 | Test Accuracy: 0.8847 | Total Training Time: 441.000 seconds
- **EfficientNetV2-S**: Epoch: 15 | Train Loss: 0.4136 | Train Accuracy: 0.8458 | Test Loss: 0.4243 | Test Accuracy: 0.8381 | Total Training Time: 458.606 seconds
- **ResNet101**: Epoch: 15 | Train Loss: 0.2616 | Train Accuracy: 0.9292 | Test Loss: 0.4049 | Test Accuracy: 0.8756 | Total Training Time: 442.883 seconds

By examining the curves, we can conclude that the TinyVGG model is overfitting, so that experiment can be deemed a failure. Comparing the other four experiments, the total training time does not vary significantly between them. However, the accuracy predictions on the test data during the EfficientNet-B2 experiment seem to be the most performant, and this classification model will be used to train the larger Food101 dataset.

The results of these experiments can be examined in the [`results`](https://github.com/mrkrisgee/FoodVisionMini/tree/main/results) directory.
