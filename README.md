# FoodVisionMini

A series of experiments designed to grant a deeper understanding of Convolutional Neural Network (CNN) architectures.

This project aims to train various models using the PyTorch framework on a subset of the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)* to identify which model performs best. That model will then be used in the [FoodVision repo](https://github.com/mrkrisgee/FoodVision) to train the whole Data Set and perform predictions on images of food.

## Steps performed in the code
1. Downloads the *[Food101 Image Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)*.
2. Creates a smaller image data set from the original set and splits them into train and test data. This way, the models can train on smaller chunks of data for optimal speed. - *In our case, our subset consists of 20% of the data from the tacos, waffles, sushi food classes.*
3. Creates Train and Test DataLoaders to better handle large datasets in batches and apply data augmentation and shuffling.
4. 
