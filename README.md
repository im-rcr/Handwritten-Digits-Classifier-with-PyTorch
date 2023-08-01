# MNIST Handwritten Digits Classifier with PyTorch

This repository contains an implementation of a handwritten digits classifier using PyTorch. The model is trained to recognize handwritten digits from the famous MNIST dataset.

## Introduction

The goal of this project is to build a neural network-based classifier that can accurately recognize handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of hand-drawn digits from 0 to 9.


## Requirements

To run the code in this project, you need the following dependencies:

- Python 3
- PyTorch (>= 1.7.0)
- torchvision
- matplotlib
- numpy

## Steps to Follow

1. **Imports**: Import the necessary libraries and modules, including PyTorch, torchvision, and matplotlib.

2. **Load the dataset**: Load the MNIST dataset using torchvision and create data loaders for both the train and test sets.

3. **Justify your preprocessing**: Explain the preprocessing steps chosen, such as `ToTensor()` and `Normalize()`, and their importance.

4. **Explore the dataset**: Use matplotlib and torch to explore the data dimensions and view sample images.

5. **Build your neural network**: Design the neural network architecture using PyTorch's `nn.Module`.

6. **Specify a loss function and an optimizer**: Define the loss function and optimizer for training the model.

7. **Running your neural network**: Train the model using the training data and validate it using the test data.

8. **Plot the training loss**: Visualize the training loss over epochs to monitor the model's training progress.

9. **Testing your model**: Evaluate the trained model's performance on the test set and calculate the accuracy.

10. **Improving your model**: Suggest potential improvements to the model, such as architecture changes or hyperparameter tuning.

11. **Saving your model**: Save the trained model for future use.

## Result

The trained model achieves an accuracy of approximately 97.73% on the test set. This performance can be further improved by fine-tuning hyperparameters or exploring different model architectures.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The model architecture and training techniques are inspired by various online tutorials and resources.
- Thanks for Udacity Team for giving me the opportunity and necessary resources to work on this project.
- Thanks to the PyTorch community for providing an excellent deep learning framework.

