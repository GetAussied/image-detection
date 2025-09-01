from PIL import Image, ImageOps
import random
import os
import numpy as np


class generatebatch:
    """
    Generate batches of images for training or processing.

    This class handles loading, shuffling, and splitting images
    into batches, along with any associated labels.
    """


    def __init__(self, dataset_root: str, batch_size: int):
            self.dataset_root = dataset_root
            self.batch_size = batch_size

    def getbatches(self, category: list):
        """
        Get all images, vectorize them, shuffle, and split into batches.

        Args:
            category (list): A list of folder names and corresponding labels.

        Yields:
            tuple: 
                A batch of images and their corresponding labels.
                The function:
                    - Gets all folders from the root.
                    - Extracts images from each folder.
                    - Shuffles all images.
                    - Yields batches of the given size with their labels.
        """
        
        self.batches = []
        images = []

        for folder_name, label in category:

            directory = os.path.join(self.dataset_root, folder_name)

            images.extend(
                (os.path.join(directory,f), label) for f in os.listdir(directory) if f.lower().endswith((".jpeg", ".jpg", ".png"))
            )

        random.shuffle(images)

        def batches(lst, batch_size):
            for index in range(0, len(lst), batch_size):

                try:
                    yield lst[index:index+batch_size]
                except Exception as e:
                    print(f" File path error: {lst[index]}, {e}")

        for batch in batches(images, self.batch_size):

            try:
                batch_x = np.array([image2vector(image) for image, _ in batch]) 
                batch_y = np.array([label for _, label in batch])

                yield (batch_x, batch_y)
                
            except Exception as e:
                print(f"{e}")
        
# Generate neural network
class hiddenlayer:
    """
    A fully connected hidden layer with sigmoid activation.

    This layer stores its weights and biases, performs forward
    and backward propagation, and updates parameters using 
    gradient descent.

    Attributes:
        weight (numpy.ndarray): Weight matrix of the layer.
        bias (numpy.ndarray): Bias vector of the layer.
        learning_rate (float): Step size for parameter updates.
    """

    def __init__(self, input_size, output_size, learning_rate: float):
        self.learning_rate = learning_rate
        self.weight = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, input: np.ndarray):
        """
        Forward pass of the neural network using the sigmoid activation function.

        Args:
            input (numpy.ndarray): Input array to the neural network.

        Returns:
            numpy.ndarray: Output after applying the sigmoid function.
        """

        self.input = input
        hidden_input = np.dot(input, self.weight) + self.bias
        self.output = 1 / (1 + np.exp(-hidden_input))  # Sigmoid

        return self.output

    def backward(self, grad_output):
        """
        Perform the backward pass of the layer and update its parameters.

        This computes gradients with respect to weights, biases, and inputs 
        using the chain rule, then updates weights and biases using gradient descent.

        Args:
            grad_output (numpy.ndarray): Gradient of the loss with respect 
                to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """

        grad_z = grad_output * self.output * (1 - self.output)

        grad_weights = self.input.T @ grad_z / self.input.shape[0]
        grad_biases = np.sum(grad_z, axis=0) / self.input.shape[0]
        grad_input = grad_z @ self.weight.T

        # Update weights and biases using gradient descent
        self.weight -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_biases

        # Return gradient for previous layer
        return grad_input

    def save(self, path: str):
        """
        Save the calculated weights and biases to a file.

        Args:
            path (str): Path to the file where the parameters will be saved.
        """

        np.savez(
            path,
            weight=self.weight,
            bias=self.bias,
        )

    def load(self, path: str):
        """
        Load saved weights and biases from a file.

        Args:
            path (str): Path to the file containing the saved parameters.
        """

        data = np.load(path)
        self.weight = data["weight"]
        self.bias = data["bias"]


# Converts image to grayscale and produces vector
def image2vector(path: str):
    """
    Load an image, convert it to grayscale, and flatten it to 512x512.

    Args:
        path (str): Relative path to the image file.

    Returns:
        numpy.ndarray: Flattened 512x512 grayscale image.
    """

    with Image.open(path) as image:
        image = ImageOps.grayscale(image)
        image = image.resize((512, 512))

        image_array = np.array(image)

        selected_image = image_array / 255.0
        selected_image = selected_image.flatten()

    return selected_image