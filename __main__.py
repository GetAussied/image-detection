import random
import os
import numpy as np

from time import sleep
import __init__ as init


# Set up const values
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS_SIZE = 3

TRAININGLAYER1 = "./database/layer1.npz"
TRAININGLAYER2 = "./database/layer2.npz"
TRAININGLAYER3 = "./database/layer3.npz"

DETECTION_ARRAY = [
    ["artwork",         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    ["cars",            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    ["dishes",          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    ["furniture",       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    ["illustrations",   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    ["landmark",        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    ["meme",            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    ["packaged",        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    ["storefronts",     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    ["toys",            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
]


# initalisation of layers
layer_1 = init.hiddenlayer(512*512, 512, LEARNING_RATE)
layer_2 = init.hiddenlayer(512, 256, LEARNING_RATE)
layer_3 = init.hiddenlayer(256, len(DETECTION_ARRAY), LEARNING_RATE)


def LoadFileStatus():
    # Checks if save files are empty, if empty initalise files, else load files
    if (
        os.path.getsize(TRAININGLAYER1) == 0
        or os.path.getsize(TRAININGLAYER2) == 0
        or os.path.getsize(TRAININGLAYER3) == 0
    ):
        print("weights don't exists, generating weights and saving")

        layer_1.save(TRAININGLAYER1)
        layer_2.save(TRAININGLAYER2)
        layer_3.save(TRAININGLAYER3)
    else:
        print("loading saved model ...")

        layer_1.load(TRAININGLAYER1)
        layer_2.load(TRAININGLAYER2)
        layer_3.load(TRAININGLAYER3)
        
def TrainModel():
        epoch_loss = 0.0
        batch_count = 0

        # Checks if training data exists
        for epoch_count in range(EPOCHS_SIZE):
            print(f"Epoch number: {epoch_count}")
            
            try:
                print("starting image batching")

                #THIS GETS ALL IMAGES AND SPLITS INTO BATCHES
                images = init.generatebatch("./train", BATCH_SIZE)
                batches = images.getbatches(DETECTION_ARRAY)

                print("Images vectorised. Batches start now...")

                LoadFileStatus()

                for batch_x, batch_y in batches:

                    output_1 = layer_1.forward(batch_x)
                    output_2 = layer_2.forward(output_1)
                    output_3 = layer_3.forward(output_2)

                    batch_loss = np.mean((output_3 - batch_y) ** 2)
                    epoch_loss += batch_loss
                    batch_count += 1

                    grad_loss = 2 * (output_3 - batch_y) / batch_y.shape[0]

                    grad_1 = layer_3.backward(grad_loss)
                    grad_2 = layer_2.backward(grad_1)
                    _ = layer_1.backward(grad_2)

                    print(f'\n Epoch Count -> {epoch_count}: Batch count: {batch_count}\n\n')

                epoch_count = 0
                epoch_count += 1

                epoch_loss /= batch_count


                print(f"saving model ... Do Not Exit")

                layer_1.save(TRAININGLAYER1)
                layer_2.save(TRAININGLAYER2)
                layer_3.save(TRAININGLAYER3)

                print(f"\n Epoch Count -> {epoch_count}: Model Loss -> {epoch_loss:.6f}\n\n")

            except KeyboardInterrupt:
                print(f"Keyboard Interruption")
                exit()

def RunModel(image_location: str):
    selected_image = init.image2vector(image_location)

    output_1 = layer_1.forward(selected_image)
    output_2 = layer_2.forward(output_1)
    output_3 = layer_3.forward(output_2)

    print(
        f"""
        Prediction:

        "Art work" {round((output_3)[0]*100)}
        "Cars" {round((output_3)[1]*100)}
        "Dishes" {round((output_3)[2]*100)}
        "Furniture" {round((output_3)[3]*100)}
        "Illustration" {round((output_3)[4]*100)}
        "Landmark" {round((output_3)[5]*100)}
        "Meme" {round((output_3)[6]*100)}
        "Packaged" {round((output_3)[7]*100)}
        "Storefronts" {round((output_3)[8]*100)}
        "Toys" {round((output_3)[9]*100)}
        """
    )


def command():
    userinput = input("Enter Command: ")

    if userinput == "train":

        while True:
            TrainModel()
            sleep(0.5)

    elif userinput == "run":

        RunModel(input("Enter Image Location: "))

    elif userinput == "deldata":

        for file in [TRAININGLAYER1, TRAININGLAYER2, TRAININGLAYER3]:
            with open(file, "w"):
                pass

    else:
        exit()


if __name__ == "__main__":
    command()
