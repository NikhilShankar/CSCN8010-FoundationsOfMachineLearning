from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class CustomNeuralNetwork2Augmented:

    def __init__(self):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomBrightness(factor=0.2),
            ]
        )
        inputs = keras.Input(shape=(180, 180, 3))
        #Additional change
        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

        #This is the only change
        x = layers.Conv2D(filters=196, kernel_size=2, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)