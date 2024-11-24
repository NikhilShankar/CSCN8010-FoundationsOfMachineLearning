from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout


class CustomVGG16Aug:


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
        conv_base = keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(180, 180, 3))
        conv_base.trainable = True
        for layer in conv_base.layers[:-2]:
            layer.trainable = False
        x = data_augmentation(inputs)
        x = conv_base(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        self.model = keras.Model(inputs=inputs, outputs=output_layer)
        print(self.model.summary())
