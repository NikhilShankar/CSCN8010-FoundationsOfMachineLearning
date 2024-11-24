from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout


class CustomVGG16:


    def __init__(self):
        inputs = keras.Input(shape=(180, 180, 3))
        conv_base = keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(180, 180, 3))
        conv_base.trainable = False
        conv_base.summary()
        x = conv_base(inputs)
        print(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        self.model = keras.Model(inputs=inputs, outputs=output_layer)
