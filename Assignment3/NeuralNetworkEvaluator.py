from tensorflow.keras.utils import image_dataset_from_directory
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
import json
import os
from datetime import datetime


class NeuralNetworkEvaluator:

    def __init__(self, datafolder):
        data_folder = pathlib.Path(datafolder)
        self.train_dataset = image_dataset_from_directory(
            data_folder / "train",
            image_size=(180, 180),
            batch_size=32)
        self.validation_dataset = image_dataset_from_directory(
            data_folder  / "validation",
            image_size=(180, 180),
            batch_size=32)
        self.test_dataset = image_dataset_from_directory(
            data_folder  / "test",
            image_size=(180, 180),
            batch_size=32)
        return
        
    def trainandvalidate(self, model, datasetType, modelname, epochs_one = 30):
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        modelFolder = f"models/{datasetType}/{modelname}/{timestamp}"

        os.makedirs(modelFolder, exist_ok=True)
        model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy", "AUC", Precision(), Recall()])
        
        callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"./{modelFolder}/model.keras",
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss")]
        history = model.fit(
            self.train_dataset,
            epochs=epochs_one,
            validation_data=self.validation_dataset,
            callbacks=callbacks)
        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()
        return
    

    def evaluateOnTest(self, modelPath):
        test_model = keras.models.load_model(modelPath)
        test_loss, test_acc = test_model.evaluate(self.test_dataset)
        print(f"Test accuracy: {test_acc:.3f}")