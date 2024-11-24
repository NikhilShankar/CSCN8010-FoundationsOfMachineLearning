import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.vgg16 import preprocess_input


from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_curve,
    average_precision_score
)

class NeuralNetworkTestEvaluatorVGG:

    def __init__(self, datafolder, model, misClassifiedExamples = 5):
        data_folder = pathlib.Path(datafolder)
        test_dataset = image_dataset_from_directory(
            data_folder  / "test",
            image_size=(180, 180),
            batch_size=32,
            shuffle=False)
        class_names = test_dataset.class_names
        self.model = model
        self.dataset = self.preprocess_dataset(test_dataset)
        self.classnames = class_names
        self.misclassified = misClassifiedExamples

    def preprocess_dataset(self, dataset):
        print(dataset)
        result = dataset.map(lambda x,y: (preprocess_input(x), y))
        print(result)
        return result

    def get_predictions(self):
        """Get predictions and true labels from dataset"""
        y_pred_list = []
        y_true_list = []
        
        for images, labels in self.dataset:
            y_pred = self.model.predict(images)
            y_pred_list.append(y_pred)
            y_true_list.append(labels.numpy())
        y_pred = np.vstack(y_pred_list)
        y_true = np.hstack(y_true_list)
        
        return y_pred, y_true
    
    def getMisclassifiedImages(self, count=10):
        """Get predictions and true labels from dataset"""
        y_pred_list = []
        y_label_list = []
        
        misclass = 0
        for images, labels in self.dataset:
            y_pred = self.model.predict(images)
            y_pred_list = ((y_pred >= 0.5).astype(int))
            y_label_list = labels.numpy()
            for index, label in enumerate(y_label_list):
                if label != y_pred_list[index][0]:
                    print(f"Actual : {label} Predicted: {y_pred_list[index][0]}")
                    plt.figure(figsize=(5, 5))
                    plt.imshow(images[index].numpy().astype("uint8"))  # Convert tensor to uint8 image
                    plt.title(f"Label: {labels[index].numpy()}")
                    plt.axis("off") 
                    plt.show()
                    misclass+=1
                if misclass >= count:
                    break
        return

    def get_model_accuracy(self):
        """Get model accuracy on test set"""
        test_loss, test_accuracy, test_auc, test_precision, test_recall = self.model.evaluate(self.dataset)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        return test_accuracy

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        y_pred, y_true = self.get_predictions()
        y_pred_classes = (y_pred >= 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classnames,
                    yticklabels=self.classnames)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return cm

    def get_classification_metrics(self):
        """Get precision, recall, F1-score"""
        y_pred, y_true = self.get_predictions()
        y_pred_classes = (y_pred >= 0.5).astype(int)
        
        report = classification_report(y_true, y_pred_classes, 
                                    target_names=self.classnames,
                                    output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                target_names=self.classnames))
        return report

    def plot_precision_recall_curve(self):
        """Plot precision-recall curve for binary classification."""
        y_pred, y_true = self.get_predictions()  # Assumes y_pred contains scores for the positive class and y_true contains binary labels (0 or 1)
        plt.figure(figsize=(10, 7))
        # Compute precision, recall, and average precision
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, label=f'AP={ap:.2f}', color='b')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Binary Classification)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self):
        """Complete model evaluation for image classification"""
        print("Model Evaluation Results:")
        print("-" * 50)
        
        results = {}
        
        # 1. Accuracy
        results['accuracy'] = self.get_model_accuracy()
        
        # 2. Confusion Matrix
        results['confusion_matrix'] = self.plot_confusion_matrix()
        
        # 3. Classification Metrics
        results['classification_metrics'] = self.get_classification_metrics()
        
        # 4. Precision-Recall Curve
        self.plot_precision_recall_curve()
        
        self.getMisclassifiedImages(self.misclassified)

        return results

# Usage example:
"""
# Get class names from your dataset
class_names = test_dataset.class_names

# Run complete evaluation
results = evaluate_model(model, test_dataset, class_names)
"""