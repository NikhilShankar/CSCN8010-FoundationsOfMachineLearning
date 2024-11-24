import os
from pathlib import Path
from PIL import Image
import pandas as pd


class EDAImage:
    def __init__(self, class_dir_map: dict):
        """
        Initialize the class with a map of class names to directory paths.
        
        Args:
        - class_dir_map (dict): A dictionary where keys are class names and values are relative directory paths.
        """
        self.class_dir_map = {cls: Path(dir_path) for cls, dir_path in class_dir_map.items()}
    
    def get_image_counts(self):
        """
        Returns a dataframe with the count of images for each class.
        
        Returns:
        - pd.DataFrame: A dataframe with columns 'Class' and 'ImageCount'.
        """
        data = []
        for class_name, dir_path in self.class_dir_map.items():
            if dir_path.is_dir():
                image_count = len([f for f in dir_path.iterdir() if f.is_file()])
                data.append({"Class": class_name, "ImageCount": image_count})
            else:
                data.append({"Class": class_name, "ImageCount": 0})
        return pd.DataFrame(data)
    
    def get_image_metadata(self):
        """
        Returns a dataframe with image height, width, and class for each image.
        
        Returns:
        - pd.DataFrame: A dataframe with columns 'Class', 'Image', 'Height', and 'Width'.
        """
        data = []
        for class_name, dir_path in self.class_dir_map.items():
            if dir_path.is_dir():
                for image_path in dir_path.iterdir():
                    if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            with Image.open(image_path) as img:
                                width, height = img.size
                                data.append({
                                    "Class": class_name,
                                    "Image": image_path.name,
                                    "Height": height,
                                    "Width": width
                                })
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
        return pd.DataFrame(data)
