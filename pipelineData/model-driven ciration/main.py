import io
import os
from pathlib import Path
from typing import Tuple
import sys

import cv2
from PIL.Image import Image
from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm



class CutMouthModel:
    def __init__(self):
        self.model = None

    def set_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise ValueError(f"Can't get model with path {model_path}")
        self.model = YOLO(model_path)
        self.model.to("cpu")
        self.names = self.model.names
        #print(self.names)

    @staticmethod
    def open_image_from_bytes(file: io.BytesIO):
        im = Image.fromarray(np.array(Image.open(file)))
        return im

    def analyze(self, photo_bytes: io.BytesIO, save_path: Path, confidence_threshold: float = 0.6):
        try:
            photo_bytes.seek(0)
            photo = self.open_image_from_bytes(photo_bytes)

            # Перенаправляем стандартный вывод
            original_stdout = sys.stdout  # Сохраняем стандартный вывод
            sys.stdout = open(os.devnull, 'w')  # Перенаправляем в "черную дыру"

            result = self.model(photo)
            #print(f"Result: {result}")

            if len(result[0].boxes) == 0:
                return False, "Mouth not detected", None

            # Получаем предсказания и проверяем уверенность
            elem = result[0].boxes.data[0]
            confidence = elem[4].item()  # Уверенность (confidence score)
            if confidence < confidence_threshold:
                return False, f"Confidence below threshold: {confidence:.2f}", None

            image: Image.Image = Image.open(photo_bytes)

            n = 25  # Ареол вокруг обнаруженного объекта
            #print(result[0].names)

            # Координаты с учетом ареола
            x_min = int(elem[0]) - n if int(elem[0]) > n else 0
            y_min = int(elem[1]) - n if int(elem[1]) > n else 0

            # Убедимся, что x_max и y_max не выходят за границы изображения
            x_max = int(elem[2]) + n if int(elem[2]) + n < image.width else image.width
            y_max = int(elem[3]) + n if int(elem[3]) + n < image.height else image.height

            tooth_type = self.names[int(elem[5])].split()[0]
            object_image = image.crop((x_min, y_min, x_max, y_max))

            # Получение имени оригинального изображения
            original_image_name = "original_image"  # По умолчанию, если имени нет
            if hasattr(photo_bytes, 'name'):
                original_image_name = os.path.basename(photo_bytes.name).split('.')[0]

            save_file_name = save_path / f"{confidence:.2f}_{tooth_type}_{original_image_name}.jpg"
            object_image.save(save_file_name)

            return (
                True,
                object_image,
                tooth_type,
            )

        except Exception as e:
            #print(f"Error during analysis: {e}")
            return False, f"Error during analysis: {e}", None


# Пример использования скрипта
if __name__ == "__main__":
    model_path = "TEMP_VAR.pt"
    image_directory = Path("D:/DATASET/Dental/Data_preresult")  # Папка с изображениями
    save_directory = Path(r"D:\DATASET\Dental\Data_result")
    save_directory.mkdir(exist_ok=True)

    cut_mouth_model = CutMouthModel()
    cut_mouth_model.set_model(model_path)

    image_paths = list(image_directory.glob("*.jpg")) + list(image_directory.glob("*.jpeg")) + list(image_directory.glob("*.png"))

    # Проходим по всем файлам в папке
    for image_path in tqdm(image_paths, desc="Processing images"):
        with open(image_path, "rb") as image_file:
            photo_bytes = io.BytesIO(image_file.read())
            photo_bytes.name = str(image_path)

        _, _, _ = cut_mouth_model.analyze(photo_bytes, save_directory)



