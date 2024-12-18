import os
import io

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

class CutMouthModel:
    def __init__(self):
        self.model = None

    def set_model(self, model_path: str):
        # Проверка существования файла модели
        if not os.path.exists(model_path):
            raise ValueError(f"Can't get model with path {model_path}. File does not exist.")

        try:
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)  # Загрузка модели YOLO
            self.model.to("cpu")
            self.names = self.model.names
            print(f"Model loaded successfully with the following classes: {self.names}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    @staticmethod
    def open_image_from_bytes(file: io.BytesIO):
        im = Image.fromarray(np.array(Image.open(file)))
        return im

    def analyze(self, photo_bytes: io.BytesIO, save_path: Path, confidence_threshold: float = 0.6):
        try:
            photo_bytes.seek(0)
            photo = self.open_image_from_bytes(photo_bytes)
            result = self.model(photo)
            print(f"Result: {result}")

            if len(result[0].boxes) == 0:
                return False, "Mouth not detected", None

            # Получаем предсказания и проверяем уверенность
            elem = result[0].boxes.data[0]
            confidence = elem[4].item()  # Уверенность (confidence score)
            if confidence < confidence_threshold:
                return False, f"Confidence below threshold: {confidence:.2f}", None

            image: Image.Image = Image.open(photo_bytes)

            n = 25  # Ареол вокруг обнаруженного объекта
            print(result[0].names)

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

            save_file_name = save_path / f"{tooth_type}_{original_image_name}.jpg"
            object_image.save(save_file_name)

            return (
                True,
                object_image,
                tooth_type,
            )
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False, f"Error during analysis: {e}", None

def check_black_corners(image_path, corner_threshold=50, square_size=10, required_black_corners=3):

    image = Image.open(image_path)

    try:
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        height, width = gray_array.shape

        def is_corner_black(corner_region):
            return np.mean(corner_region) < corner_threshold

        corners = {
            "top_left": gray_array[0:square_size, 0:square_size],
            "top_right": gray_array[0:square_size, width - square_size:width],
            "bottom_left": gray_array[height - square_size:height, 0:square_size],
            "bottom_right": gray_array[height - square_size:height, width - square_size:width]
        }

        black_corners_count = sum(is_corner_black(corner) for corner in corners.values())

        return black_corners_count >= required_black_corners
    finally:
        image.close()  # Закрытие изображения


def process_dataset(dataset_path, save_path, model_path, corner_threshold=50, square_size=10, required_black_corners=3):
    model = CutMouthModel()

    model.set_model(model_path)
    save_path = Path(save_path)
    #save_path.mkdir(parents=True, exist_ok=True)

    # Используем tqdm для временной полоски выполнения
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = Path(root)/file

                # Проверка на черные углы
                if not check_black_corners(image_path, corner_threshold, square_size, required_black_corners):
                    print(f"Черные углы не найдены для {image_path}, анализ нейронной сетью...")

                    with open(image_path, "rb") as image_file:
                        photo_bytes = io.BytesIO(image_file.read())
                        photo_bytes.name = image_path

                    result, message_or_image, tooth_type = model.analyze(photo_bytes, save_path)
                    if result:
                        print(f"Изображение {file} успешно обработано и сохранено.")
                    else:
                        print(f"Ошибка при обработке изображения {file}: {message_or_image}")
                else:
                    print(f"Изображение {file} пропущено из-за черных углов.")

# Пример использования:

if __name__ == "__main__":
    dataset_path = "D:\DATASET\Dental\Data_2"  # Путь к папке с датасетом
    save_path = "D:\DATASET\Dental\Data_result"  # Папка для сохранения обработанных изображений
    model_path = "TEMP_VAR.pt"  # Путь к модели YOLO

    process_dataset(dataset_path, save_path, model_path)