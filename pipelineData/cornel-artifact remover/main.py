import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_corners(image_path, black_corner_threshold=50, white_corner_threshold=200, square_size=10,
                  required_black_corners=3, required_white_corners=3,):

    image = Image.open(image_path)

    try:
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        height, width = gray_array.shape

        def is_corner_black(corner_region):
            # Проверяем, является ли среднее значение угла ниже порога (черный угол)
            return np.mean(corner_region) < black_corner_threshold

        def is_corner_white(corner_region):
            # Проверяем, является ли среднее значение угла выше порога (белый угол)
            return np.mean(corner_region) > white_corner_threshold

        # Извлекаем области углов
        corners = {
            "top_left": gray_array[0:square_size, 0:square_size],
            "top_right": gray_array[0:square_size, width - square_size:width],
            "bottom_left": gray_array[height - square_size:height, 0:square_size],
            "bottom_right": gray_array[height - square_size:height, width - square_size:width]
        }

        black_corners_count = sum(is_corner_black(corner) for corner in corners.values())
        white_corners_count = sum(is_corner_white(corner) for corner in corners.values())

        black_corners_result = black_corners_count >= required_black_corners
        white_corners_result = white_corners_count >= required_white_corners

        if black_corners_result or white_corners_result:
            return True
        else:
            return False
    finally:
        image.close()



def process_images(dataset_path,
                   save_path,
                   black_corner_threshold=50,
                   white_corner_threshold=200,
                   square_size=5,
                   required_black_corners=2):


    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  # Создаем папку для сохранения изображений, если её нет

    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = Path(root) / file

                # Проверка изображения на черные углы
                if not check_corners(image_path,
                                     black_corner_threshold,
                                     white_corner_threshold,
                                     square_size,
                                     required_black_corners):

                    save_file_path = save_path / file
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            img = img.convert("RGB")
                        img.save(save_file_path)



if __name__ == "__main__":
    dataset_path = "D:/DATASET/Dental/Data_7"  # Путь к папке с изображениями
    save_path = "D:/DATASET/Dental/Data_preresult"  # Путь для сохранения изображений, которые прошли проверку
    process_images(dataset_path, save_path)


