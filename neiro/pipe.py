import numpy as np

from cut_image import cut_mouth_model, cut_images
from single_tooth import single_tooth_model
from caries import caries_model
from transform import image_to_numpy, save_dataclass, save_image_from_numpy

from PIL import Image, ImageDraw
from typing import List
from pathlib import Path

# 0: "Обнаружен Кариес",
# 1: "Небольшое повреждение",
# 2: "Здоров "

output_boxes_folder = Path("./results/cropped_boxes")
output_drawn_path = Path("./results/drawn_boxes_7.jpg")

test_image_1 = image_to_numpy("./results/0.90_Front_Frame_1146_JPG.jpg")
test_image_2 = image_to_numpy("./results/0.95_Upper_2023-08-12-05-36-04_Upper_jpg.jpg")
test_image_3 = image_to_numpy("./results/0.96_Lower_253_jpg.jpg")
test_list = [test_image_1, test_image_2, test_image_3]


def pipeline_caries(list_images: List[np.ndarray]):
    result_list = []
    result_dict = {}
    exit_code, error, datacls_list = cut_images(list_images)
    if exit_code is False:
        return error, result_list, result_dict

    for datacls in datacls_list:
        single_tooth_model.analyze(datacls, conf_threshold=0.2)
        caries_model.analyze(datacls)
        image = Image.fromarray(datacls.array)
        draw = ImageDraw.Draw(image)
        type_of_image = datacls.mouth_type

        caries_count = {
            "type_0": 0,  # Количество кариеса типа 0
            "type_1": 0  # Количество кариеса типа 1
        }

        for caries in datacls.caries_coord:
            x_min, y_min, x_max, y_max = caries.caries_coord
            if caries.caries_type == 0:
                color = "red"
                caries_count["type_0"] += 1
            elif caries.caries_type == 1:
                color = "blue"
                caries_count["type_1"] += 1
            else:
                continue
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        upd_img = np.array(image)
        result_list.append(upd_img)
        result_dict[type_of_image] = caries_count

    return None, result_list, result_dict

def save_resylts(list_images: List[np.ndarray]):
    for index, image in enumerate(list_images):
        save_image_from_numpy(image, f"results_pipe_{index}.jpg")

#save_dataclass(datacls_list[1], Path("./results/mouth_image_1.json"))


error, result_list, result_dict = pipeline_caries(test_list)
if error is None:
    save_resylts(result_list)
    print(result_dict)
else:
    print(error)
