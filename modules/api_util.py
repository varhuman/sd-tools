import httpx
from modules.api_models import Txt2ImgModel, Img2ImgModel,CheckpointModel
import modules.data_manager as data_manager
import gradio as gr
import modules.file_util as file_util
import base64
from PIL import Image
import json
import os
from io import BytesIO
import requests
import modules.template_utils as template_utils

modeltitle = "nightSkyYOZORAStyle_yozoraV1Origin.safetensors [e7bf829cff]"
input_image_url = "C:/Users/Ieunn/Pictures/00013-814328471.jpg"
input_mask_url = ""
setup_args = {
    "prompt": "white hair girl, red eyes, fashionclothes", #for txt2img
    #"init_images": "imgurl", #for img2img
    "steps": 50,
}
controlnet_args = {
    "module": "depth",
    "model": "control_sd15_depth [fef5e48e]",
}
ip = "127.0.0.1:7860"

# ... Txt2ImgModel definition ...
def save_base64_image(base64_data, output_path):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    image.save(output_path, "JPEG")

from PIL import Image, ImageDraw, ImageFont

def save_base64_images_to_grids(base64_data_list, output_path, labels, images_per_row = 4):
    images = [Image.open(BytesIO(base64.b64decode(base64_data))) for base64_data in base64_data_list]
    save_images_to_grids(images, output_path, labels, images_per_row)

def save_images_to_grids(images, output_path, labels, images_per_row = 4):
    # 计算拼接后的图片尺寸
    max_width = max([img.size[0] for img in images])
    max_height = max([img.size[1] for img in images])
    total_rows = (len(images) + images_per_row - 1) // images_per_row
    total_width = max_width * images_per_row
    total_height = max_height * total_rows

    # 创建一个空白画布，用于绘制拼接后的图片
    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # 设置字体和字体大小
    font = ImageFont.truetype("arial.ttf", size=20)
    draw = ImageDraw.Draw(canvas)

    # 将每张图片及其对应的标签放置到合适的位置
    for i, (image, label) in enumerate(zip(images, labels)):
        x = (i % images_per_row) * max_width
        y = (i // images_per_row) * max_height
        canvas.paste(image, (x, y))
        draw.text((x, y + max_height - 30), label, font=font, fill=(0, 0, 0))

    output_path = output_path + ".jpg"

    # 保存拼接后的图片
    canvas.save(output_path, "JPEG")

def merge_images_horizontally(base64_data_list, images_per_row, output_path):
    images = [Image.open(BytesIO(base64.b64decode(base64_data))) for base64_data in base64_data_list]
    # 获取每张图片的宽度和高度
    img_width, img_height = images[0].size

    # 计算合并后图片的总宽度
    total_width = img_width * images_per_row

    # 计算合并后图片的总高度
    total_height = img_height * ((len(images) - 1) // images_per_row + 1)

    # 创建一个新的空白图片，用于存放合并后的图片
    merged_image = Image.new("RGB", (total_width, total_height))

    # 遍历图片列表，将每张图片粘贴到新创建的空白图片中
    for index, image in enumerate(images):
        x = img_width * (index % images_per_row)
        y = img_height * (index // images_per_row)
        merged_image.paste(image, (x, y))

    output_path = output_path + ".jpg"
    # 保存合并后的图片
    merged_image.save(output_path)
    return merged_image


def get_models():
    url = f"http://{ip}/sdapi/v1/sd-models"
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:
        print("Connection refused")
        return []
    if response.status_code == 200:
        result = json.loads(response.text)
        models = [CheckpointModel(**model) for model in result]
        return models
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return []

async def txt2img_post_async(img_name, txt2img_model: Txt2ImgModel, output_folder: str):
    url = f"http://{ip}/sdapi/v1/txt2img"

    data = json.dumps(txt2img_model.dict(), default=lambda o: o.__dict__)

    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, data=data, headers=headers)

        if response.status_code == 200:
            result = json.loads(response.text)

            images = result["images"]
            os.makedirs(output_folder, exist_ok=True)
            save_name = file_util.get_new_file_name(output_folder, img_name, "jpg")
            for index, image_base64 in enumerate(images):
                image_path = os.path.join(output_folder, save_name)
                save_base64_image(image_base64, image_path)
                print(f"Image {index} saved at {image_path}")
                return image_base64
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")

async def img2img_post_async(img_name, img2img_model: Img2ImgModel, output_folder: str):
    url = f"http://{ip}/sdapi/v1/img2img"

    data = json.dumps(img2img_model.custom_to_dict(), default=lambda o: o.__dict__)

    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, data=data, headers=headers)

        if response.status_code == 200:
            result = json.loads(response.text)

            images = result["images"]
            os.makedirs(output_folder, exist_ok=True)
            save_name = file_util.get_new_file_name(output_folder, img_name, "jpg")
            for index, image_base64 in enumerate(images):
                image_path = os.path.join(output_folder, save_name)
                save_base64_image(image_base64, image_path)
                print(f"Image {index} saved at {image_path}")
                return image_base64
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")



async def submit_all(mention_label:gr.Label):
    submit_list = data_manager.submit_list
    for i in range(len(submit_list)):
        if submit_list[i].is_submit:
            times = submit_list[i].submit_times
            folder_name = submit_list[i].submit_folder
            save_path = template_utils.get_image_save_path(folder_name)
            image_save_path = os.path.join(save_path, "images")
            grids_save_path = os.path.join(save_path, "grids")
            file_util.check_folder(image_save_path)
            file_util.check_folder(grids_save_path)

            for j in range(times):
                images = []
                image_titles = []
                for submit_template in submit_list[i].submit_items:
                    if submit_template.is_submit:
                        sub_times = submit_template.submit_times
                        template_name = submit_template.submit_template
                        sub_images = []
                        for k in range(sub_times):
                            mention_label.update(value = f"正在提交{folder_name}\\{template_name}")

                            if submit_template.data.template_type  == "txt2img":
                                save_image_name = f"{template_name}_txt_{j}_{k}"
                                txt2img_model = submit_template.data.api_model
                                base_64 = await txt2img_post_async(save_image_name, txt2img_model, image_save_path)
                                sub_images.append(base_64)
                            elif submit_template.data.template_type  == "img2img":
                                save_image_name = f"{template_name}_img_{j}_{k}"
                                img2img_model = submit_template.data.api_model
                                base_64 = await img2img_post_async(save_image_name, img2img_model, image_save_path)
                                sub_images.append(base_64)
                            
                            mention_label.update(value = f"成功生成{folder_name}\\{template_name}!")
                        save_image_name = f"{template_name}_sub_grid_{j}"
                        if len(sub_images) > 1:
                            image = merge_images_horizontally(sub_images, 3, os.path.join(grids_save_path, save_image_name))
                            images.append(image)
                        elif len(sub_images) == 1:
                            image_data = base64.b64decode(sub_images[0])
                            image = Image.open(BytesIO(image_data))
                            images.append(image)
                        image_titles.append(template_name)
                if len(images) > 1:
                    save_image_name = f"{folder_name}_grid_{j}"
                    save_images_to_grids(images, os.path.join(grids_save_path, save_image_name), image_titles)
    return "处理完成"
