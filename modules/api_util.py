import modules.utils as api
from modules.api_models import Txt2ImgModel, Img2ImgModel,CheckpointModel
import modules.template_utils as template_utils
import modules.data_manager as data_manager
import gradio as gr
import modules.file_util as file_util
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


import io, base64
import matplotlib.pyplot as plt
from PIL import Image

def test_show_image(js):
    pil_img = Image.open(input_image_url)
    image = Image.open(io.BytesIO(base64.b64decode(js["images"][0])))
    mask_image = Image.open(io.BytesIO(base64.b64decode(js["images"][1])))

    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(pil_img)   
    axarr[1].imshow(image)
    axarr[2].imshow(mask_image)
    plt.show()

#api.utils.set_checkpoints(modeltitle)


import base64
import json
import os
from io import BytesIO
from PIL import Image
import requests

ip = "127.0.0.1:7860"

# ... Txt2ImgModel definition ...
def save_base64_image(base64_data, output_path):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    image.save(output_path, "JPEG")

def get_models():
    url = f"http://{ip}/sdapi/v1/sd-models"
    response = requests.get(url)
    if response.status_code == 200:
        result = json.loads(response.text)
        models = [CheckpointModel(**model) for model in result]
        return models
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return []

def txt2img_post(img_name, txt2img_model: Txt2ImgModel, output_folder: str):
    url = f"http://{ip}/sdapi/v1/txt2img"

    data = json.dumps(txt2img_model.dict(), default=lambda o: o.__dict__)

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=data, headers=headers)

    if response.status_code == 200:
        result = json.loads(response.text)

        images = result["images"]
        os.makedirs(output_folder, exist_ok=True)
        save_name = file_util.get_new_file_name(output_folder, img_name, "jpg")
        for index, image_base64 in enumerate(images):
            image_path = os.path.join(output_folder, save_name)
            save_base64_image(image_base64, image_path)
            print(f"Image {index} saved at {image_path}")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

def img2img_post(img_name, img2img_model: Img2ImgModel, output_folder: str):
    url = f"http://{ip}/sdapi/v1/img2img"

    data = json.dumps(img2img_model.dict(), default=lambda o: o.__dict__)

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=data, headers=headers)

    if response.status_code == 200:
        result = json.loads(response.text)

        images = result["images"]
        os.makedirs(output_folder, exist_ok=True)
        save_name = file_util.get_new_file_name(output_folder, img_name, "jpg")
        for index, image_base64 in enumerate(images):
            image_path = os.path.join(output_folder, save_name)
            save_base64_image(image_base64, image_path)
            print(f"Image {index} saved at {image_path}")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

def submit_all(mention_label:gr.Label):
    submit_list = data_manager.submit_list
    # for i in range(len(submit_list)):
    #     if submit_list[i].is_submit:
    #         for submit_template in submit_list[i].submit_items:
    #             if submit_template.is_submit:
    #                 if submit_template.data.template_type  == "txt2img":
    #                     txt2img_model = submit_template.data.api_model
    #                     txt2img_post(f"{submit_template.submit_template}_output", txt2img_model, "output")
    #                     mention_label.update(value = f"成功提交{submit_list[i].submit_folder}\\{submit_template.submit_template}!")
    #                 elif submit_template.data.template_type  == "img2img":
    #                     img2img_model = submit_template.data.api_model
    #                     img2img_post(f"{submit_template.submit_template}_output", img2img_model, "output")
    #                     mention_label.update(value = f"img成功提交{submit_list[i].submit_folder}\\{submit_template.submit_template}!")
