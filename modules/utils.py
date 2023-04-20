import os, sys, cv2
from base64 import b64encode
import requests
import PIL.Image as Image
import io
import base64
def setup_test_env():
    ext_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if ext_root not in sys.path:
        sys.path.append(ext_root)

#读取文件地址为图片并转换为base64返回
def image_path_to_base64(image_path):
    if not image_path:
        return image_path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_title_from_model_name(model_name: str) -> str:
        
    from modules.data_manager import checkpoints_models
    matching_titles = [model.title for model in checkpoints_models if model.model_name == model_name]

    if matching_titles:
        return next(iter(matching_titles))
    else:
        return None
    
def get_model_name_from_title(title: str) -> str:
    from modules.data_manager import checkpoints_models
    matching_model_name = [model.title for model in checkpoints_models if model.title == title]

    if matching_model_name:
        return next(iter(matching_model_name))
    else:
        return None

def image_to_base64(image:Image):
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        contents = output.getvalue()
        return base64.b64encode(contents).decode("utf-8")
    
def readImage(path):
    if path == "":
        return path
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img

def get_checkpoints():
    r = requests.get("http://localhost:7860/sdapi/v1/sd-models")
    result = r.json()
    for checkpoint in result:
        print(checkpoint)

def set_checkpoints(title):
    checkpoint_json = {
        "sd_model_checkpoint": title,
    }
    response = requests.post(url="http://localhost:7860/sdapi/v1/options", json=checkpoint_json)
    return response.json()

def get_controlnet_model():
    r = requests.get("http://localhost:7860/controlnet/model_list")
    result = r.json()
    if "model_list" in result:
        result = result["model_list"]
        for item in result:
            print("Using model: ", item)
            return item
    return "None"
