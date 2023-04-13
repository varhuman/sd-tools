import os, sys, cv2
from base64 import b64encode
import requests

class utils():
    def setup_test_env():
        ext_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if ext_root not in sys.path:
            sys.path.append(ext_root)

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

utils.setup_test_env()
