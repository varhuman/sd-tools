from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Union
import requests
from utils import utils
#give a enum for txt and img
class ApiType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"

class apiTypeModel(BaseModel):
    name: str
    value: str
    type: ApiType


class Txt2ImgModel(BaseModel):
    enable_hr: bool = False
    denoising_strength: int = 0
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: int = 2
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    prompt: str = "A Girl, laying sofa"
    styles: List[str] = ["string"]
    override_settings: Dict[str, Union[str, int]] = {"sd_model_checkpoint": "wlop-any.ckpt [7331f3bc87]"}
    seed: int = -1
    subseed: int = -1
    subseed_strength: int = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: int = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
    eta: int = 0
    s_churn: int = 0
    s_tmax: int = 0
    s_tmin: int = 0
    s_noise: int = 1
    override_settings_restore_afterwards: bool = True
    script_args: List[str] = []
    sampler_index: str = "Euler"


class ApiBase():
    def setup_route(self, url, setup_args, controlnet_args):
        self.url = url
        self.body = {
            "enable_hr": False,
            "denoising_strength": 0,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "prompt": "",
            "init_images": "",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 45,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "restore_faces": False,
            "tiling": False,
            "negative_prompt": "",
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "sampler_index": "Euler a",
            "alwayson_scripts": {}
        }
        self.setup_params(setup_args)
        self.setup_controlnet_params(controlnet_args)

    def setup_params(self, setup_args):
        for k, v in setup_args.items():
            self.body[k] = v

    def setup_controlnet_params(self, controlnet_args):
        self.body["alwayson_scripts"]["ControlNet"] = {
            "args": controlnet_args
        }

    def send_request(self):
        r = requests.post(self.url, json=self.body)
        return r.json()

class ControlNet_Api(ApiBase):
    def setup(self, url, image, mask, setup_args, cn_params):
        controlnet_unit = {
            "enabled": True,
            "module": "canny",
            "model": "control_sd15_canny [fef5e48e]",
            "weight": 1.0,
            "image": utils.readImage(image),
            "mask": utils.readImage(mask),
            "invert_image": False,
            "resize_mode": 1,
            "rgbbgr_mode": False,
            "lowvram": False,
            "processor_res": 512,
            "threshold_a": 64,
            "threshold_b": 64,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "guessmode": False,
        }
        for k, v in cn_params.items():
            controlnet_unit[k] = v

        controlnet_args = [
            controlnet_unit
        ]
        self.setup_route(url, setup_args, controlnet_args)

class Txt2Img(ControlNet_Api):
    def setup(self, image, mask, setup_args, cn_params):
        url_txt2img = "http://localhost:7860/sdapi/v1/txt2img"
        return super().setup(url_txt2img, image, mask, setup_args, cn_params)
    
class Img2Img(ControlNet_Api):
    def setup(self, image, mask, setup_args, cn_params):
        url_img2img = "http://localhost:7860/sdapi/v1/img2img"
        return super().setup(url_img2img, image, mask, setup_args, cn_params)
    
def setType():
    # global api_type
    api_type = ["txt2img", "img2img"]
        

setType()