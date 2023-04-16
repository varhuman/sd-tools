from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Union, Any
import requests
import modules.utils as utils
#give a enum for txt and img
class ApiType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"

class controlnet_modules(Enum):
    none = "none"
    canny = "canny"
    depth = "depth"
    depth_leres = "depth_leres"
    fake_scribble = "fake_scribble"
    hed = "hed"
    mlsd = "mlsd"
    normal_map = "normal_map"
    openpose = "openpose"
    segmentation = "segmentation"
    binary = "binary"
    color = "color"

def to_serializable(obj: Any):
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj.__dict__


class Txt2ImgModel(BaseModel):
    prompt: str = "A Girl, laying sofa"
    negative_prompt: str = ""
    override_settings: Dict[str, Union[str, int]] = {"sd_model_checkpoint": "wlop-any.ckpt [7331f3bc87]"}
    seed: int = -1
    batch_size: int = 1 #每次张数
    n_iter: int = 1 # 生成批次
    steps: int = 50
    cfg_scale: int = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
    eta: int = 0
    script_args: List[str] = []
    sampler_index: str = "Euler a"


    def create(self, prompt, negative_prompt, checkpoint_model, seed, batch_size, n_iter, steps, cfg_scale, width, height, restore_faces, tiling, eta, sampler_index):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index)
        self.set_override_settings(checkpoint_model)
        return self

    def set_override_settings(self, model):
        self.override_settings={}
        self.override_settings['sd_model_checkpoint'] = model

    def get_checkpoint_model(self):
        return self.override_settings['sd_model_checkpoint'] if 'sd_model_checkpoint' in self.override_settings else None

class resize_mode(Enum):
    img2img = 1
    inpaint = 2
    inpaint_sketch = 3
    inpaint_upload_mask = 4

 # 0 img2img 1 img2img 2 inpaint 3 inpaint sketch 4 inpaint upload mask 暂时我们只需要4和1
class Img2ImgModel(Txt2ImgModel):
# 上面的所有参数写出来
    init_images: List[str] = None #img2img 基础的图都在里面： base64
    mask:str = None # base64
    resize_mode: int = 1#["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
    denoising_strength: float = 0.72
    # image_cfg_scale: float =  0.72
    # init_latent = None
    # image_mask = ""
    # latent_mask = None
    # mask_for_overlay = None
    mask_blur:float = 0.0 #蒙版模糊 4
    inpainting_fill:float = 0.0# 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
    inpaint_full_res:bool = False # inpaint area 0 whole picture 1：only masked
    inpaint_full_res_padding:int = 32 # Only masked padding, pixels 32
    inpainting_mask_invert:bool = False # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容
    # nmask = None
    # image_conditioning = None
    alwayson_scripts: Dict[str, Dict[str, str]] = {}

    def create(self, prompt, negative_prompt, checkpoint_model, seed, batch_size, n_iter, steps, cfg_scale, width, height, restore_faces, tiling, eta, sampler_index, inpaint_full_res, inpaint_full_res_padding, init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index)
        self.set_override_settings(checkpoint_model)
        self.setup_img2img_params(init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert, inpaint_full_res, inpaint_full_res_padding)
        self.setup_controlnet_params()
        return self
    
    def setup_img2img_params(self, init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert, inpaint_full_res, inpaint_full_res_padding):
        self.init_images = init_image
        self.mask = init_mask
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert

    def set_override_settings(self, model):
        self.override_settings={}
        self.override_settings['sd_model_checkpoint'] = model

    def get_checkpoint_model(self):
        return self.override_settings['sd_model_checkpoint'] if 'sd_model_checkpoint' in self.override_settings else None

    def setup_controlnet_params(self, controlnet_args = None):
        if controlnet_args is None:
            controlnet = ControlNet_Model()
            controlnet_args = controlnet.json()
        self.alwayson_scripts["ControlNet"] = {
            "args": controlnet_args
        }

class ControlNet_Model(BaseModel):
    enabled: bool = True
    module: str = controlnet_modules.openpose.value
    model: str = "control_openpose-fp16 [9ca67cc5]"
    weight: float = 1.0
    image: str = "./1.png"
    mask: str = ""
    invert_image: bool = False
    resize_mode: int = 1
    rgbbgr_mode: bool = False
    lowvram: bool = False
    processor_res: int = 512
    threshold_a: int = 64
    threshold_b: int = 64
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    guessmode: bool = False


class TemplateBaseModel(BaseModel):
    template_name: str = ""
    api_model: Txt2ImgModel = None # txt2img or img2img
    options: str = "default"
    template_type: str = ApiType.txt2img.value
    def __init__(self, **data: Any):
        super().__init__(**data)
        # if data == {} return
        if data == {}:
            return
        if self.template_type == ApiType.txt2img.value:
            self.api_model = Txt2ImgModel(**data["api_model"])
        elif self.template_type == ApiType.img2img.value:
            self.api_model = Img2ImgModel(**data["api_model"])

class SubmitItemModel(BaseModel):
    submit_template: str = ""
    data: TemplateBaseModel = None
    submit_times: int = 1
    is_submit: bool = True

class SubmitFolderModel(BaseModel):
    is_submit: bool = True
    submit_folder: str = ""
    submit_times: int = 1
    submit_items: List[SubmitItemModel] = []
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
    
