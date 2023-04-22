from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Union, Any
import requests
import modules.utils as utils
import re

#give a enum for txt and img
class ApiType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"

class CheckpointModel(BaseModel):
    title: str
    model_name: str
        # "title": "AsiaFacemix-pruned-fix.safetensors [75230c2f99]",
        # "model_name": "AsiaFacemix-pruned-fix",

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

    def get_attribute_value(self, attribute_name):
        if attribute_name == "sd_model_checkpoint":
            title = self.override_settings.get(attribute_name)
            model_name = utils.get_model_name_from_title(title)
            return model_name if model_name else None
        return getattr(self, attribute_name)
    
    def create(self, prompt="", negative_prompt="", checkpoint_model="", seed=-1, batch_size=1, n_iter=1, steps=50, cfg_scale=7, width=512, height=512, restore_faces=False, tiling=False, eta=0, sampler_index="Euler a"):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index)
        self.set_override_settings(checkpoint_model)
        return self

    def set_override_settings(self, model):
        self.override_settings={}
        title = utils.get_title_from_model_name(model)
        self.override_settings['sd_model_checkpoint'] = title if title else None

    def get_checkpoint_model(self):
        return self.override_settings['sd_model_checkpoint'] if 'sd_model_checkpoint' in self.override_settings else None

class resize_mode(Enum):
    img2img = 1
    inpaint = 2
    inpaint_sketch = 3
    inpaint_upload_mask = 4


inpainting_fill_choices = ['fill', 'original', 'latent noise', 'latent nothing']
inpainting_mask_invert_choices = ['Inpaint masked', 'Inpaint not masked']
resize_mode_choices = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
inpaint_full_res_choices = ['Whole picture', 'Only masked']
controlnet_resize_mode = ["Just Resize", "Inner Fit (Scale to Fit)", "Outer Fit (Shrink to Fit)"]
 # 0 img2img 1 img2img 2 inpaint 3 inpaint sketch 4 inpaint upload mask 暂时我们只需要4和1
class Img2ImgModel(Txt2ImgModel):
# 上面的所有参数写出来
    init_images: List[str] = None #img2img 基础的图都在里面： 文件地址
    mask:str = None # 文件地址
    resize_mode: int = 1#["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
    denoising_strength: float = 0.72
    mask_blur:int = 0 #蒙版模糊 4
    inpainting_fill:int = 0# 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
    inpaint_full_res:int = 0 # inpaint area 0 whole picture 1：only masked
    inpaint_full_res_padding:int = 32 # Only masked padding, pixels 32
    inpainting_mask_invert:int = 0 # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容
    
    # image_cfg_scale: float =  0.72
    # init_latent = None
    # image_mask = ""
    # latent_mask = None
    # mask_for_overlay = None
    # nmask = None
    # image_conditioning = None
    alwayson_scripts: Dict[str, Dict[str, Any]] = {}

    def get_attribute_value(self, attribute_name):
        if attribute_name == "init_images":
            return self.init_images[0] if self.init_images else None
        elif attribute_name == "inpainting_fill":
            return inpainting_fill_choices[self.inpainting_fill]
        elif attribute_name == "inpainting_mask_invert":
            return inpainting_mask_invert_choices[self.inpainting_mask_invert]
        elif attribute_name == "inpaint_full_res":
            return inpaint_full_res_choices[self.inpaint_full_res]
        elif attribute_name == "resize_mode":
            return resize_mode_choices[self.resize_mode]
        #attribute_name包含control_的话，就是控制模块的参数
        elif attribute_name.startswith("control_"):
            key = attribute_name.replace("control_", "")
            args = self.get_controlnet_params()

            if key == "resize_mode":
                return controlnet_resize_mode[args[key]]
            return args[key] if key in args else None
        return super().get_attribute_value(attribute_name)
    
    def get_init_image(self):
        return self.init_images[0] if self.init_images else None
    
    def create(self, prompt="", negative_prompt="", checkpoint_model="", seed=0, batch_size=1, n_iter=0, steps=1000, cfg_scale=0.72, width=0, height=0, restore_faces=False, tiling=False, eta=0.1, sampler_index=0, inpaint_full_res=0, inpaint_full_res_padding=32, init_image="", init_mask="", mask_blur=4, inpainting_fill=0, inpainting_mask_invert=0,resize_mode=1, denoising_strength=0.72,
                control_enabled=False, control_module="", control_model="", control_weight=0.0, control_image="", control_mask="", control_invert_image=False, control_resize_mode=0, control_rgbbgr_mode=0, control_lowvram=False, control_processor_res=0, control_threshold_a=0.0, control_threshold_b=0.0, control_guidance_start=0, control_guidance_end=0, control_guessmode=0):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index,resize_mode=resize_mode, denoising_strength=denoising_strength)
        self.set_override_settings(checkpoint_model)
        self.setup_img2img_params(init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert, inpaint_full_res, inpaint_full_res_padding)
        self.setup_controlnet_params(control_enabled, control_module, control_model, control_weight, control_image, control_mask, control_invert_image, control_resize_mode, control_rgbbgr_mode, control_lowvram, control_processor_res, control_threshold_a, control_threshold_b, control_guidance_start, control_guidance_end, control_guessmode)
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

    def setup_controlnet_params(self, enabled, module, model, weight, image, mask, invert_image, resize_mode, rgbbgr_mode, lowvram, processor_res, threshold_a, threshold_b, guidance_start, guidance_end, guessmode):
        controlnet_args = {
            "enabled": enabled,
            "module": module,
            "model": model,
            "weight": weight,
            "image": image,
            "mask": mask,
            "invert_image": invert_image,
            "resize_mode": resize_mode,
            "rgbbgr_mode": rgbbgr_mode,
            "lowvram": lowvram,
            "processor_res": processor_res,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
            "guessmode": guessmode
        }
        self.alwayson_scripts["ControlNet"] = {
            "args": controlnet_args
        }

    def get_controlnet_params(self):
        return self.alwayson_scripts["ControlNet"]["args"] if "ControlNet" in self.alwayson_scripts else ControlNet_Model()

    def custom_to_dict(self):
        res_dict = self.dict()
        init_image = utils.image_path_to_base64(self.init_images[0])
        res_dict["init_images"][0] = init_image
        res_dict["mask"] = utils.image_path_to_base64(self.mask)
        controlnet_args:ControlNet_Model = self.get_controlnet_params()
        if controlnet_args:
            res_dict["alwayson_scripts"]["ControlNet"]["args"]["image"] = utils.image_path_to_base64(controlnet_args["image"])
            res_dict["alwayson_scripts"]["ControlNet"]["args"]["mask"] = utils.image_path_to_base64(controlnet_args["mask"])
        return res_dict

def parse_string_to_img2img_model(s: str) -> Img2ImgModel:
    def get_int_value(pattern: str, default: int = 0):
        match = re.search(pattern, s)
        return int(match.group(1)) if match else default

    def get_float_value(pattern: str, default: float = 0.0):
        match = re.search(pattern, s)
        return float(match.group(1)) if match else default

    def get_string_value(pattern: str, default: str = ''):
        match = re.search(pattern, s)
        return match.group(1) if match else default
    *lines, lastline = s.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''
    done_with_prompt = False
    prompt = ""
    negative_prompt = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line
    steps = get_int_value(r'Steps: (\d+),')
    sampler = get_string_value(r'Sampler: ([^,]+),', '')
    cfg_scale = get_int_value(r'CFG scale: (\d+),')
    seed = get_int_value(r'Seed: (\d+),')
    restoration = get_string_value(r'Face restoration: ([^,]+),', '')
    size = re.search(r'Size: (\d+)x(\d+),', s)
    width, height = int(size.group(1)), int(size.group(2)) if size else (512, 512)
    model_hash = get_string_value(r'Model hash: ([^,]+),', '')
    denoising_strength = get_float_value(r'Denoising strength: ([^,]+),')
    mask_blur = get_int_value(r'Mask blur: (\d+),')

    checkpoint_model = utils.get_model_name_from_hash(model_hash) 

    return Img2ImgModel().create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        sampler_index=sampler,
        restore_faces=True if restoration else False,
        checkpoint_model=checkpoint_model,
        denoising_strength=denoising_strength,
        mask_blur=mask_blur
    )
    
re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
def parse_string_to_txt2img_model(s: str) -> Txt2ImgModel:
    def get_int_value(pattern: str, default: int = 0):
        match = re.search(pattern, s)
        return int(match.group(1)) if match else default

    def get_float_value(pattern: str, default: float = 0.0):
        match = re.search(pattern, s)
        return float(match.group(1)) if match else default

    def get_string_value(pattern: str, default: str = ''):
        match = re.search(pattern, s)
        return match.group(1) if match else default

    *lines, lastline = s.strip().split("\n")
    prompt = ""
    negative_prompt = ""
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''
    done_with_prompt = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    steps = get_int_value(r'Steps: (\d+),')
    sampler = get_string_value(r'Sampler: ([^,]+),', '')
    cfg_scale = get_int_value(r'CFG scale: (\d+),')
    seed = get_int_value(r'Seed: (\d+),')
    restoration = get_string_value(r'Face restoration: ([^,]+),', '')
    size = re.search(r'Size: (\d+)x(\d+),', s)
    width, height = int(size.group(1)), int(size.group(2)) if size else (512, 512)
    model_hash = get_string_value(r'Model hash: ([^,]+),', '')

    checkpoint_model = utils.get_model_name_from_hash(model_hash) 
    return Txt2ImgModel().create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        sampler_index=sampler,
        restore_faces=True if restoration else False,
        checkpoint_model=checkpoint_model,
    )


class ControlNet_Model(BaseModel):
    enabled: bool = True
    module: str = controlnet_modules.openpose.value
    model: str = "control_openpose-fp16 [9ca67cc5]"
    weight: float = 1.0
    image: str = None
    mask: str = None
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
    submit_folder: str = ""

class SubmitFolderModel(BaseModel):
    is_submit: bool = True
    submit_folder: str = ""
    submit_times: int = 1
    submit_items: List[SubmitItemModel] = []