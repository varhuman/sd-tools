from modules.api_models import TemplateBaseModel, Txt2ImgModel, Img2ImgModel, ApiType
import modules.template_utils as template_utils
import os

ip = "127.0.0.1:7860"
demo = None
txt_img_data: Txt2ImgModel = Txt2ImgModel()
img_img_data: Img2ImgModel = Img2ImgModel()
base_data: TemplateBaseModel = TemplateBaseModel()
templates_folders = []
templates = []
choose_template = None
choose_folder = None

samplers_k_diffusion = [
    'Euler a',
    'DPM++ 2M',
    'DPM++ 2S a Karras',
    'DPM++ 2M Karras',
    'DPM++ SDE Karras',
]

def refresh_ip(ip):
    return ""

def refresh_templates_folders():
    templates_folders = template_utils.get_all_templates_folders()

def refresh_templates():
    if choose_folder is None:
        templates = []
    else:
        templates = template_utils.get_templates_from_folder(choose_folder)

def get_txt2img_model(txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, restore_faces, tiling, batch_count, batch_size, cfg_scale, seed, height, width, override_settings, eta):
    return Txt2ImgModel(prompt=txt2img_prompt, negative_prompt=txt2img_negative_prompt, steps=steps, sampler_index=sampler_index, restore_faces=restore_faces, tiling=tiling, n_iter=batch_count, batch_size=batch_size, cfg_scale=cfg_scale, seed=seed, height=height, width=width, override_settings=override_settings, eta=eta)


# name: str = "default"
#     api_model: Any = None # txt2img or img2img
#     options: str = "default"
#     type: ApiType = ApiType.txt2img
def save_parameter(template_path:str, name:str, options:str, type:ApiType, **args):
    base_data.name = name
    base_data.options = options
    base_data.type = type
    if type == ApiType.img2img:
        base_data.api_model = Img2ImgModel(**args)
    elif base_data.type == ApiType.txt2img:
        base_data.api_model = get_txt2img_model(**args)
    template_utils.save_template_model(template_path, base_data)

def list_files_with_name(filename):
    res = []
    
    dirpath = os.path.join(os.getcwd(), "css")
    path = os.path.join(dirpath, filename)
    if os.path.isfile(path):
        res.append(path)

    return res
