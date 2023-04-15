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

checkpoints_models = []

def refresh_ip():
    return ""

def refresh_templates_folders():
    global templates_folders
    templates_folders = template_utils.get_all_templates_folders()

def refresh_templates():
    global templates
    if choose_folder is None:
        templates = []
    else:
        templates = template_utils.get_templates_from_folder(choose_folder)

def get_txt2img_model(txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, restore_faces, tiling, batch_count, batch_size, cfg_scale, seed, height, width, eta, checkpoint_model):
    return Txt2ImgModel().create(prompt=txt2img_prompt, negative_prompt=txt2img_negative_prompt, 
                        steps=steps, sampler_index=sampler_index, restore_faces=restore_faces, 
                        tiling=tiling, n_iter=batch_count, batch_size=batch_size, cfg_scale=cfg_scale, 
                        seed=seed, height=height, width=width, checkpoint_model=checkpoint_model, eta=eta)


def load_parameter(template_path, name):
    if not template_path:
        template_path = template_utils.get_new_template_folder_name()
    #name none or empty
    if not name:
        name = template_utils.get_new_template_name(template_path)

    choose_folder = template_path
    base_data.template_name = name
    temp_data = template_utils.get_model_from_folder(choose_folder, name)
    base_data.options = temp_data.options
    base_data.type = temp_data.type
    if base_data.type == ApiType.img2img.value:
        base_data.api_model = temp_data.api_model
    elif base_data.type == ApiType.txt2img.value:
        base_data.api_model = temp_data.api_model
    return f"成功加载{choose_folder}文件夹下的{base_data.template_name}模板"

def save_parameter(template_path, name, options, type, *args):
    global choose_folder
    if not template_path:
        template_path = template_utils.get_new_template_folder_name()
    #name none or empty
    if not name:
        name = template_utils.get_new_template_name(template_path)

    choose_folder = template_path
    base_data.template_name = name
    base_data.options = options
    base_data.type = type
    if type == ApiType.img2img.value:
        base_data.api_model = Img2ImgModel(*args)
    elif base_data.type == ApiType.txt2img.value:
        base_data.api_model = get_txt2img_model(*args)
    template_utils.save_template_model(choose_folder, base_data)
    return f"成功存储在{choose_folder}文件夹下的{base_data.template_name}"
    


def list_files_with_name(filename):
    res = []

    dirpath = os.path.join(os.getcwd(), "css")
    path = os.path.join(dirpath, filename)
    if os.path.isfile(path):
        res.append(path)

    return res
