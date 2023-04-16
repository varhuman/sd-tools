from modules.api_models import TemplateBaseModel, Txt2ImgModel, Img2ImgModel, ApiType, SubmitFolderModel, SubmitItemModel
import modules.template_utils as template_utils
import os
from gradio import Blocks
import PIL.Image as Image
import modules.utils as utils

ip = "127.0.0.1:7860"
demo:Blocks = None
txt_img_data: Txt2ImgModel = Txt2ImgModel()
img_img_data: Img2ImgModel = Img2ImgModel()
base_data: TemplateBaseModel = TemplateBaseModel()
templates_folders = []
templates = []
choose_template = None
choose_folder = None
submit_list: list[SubmitFolderModel] = []

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

def refresh_submit_list():
    global submit_list
    temp_list = []
    folders = template_utils.get_all_templates_folders()
    for folder in folders:
        templates = template_utils.get_templates_from_folder(folder)
        if not templates:
            continue
        new_submit_folder = SubmitFolderModel()
        new_submit_folder.submit_folder = folder
        new_submit_folder.submit_times = 1
        new_submit_folder.is_submit = True
        new_submit_folder.submit_items = []

        copy_submit_folder = None
        for item in submit_list:
            if item.submit_folder == new_submit_folder.submit_folder:
                new_submit_folder.is_submit = item.is_submit
                new_submit_folder.submit_times = item.submit_times
                copy_submit_folder = item
                break
        
        for template in templates:
            submit_item = SubmitItemModel()
            submit_item.is_submit = True
            submit_item.submit_template = template
            temp_data = template_utils.get_model_from_folder(folder, template)
            submit_item.data = temp_data
            new_submit_folder.submit_items.append(submit_item)

            if copy_submit_folder is not None:
                for copy_item in copy_submit_folder.submit_items:
                    if copy_item.submit_template == submit_item.submit_template:
                        submit_item.is_submit = copy_item.is_submit
                        submit_item.submit_times = copy_item.submit_times
                        break
        temp_list.append(new_submit_folder)
    
    submit_list = temp_list


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

def get_img2img_model(img2img_prompt, img2img_negative_prompt, restore_faces, tiling, seed, sampler_index, steps, cfg_scale, width, height, batch_size, batch_count, eta, inpaint_full_res, inpaint_full_res_padding, checkpoint_model, img_inpaint:Image, mask_inpaint:Image, mask_blur, inpainting_fill, inpainting_mask_invert):
    # to base64
    if img_inpaint is not None:
        init_image = [utils.image_to_base64(img_inpaint)]
    if mask_inpaint is not None:
        init_mask = utils.image_to_base64(mask_inpaint)
    else:
        init_mask = ""
    return Img2ImgModel().create(prompt=img2img_prompt, negative_prompt=img2img_negative_prompt, 
                        restore_faces=restore_faces, tiling=tiling, seed=seed, sampler_index=sampler_index, 
                        steps=steps, cfg_scale=cfg_scale, width=width, height=height, batch_size=batch_size, 
                        n_iter=batch_count, eta=eta, inpaint_full_res=inpaint_full_res, 
                        inpaint_full_res_padding=inpaint_full_res_padding, checkpoint_model=checkpoint_model, 
                        init_image=init_image, init_mask=init_mask, mask_blur=mask_blur, 
                        inpainting_fill=inpainting_fill, inpainting_mask_invert=inpainting_mask_invert)

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
    base_data.template_type = temp_data.template_type
    base_data.api_model = temp_data.api_model

    if base_data.template_type == ApiType.img2img.value:
        img_img_data = temp_data.api_model
    elif base_data.template_type == ApiType.txt2img.value:
        txt_img_data = temp_data.api_model
    return f"成功加载{choose_folder}文件夹下的{base_data.template_name}模板"

def save_parameter(template_path, name, options, template_type_label, *args):
    global choose_folder
    if not template_path:
        template_path = template_utils.get_new_template_folder_name()
    #name none or empty
    if not name:
        name = template_utils.get_new_template_name(template_path)

    choose_folder = template_path
    base_data.template_name = name
    base_data.options = options
    base_data.template_type = template_type_label['label']
    if base_data.template_type == ApiType.txt2img.value:
        base_data.api_model = get_txt2img_model(*args)
    elif base_data.template_type == ApiType.img2img.value:
        base_data.api_model = get_img2img_model(*args)
    template_utils.save_template_model(choose_folder, base_data)
    return f"成功存储在{choose_folder}文件夹下的{base_data.template_name}"

def list_files_with_name(filename):
    res = []

    dirpath = os.path.join(os.getcwd(), "css")
    path = os.path.join(dirpath, filename)
    if os.path.isfile(path):
        res.append(path)

    return res