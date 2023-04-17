import gradio as gr
from modules.api_models import ApiType, TemplateBaseModel, SubmitFolderModel,SubmitItemModel, CheckpointModel
import modules.api_models as api_models
import modules.data_manager as data_manager
from ui_components import FormRow, ToolButton, FormGroup
import modules.template_utils as template_utils
import os
import modules.api_util as api_util
import time
txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
#获得model的参数并赋值给txt2ImgData or img2ImgData
def get_model_data(model_path):
    data:TemplateBaseModel = template_utils.get_model_from_template_path(model_path)
    data_manager.base_data = data
    if data.template_type == ApiType.img2img:
        data_manager.img_img_data = data.api_model
    elif data.template_type == ApiType.txt2img:
        data_manager.txt_img_data = data.api_model

def connect_paste(button, paste_fields, input_comp):
    def paste_func(prompt):
        base_data = template_utils.get_model_from_template(prompt)
        if base_data.template_type == ApiType.img2img:
            params = base_data.api_model
        elif base_data.template_type == ApiType.txt2img:
            params = base_data.api_model
        res = []

        for output, key in paste_fields:
            v = params.get(key, None)

            if v is None:
                res.append(gr.update())
            elif isinstance(v, type_of_gr_update):
                res.append(v)
            else:
                try:
                    valtype = type(output.value)

                    if valtype == bool and v == "False":
                        val = False
                    else:
                        val = valtype(v)

                    res.append(gr.update(value=val))
                except Exception:
                    res.append(gr.update())

        return res
    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
    )