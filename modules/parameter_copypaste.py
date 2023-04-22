import gradio as gr
from modules.api_models import ApiType
import modules.data_manager as data_manager
from modules.api_models import TemplateBaseModel, parse_string_to_img2img_model, parse_string_to_txt2img_model

def paste_func(paste_fields, template_model:TemplateBaseModel, folder = None, template_name = None):
    res = []

    if folder is not None:
        res.append(data_manager.choose_folder)
    if template_name is not None:
        res.append(data_manager.choose_template)
    if template_model is None:
        for component, key in paste_fields:
            res.append(None)
        return res

    data_manager.base_data = template_model
    if template_model.template_type == ApiType.img2img.value:
        params = template_model.api_model
        data_manager.img_img_data = template_model.api_model
    elif template_model.template_type == ApiType.txt2img.value:
        params = template_model.api_model
        data_manager.txt_img_data = template_model.api_model


    for component, key in paste_fields:
        v = params.get_attribute_value(key)

        if v is None:
            res.append(gr.update())
        # elif isinstance(v, type_of_gr_update):
        #     res.append(v)
        else:
            try:
                valtype = type(component.value)

                if valtype == bool and v == "False":
                    val = False
                else:
                    val = valtype(v)

                res.append(gr.update(value=val))
            except Exception:
                #if component type is pil
                ty = type(component)
                ishas = hasattr(component, 'pil')
                if ty == gr.components.Image and ishas:
                    res.append(v)
                else:
                    res.append(gr.update())

    return res
    

def connect_paste(button, paste_fields, info_tex, folder, template_name):
    def f():
        pre_choose_template = data_manager.pre_choose_template
        if pre_choose_template is None:
            return ["未选择模板！选择模板后，先点击查看模板信息才能加载"] + paste_func(paste_fields, pre_choose_template, folder, template_name)
        else:
            return ["加载模板成功！"] + paste_func(paste_fields, pre_choose_template, folder, template_name)
        
    button.click(
        fn=f,
        inputs=[],
        outputs=[info_tex, folder, template_name] + [x[0] for x in paste_fields],
    )

def connect_paste_with_text(button, paste_fields, input_text, is_txt2img):
    def f(input_text):
        if is_txt2img:
            data_manager.txt_img_data = parse_string_to_txt2img_model(input_text)
            data_manager.base_data.api_model = data_manager.txt_img_data
            data_manager.base_data.template_type = ApiType.txt2img.value

        else:
            data_manager.img_img_data = parse_string_to_img2img_model(input_text)
            data_manager.base_data.api_model = data_manager.img_img_data
            data_manager.base_data.template_type = ApiType.img2img.value

        pre_choose_template = data_manager.base_data
        data_manager.pre_choose_template = pre_choose_template
        if pre_choose_template is None:
            return "未选择模板！"
        else:
            return paste_func(paste_fields, pre_choose_template)
        
    button.click(
        fn=f,
        inputs=[input_text],
        outputs=[] + [x[0] for x in paste_fields],
    )