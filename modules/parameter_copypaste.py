import gradio as gr
from modules.api_models import ApiType
import modules.data_manager as data_manager

def connect_paste(button, paste_fields):
    def paste_func():
        base_data, t2i_data, i2i_data = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data
        pre_choose_template = data_manager.pre_choose_template
        if pre_choose_template is None:
            return "未选择模板！"
        base_data = pre_choose_template

        if pre_choose_template.template_type == ApiType.img2img.value:
            params = pre_choose_template.api_model
            i2i_data = pre_choose_template.api_model
        elif pre_choose_template.template_type == ApiType.txt2img.value:
            params = pre_choose_template.api_model
            t2i_data = pre_choose_template.api_model
        res = []


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
                    if ty == gr.components.Image:
                        res.append(v)
                    else:
                        res.append(gr.update())

        return res
    button.click(

        fn=paste_func,
        inputs=[],
        outputs=[] + [x[0] for x in paste_fields],
    )