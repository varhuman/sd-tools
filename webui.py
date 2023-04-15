import gradio as gr
from modules.api_models import ApiType, TemplateBaseModel
import modules.api_models as api_models
import modules.data_manager as data_manager
from ui_components import FormRow, ToolButton, FormGroup
import modules.template_utils as template_utils
import os

refresh_symbol = '\U0001f504'  # 🔄
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


#获得model的参数并赋值给txt2ImgData or img2ImgData
def get_model_data(model_path):
    data:TemplateBaseModel = template_utils.get_template_model(model_path)
    api_models.base_data = data
    if data.type == ApiType.img2img:
        api_models.img_img_data = data.api_model
    elif data.type == ApiType.txt2img:
        api_models.txt_img_data = data.api_model

def update_tabs(input):
    return f"update_tabs: {input}"

def create_txt2img_ui():
    base_data, t2i_data, i2i_data, samplers = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data, data_manager.samplers_k_diffusion

    with gr.Blocks() as txt2img_bolcks:
        with gr.Row():
            with gr.Column(variant='compact'):
                with FormRow(elem_id="txt2img row1"):
                    checkpoint_model = gr.Dropdown(label='Model', elem_id="txt2img_checkpoint_model", choices=data_manager.checkpoints_models, value=t2i_data.get_checkpoint_model())
                with FormRow(elem_id="txt2img row1"):
                    txt2img_prompt = gr.Textbox(label="prompt", elem_id="txt2img_prompt", value=t2i_data.prompt)
                with FormRow(elem_id="txt2img row2"):
                    txt2img_negative_prompt = gr.Textbox(label="negative_prompt", elem_id="txt2img_negative_prompt", value=t2i_data.negative_prompt)
                with FormRow(elem_id="txt2img row3"):
                    restore_faces = gr.Checkbox(label="restore_faces", elem_id="txt2img_restore_faces", value=t2i_data.restore_faces)
                    tiling = gr.Checkbox(label="tiling", elem_id="txt2img_tiling", value=t2i_data.tiling)
                    seed = gr.Number(label='Seed', value= -1 , elem_id = 'txt2img_seed')
                    sampler_index = gr.Dropdown(label='Sampling method', elem_id="txt2img_sampling", choices=samplers, value=t2i_data.sampler_index, type="index")
                    
            with gr.Column(variant='compact'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id="txt2img_steps", label="Sampling steps", value=t2i_data.steps)
                cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.5, elem_id="txt2img_cfg_scale", label="cfg_scale", value=t2i_data.cfg_scale)
                width = gr.Slider(minimum=64, maximum=1024, step=8, elem_id="txt2img_width", label="width", value=t2i_data.width)
                height = gr.Slider(minimum=64, maximum=1024, step=8, elem_id="txt2img_height", label="height", value=t2i_data.height)
                batch_size = gr.Slider(minimum=1, maximum=1024, step=1, elem_id="txt2img_batch_size", label="batch_size", value=t2i_data.batch_size)
                batch_count = gr.Slider(minimum=1, maximum=100, step=1, elem_id="txt2img_n_iter", label="n_iter", value=t2i_data.n_iter)
                eta = gr.Slider(minimum=0, maximum=10, step=1, elem_id="txt2img_eta", label="eta", value=t2i_data.eta)
                # t2i_data.script_args = gr.Textbox(label="script_args", elem_id="txt2img_script_args", value=t2i_data.script_args) # not use for now
        txt2img_args = [
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    height,
                    width,
                    eta,
                    checkpoint_model,
                ]
        
    return txt2img_bolcks, txt2img_args

def create_ui():
    base_data, t2i_data, i2i_data, samplers = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data, data_manager.samplers_k_diffusion
    css = ""

    for cssfile in data_manager.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        with open(cssfile, "r", encoding="utf8") as file:
            css += file.read() + "\n"

    with gr.Blocks() as img2img_interface:
        with gr.Row():
            with gr.Column(variant='compact', elem_id="img2img"):
                with FormRow(elem_id=f"Enable HR"):
                    # sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
                    # steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
                    seed = gr.Textbox(label='Seed2', value= -1 , elem_id = 'img2img_seed')

    txt2img_interface, txt2img_args = create_txt2img_ui()
    interfaces = [
        (txt2img_interface, "txt2img", "txt2img", txt2img_args),
        (img2img_interface, "img2img", "img2img", None),
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        # parameter_type = gr.Dropdown(label='type', elem_id="parameter_type", on_change=update_parameter_display, choices=api_type, value=api_type[0])
        #一个可以输入ip地址得文本框，并加入一个点击按钮
        with FormGroup():
            with FormRow():
                ip_text = gr.Textbox(label='ip', value= data_manager.ip , elem_id = 'connect_ip')
                connect_ip = gr.Button('connect', elem_id = 'connect_ip')
                all_templates_folders = gr.Dropdown(label='所有模板文件夹', elem_id="all_templates_folders", choices=["None"] + list(data_manager.templates_folders), value=data_manager.choose_folder)
                create_refresh_button(all_templates_folders, data_manager.refresh_templates_folders, lambda: {"choices": ["None"] + list(data_manager.templates_folders)}, "refresh_all_templates_folders")
                folder_templates = gr.Dropdown(label='所选文件夹中模板', elem_id="folder_templates", choices=["None"] + list(data_manager.templates), value=data_manager.choose_template)
                create_refresh_button(folder_templates, data_manager.refresh_templates, lambda: {"choices": ["None"] + list(data_manager.templates)}, "refresh_all_templates_in_folder")

        with FormRow():
            template_name = gr.Textbox(label="正在编辑的模板名称", elem_id="template_name", value=base_data.template_name)
            template_folder = gr.Textbox(label="正在编辑的模板所处文件夹", elem_id="template_folder", value=data_manager.choose_folder)
            template_option = gr.Textbox(label="特殊设置", elem_id="template_option", value=base_data.options)
            template_type = gr.Dropdown(label='类型', elem_id="template_type", choices=["txt2img", "img2img"], value="txt2img")


        with FormRow():
            save = gr.Button('保存模板', elem_id = 'save_template')
            load = gr.Button('加载模板', elem_id = 'load_template')
            test = gr.Label("test",elem_id="test")
        save.click(
            fn=data_manager.save_parameter,
            inputs= [template_folder, template_name, template_option, template_type] + txt2img_args,
            outputs=[
                test
            ]
        )
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, id, args in interfaces:
                with gr.TabItem(label, id=id, elem_id='tab_' + id):
                    interface.render()

        tabs.change(
            fn=update_tabs,
            inputs=txt2img_interface,
            outputs=test,
        )
                
        # parameter_type.change(update_parameter_display, parameter_type, test)
        # update_parameter_display(parameter_type)
    return demo

def webui():
    api_models.demo = create_ui()
    api_models.demo.launch(
        share=False,
    )

if __name__ == "__main__":
    webui()


# # model.load('parameters.json')

# def update_parameters(model: Txt2ImgModel):
#     # Save the model to a JSON file
#     # model.save('parameters.json')

#     return "Parameters saved to parameters.json"

# iface = gr.Interface(
#     update_parameters,
#     [
#         gr.inputs.Checkbox(label="Enable HR", default=model.enable_hr),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.denoising_strength, label="Denoising Strength"),
#         gr.inputs.Slider(minimum=0, maximum=1000, default=model.firstphase_width, label="First Phase Width"),
#         gr.inputs.Slider(minimum=0, maximum=1000, default=model.firstphase_height, label="First Phase Height"),
#         gr.inputs.Slider(minimum=1, maximum=10, default=model.hr_scale, label="HR Scale"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.hr_second_pass_steps, label="HR Second Pass Steps"),
#         gr.inputs.Slider(minimum=0, maximum=1000, default=model.hr_resize_x, label="HR Resize X"),
#         gr.inputs.Slider(minimum=0, maximum=1000, default=model.hr_resize_y, label="HR Resize Y"),
#         gr.inputs.Textbox(default=model.prompt, label="Prompt"),
#         gr.inputs.Textbox(default=",".join(model.styles), label="Styles (comma-separated)"),
#         gr.inputs.Textbox(default=model.override_settings["sd_model_checkpoint"], label="SD Model Checkpoint"),
#         gr.inputs.Slider(minimum=-1, maximum=1000, default=model.seed, label="Seed"),
#         gr.inputs.Slider(minimum=-1, maximum=1000, default=model.subseed, label="Subseed"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.subseed_strength, label="Subseed Strength"),
#         gr.inputs.Slider(minimum=-1, maximum=1000, default=model.seed_resize_from_h, label="Seed Resize From H"),
#         gr.inputs.Slider(minimum=-1, maximum=1000, default=model.seed_resize_from_w, label="Seed Resize From W"),
#         gr.inputs.Slider(minimum=1, maximum=10, default=model.batch_size, label="Batch Size"),
#         gr.inputs.Slider(minimum=1, maximum=100, default=model.n_iter, label="Number of Iterations"),
#         gr.inputs.Slider(minimum=1, maximum=1000, default=model.steps, label="Steps"),
#         gr.inputs.Slider(minimum=1, maximum=10, default=model.cfg_scale, label="CFG Scale"),
#         gr.inputs.Slider(minimum=1, maximum=4096, default=model.width, label="Width"),
#         gr.inputs.Slider(minimum=1, maximum=4096, default=model.height, label="Height"),
#         gr.inputs.Checkbox(label="Restore Faces", default=model.restore_faces),
#         gr.inputs.Checkbox(label="Tiling", default=model.tiling),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.eta, label="Eta"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.s_churn, label="S Churn"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.s_tmax, label="S Tmax"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.s_tmin, label="S Tmin"),
#         gr.inputs.Slider(minimum=0, maximum=100, default=model.s_noise, label="S Noise"),
#         gr.inputs.Checkbox(label="Override Settings Restore Afterwards", default=model.override_settings_restore_afterwards),
#         gr.inputs.Textbox(default=",".join(model.script_args), label="Script Args (comma-separated)"),
#         gr.inputs.Dropdown(choices=["Euler", "Other1", "Other2"], default=model.sampler_index, label="Sampler Index"),
#     ],
#     "text",
#     examples=[
#         # Add some example inputs if needed
#     ]
# )
# iface.launch()

# Load parameters from JSON file
# model = parse_json('parameters.json')