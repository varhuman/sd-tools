import gradio as gr
from modules.api_models import ApiType, TemplateBaseModel, SubmitFolderModel,SubmitItemModel, CheckpointModel
import modules.api_models as api_models
import modules.data_manager as data_manager
from ui_components import FormRow, ToolButton, FormGroup
import modules.template_utils as template_utils
import os
import modules.api_util as api_util
import time

refresh_symbol = '\U0001f504'  # 🔄

def restart_ui():
    #this is not working
    time.sleep(0.5)
    data_manager.demo.close()
    
    time.sleep(0.5)
    data_manager.demo = create_ui()
    data_manager.demo.launch(
        share=False,
    )
    
    time.sleep(0.5)

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

def change_folder(choose_folder):
    data_manager.choose_folder = choose_folder
    data_manager.refresh_templates()
    return gr.Dropdown.update(value=data_manager.templates[0] if data_manager.templates else "",choices=data_manager.templates)

#获得model的参数并赋值给txt2ImgData or img2ImgData
def get_model_data(model_path):
    data:TemplateBaseModel = template_utils.get_model_from_template_path(model_path)
    data_manager.base_data = data
    if data.template_type == ApiType.img2img:
        data_manager.img_img_data = data.api_model
    elif data.template_type == ApiType.txt2img:
        data_manager.txt_img_data = data.api_model

def set_txt2img_type():
    return ApiType.txt2img.value

def set_img2img_type():
    return ApiType.img2img.value

def create_txt2img_ui():
    base_data, t2i_data, i2i_data, samplers = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data, data_manager.samplers_k_diffusion

    with gr.Blocks() as txt2img_bolcks:
        with gr.Row():
            with gr.Column(variant='compact'):
                with FormRow(elem_id="txt2img row1"):
                    checkpoint_model = gr.Dropdown(label='Model', elem_id="txt2img_checkpoint_model", choices=[x.model_name for x in data_manager.checkpoints_models], value=t2i_data.get_checkpoint_model())
                    create_refresh_button(checkpoint_model, data_manager.refresh_checkpoints, lambda: {"choices": [x.model_name for x in data_manager.checkpoints_models]}, "txt2img_checkpoint_model")
                with FormRow(elem_id="txt2img row1"):
                    txt2img_prompt = gr.Textbox(label="prompt", elem_id="txt2img_prompt", value=t2i_data.prompt)
                with FormRow(elem_id="txt2img row2"):
                    txt2img_negative_prompt = gr.Textbox(label="negative_prompt", elem_id="txt2img_negative_prompt", value=t2i_data.negative_prompt)
                with FormRow(elem_id="txt2img row3"):
                    restore_faces = gr.Checkbox(label="restore_faces", elem_id="txt2img_restore_faces", value=t2i_data.restore_faces)
                    tiling = gr.Checkbox(label="tiling", elem_id="txt2img_tiling", value=t2i_data.tiling)
                    seed = gr.Number(label='Seed', value= -1 , elem_id = 'txt2img_seed')
                    sampler_index = gr.Dropdown(label='Sampling method', elem_id="txt2img_sampling", choices=samplers, value=t2i_data.sampler_index)
                    
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

def create_img2img_ui():
    base_data, t2i_data, i2i_data, samplers = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data, data_manager.samplers_k_diffusion

    with gr.Blocks() as img2img_bolcks:
        with gr.Row():
            with gr.Column(variant='compact'):
                with FormRow(elem_id="img2img images"):
                    img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base", value=i2i_data.init_images)
                    mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask", value=i2i_data.mask)
                with FormRow(elem_id="img2img image params"):
                    with FormGroup(elem_id="inpaint_controls") as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=i2i_data.mask_blur, elem_id="img2img_mask_blur")
                            with FormRow():
                                inpainting_mask_invert = gr.Radio(label='重绘蒙版', choices=['Inpaint masked', 'Inpaint not masked'], value=i2i_data.inpainting_mask_invert, type="index", elem_id="img2img_mask_mode")

                            with FormRow():
                                inpainting_fill = gr.Radio(label='蒙版遮住的内容', choices=['fill', 'original', 'latent noise', 'latent nothing'], value=i2i_data.inpainting_fill, type="index", elem_id="img2img_inpainting_fill")

                            with FormRow():
                                with gr.Column():
                                    inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value=i2i_data.inpaint_full_res, elem_id="img2img_inpaint_full_res")

                                with gr.Column(scale=4):
                                    inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=i2i_data.inpaint_full_res_padding, elem_id="img2img_inpaint_full_res_padding")

                with FormRow(elem_id="img2img row1"):
                    checkpoint_model = gr.Dropdown(label='Model', elem_id="img2img_checkpoint_model", choices=data_manager.checkpoints_models, value=t2i_data.get_checkpoint_model())
                with FormRow(elem_id="img2img row1"):
                    img2img_prompt = gr.Textbox(label="prompt", elem_id="img2img_prompt", value=t2i_data.prompt)
                with FormRow(elem_id="img2img row2"):
                    img2img_negative_prompt = gr.Textbox(label="negative_prompt", elem_id="img2img_negative_prompt", value=t2i_data.negative_prompt)
                with FormRow(elem_id="img2img row3"):
                    restore_faces = gr.Checkbox(label="restore_faces", elem_id="img2img_restore_faces", value=t2i_data.restore_faces)
                    tiling = gr.Checkbox(label="tiling", elem_id="img2img_tiling", value=t2i_data.tiling)
                    seed = gr.Number(label='Seed', value= -1 , elem_id = 'img2img_seed')
                    sampler_index = gr.Dropdown(label='Sampling method', elem_id="img2img_sampling", choices=samplers, value=t2i_data.sampler_index)
                    
            with gr.Column(variant='compact'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id="img2img_steps", label="Sampling steps", value=t2i_data.steps)
                cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.5, elem_id="img2img_cfg_scale", label="cfg_scale", value=t2i_data.cfg_scale)
                width = gr.Slider(minimum=64, maximum=1024, step=8, elem_id="img2img_width", label="width", value=t2i_data.width)
                height = gr.Slider(minimum=64, maximum=1024, step=8, elem_id="img2img_height", label="height", value=t2i_data.height)
                batch_size = gr.Slider(minimum=1, maximum=1024, step=1, elem_id="img2img_batch_size", label="batch_size", value=t2i_data.batch_size)
                batch_count = gr.Slider(minimum=1, maximum=100, step=1, elem_id="img2img_n_iter", label="n_iter", value=t2i_data.n_iter)
                eta = gr.Slider(minimum=0, maximum=10, step=1, elem_id="img2img_eta", label="eta", value=t2i_data.eta)
        img2img_args = [
            img2img_prompt,
            img2img_negative_prompt,
            restore_faces,
            tiling,
            seed,
            sampler_index,
            steps,
            cfg_scale,
            width,
            height,
            batch_size,
            batch_count,
            eta,
            inpaint_full_res,
            inpaint_full_res_padding,
            checkpoint_model,
            img_inpaint,
            mask_inpaint,
            mask_blur,
            inpainting_fill,
            inpainting_mask_invert,
                ]
        
    return img2img_bolcks, img2img_args
        


def create_submit_item(item: SubmitItemModel):
    with gr.Blocks() as submit_item:
        with gr.Row().style(equal_height=True):
            is_submit = gr.Checkbox(label="是否执行", elem_id="is_submit" + item.submit_template, value=item.is_submit)
            submit_template = gr.Label(item.submit_template)
            submit_times = gr.Slider(label='执行次数', value=item.submit_times, step=1, minimum=1, maximum=30, elem_id = 'submit_times' + item.submit_template)
    return is_submit, submit_template, submit_times, submit_item

def submit_times_change(item: SubmitItemModel):
    def fn(times, item=item):
        item.submit_times = times
    return fn
def submit_enabled_change(item: SubmitFolderModel):
    def fn(enabled, item=item):
        submit_list = data_manager.submit_list
        for submit_item in submit_list:
            if submit_item.submit_folder == item.submit_folder:
                submit_item.is_submit = enabled
        
    return fn

def refresh_submit_table():
    data_manager.refresh_submit_list()

    return gr.update()

def create_submit():
    submit_list = data_manager.submit_list

    all_ui = []
    with gr.Blocks() as submit_tab:
        with FormRow():
            # check = gr.Button(value="Check for updates")
            create_refresh_button(submit_tab, refresh_submit_table, lambda: {"update": ()}, "refresh_submit_list")
            message = gr.Label("提示信息")
        submit_btn = gr.Button("提交", elem_id="submit_btn")

        submit_btn.click(
            fn=api_util.submit_all,
            inputs=[message]
        )
        # table = gr.HTML(lambda: generate_submit_table()) 虽然可以动态刷新列表，但是效果不理想，暂时不用

        # check.click(
        #     fn=refresh_submit_table,
        #     inputs=[],
        #     outputs=[table]
        # )

        #用这一套目前还不知道如何动态加载，每次都要重新启动才能刷新
        for submit_folder in submit_list:
            submit_item_is_submit = submit_folder.is_submit
            folder_name = submit_folder.submit_folder
            times = submit_folder.submit_times
            submit_items = submit_folder.submit_items
            with gr.Group():
                with gr.Accordion(folder_name, open=False):
                    with gr.Row():
                        enabled = gr.Checkbox(label="是否执行（整体）", value=submit_item_is_submit)
                        submit_times = gr.Slider(label='执行次数（整体)', value=times, step=1, minimum=1, maximum=30, elem_id = 'submit_times' + folder_name)
                        all_ui.append(enabled)
                        all_ui.append(submit_times)
                        enabled.change(
                            fn=submit_enabled_change(item=submit_folder),
                            inputs=[enabled],
                        )
                        submit_times.change(
                            fn=submit_times_change(item=submit_folder),
                            inputs=[submit_times],
                        )
                    with gr.Accordion("子项", open=False):
                        for submit_item in submit_items:
                            submit_item_is_submit, submit_item_template, submit_item_times, submit_blocks = create_submit_item(submit_item)
                            all_ui.append(submit_item_is_submit)
                            all_ui.append(submit_item_times)

    return submit_tab

def generate_submit_table():
    submit_list = data_manager.submit_list
    table_code = f"""
    <table>
        <thead>
            <tr>
                <th>Enabled</th>
                <th>Folder Name</th>
                <th>Times</th>
                <th>Sub-items</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for submit_folder in submit_list:
        # Generate table rows for each submit folder
        folder_name = submit_folder.submit_folder
        times = submit_folder.submit_times
        submit_items = submit_folder.submit_items
        
        sub_items_html = ""
        for submit_item in submit_items:
            # Generate HTML for sub-items
            sub_item_template = submit_item.submit_template
            sub_item_times = submit_item.submit_times
            sub_items_html += f"""
                <li>
                    <span>{sub_item_template}: {sub_item_times} times</span>
                </li>
            """

        row_code = f"""
            <tr>
                <td><label><input class="gr-check-radio gr-checkbox" name="enable_{folder_name}" type="checkbox" {'checked="checked"' if submit_folder.is_submit else ''}>{folder_name}</label></td>
                <td>{folder_name}</td>
                <td><input type="number" min="1" max="30" name="times_{folder_name}" value="{times}"></td>
                <td>
                    <ul>
                        {sub_items_html}
                    </ul>
                </td>
            </tr>
        """
        table_code += row_code

    table_code += """
        </tbody>
    </table>
    """

    return table_code

def init_data():
    data_manager.refresh_checkpoints()
    data_manager.refresh_templates_folders()
    data_manager.refresh_templates()
    data_manager.refresh_submit_list()


def create_ui():
    base_data, t2i_data, i2i_data, samplers, submit_list = data_manager.base_data, data_manager.txt_img_data, data_manager.img_img_data, data_manager.samplers_k_diffusion, data_manager.submit_list
    css = ""

    # for cssfile in data_manager.list_files_with_name("style.css"):
    #     if not os.path.isfile(cssfile):
    #         continue

    #     with open(cssfile, "r", encoding="utf8") as file:
    #         css += file.read() + "\n"


    txt2img_interface, txt2img_args = create_txt2img_ui()
    img2img_interface, img2img_args = create_img2img_ui()
    submit_interface = create_submit()

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img", txt2img_args),
        (img2img_interface, "img2img", "img2img", img2img_args),
        (submit_interface, "submit", "submit", txt2img_args),
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with FormGroup():
            with FormRow():
                with gr.Row().style(equal_height=False):
                    ip_text = gr.Textbox(label='ip', value= api_util.ip , elem_id = 'connect_ip')
                    connect_ip = gr.Button('connect', elem_id = 'connect_ip')
                    all_templates_folders = gr.Dropdown(label='所有模板文件夹', elem_id="all_templates_folders", choices=[""] + list(data_manager.templates_folders), value=data_manager.choose_folder)
                    create_refresh_button(all_templates_folders, data_manager.refresh_templates_folders, lambda: {"choices": [""] + list(data_manager.templates_folders)}, "refresh_all_templates_folders")
                    folder_templates = gr.Dropdown(label='所选文件夹中模板', elem_id="folder_templates", choices=[""] + list(data_manager.templates), value=data_manager.choose_template)
                    create_refresh_button(folder_templates, data_manager.refresh_templates, lambda: {"choices": [""] + list(data_manager.templates)}, "refresh_all_templates_in_folder")
                    load = gr.Button('加载模板', elem_id = 'load_template')

                    all_templates_folders.change(
                        fn=change_folder,
                        inputs=[all_templates_folders],
                        outputs=folder_templates
                    )

        with FormRow():
            template_name = gr.Textbox(label="正在编辑的模板名称", elem_id="template_name", value=base_data.template_name)
            template_folder = gr.Textbox(label="正在编辑的模板所处文件夹", elem_id="template_folder", value=data_manager.choose_folder)
            template_option = gr.Textbox(label="特殊设置", elem_id="template_option", value=base_data.options)
            template_type_label = gr.Label(base_data.template_type, elem_id="template_type")


        with FormRow():
            save = gr.Button('保存模板', elem_id = 'save_template')
            save2 = gr.Button('保存模板2', elem_id = 'save_template')
            test = gr.Label("test",elem_id="test")
        save.click(
            fn=data_manager.save_parameter,
            inputs= [template_folder, template_name, template_option, template_type_label] + txt2img_args,
            outputs=[
                test
            ]
        )
        save2.click(
            fn=data_manager.save_parameter,
            inputs= [template_folder, template_name, template_option, template_type_label] + img2img_args,
            outputs=[
                test
            ]
        )
        load.click(
            fn=data_manager.load_parameter,
            inputs= [all_templates_folders, folder_templates],
            outputs=[
                txt2img_interface
            ]
        )
        tab_items = []
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, id, args in interfaces:
                with gr.TabItem(label, id=id, elem_id='tab_' + id) as tab_item:
                    data_manager.refresh_submit_list()
                    interface.render()
                tab_items.append(tab_item)
                if label == "txt2img":
                    tab_item.select(
                        fn=set_txt2img_type,
                        inputs=[],
                        outputs=template_type_label
                    )
                elif label == "img2img":
                    tab_item.select(
                        fn=set_img2img_type,
                        inputs=[],
                        outputs=template_type_label
                    )
                elif label == "submit":
                    tab_item.select(
                        fn=create_submit,
                        inputs=[],
                        outputs=[submit_interface]
                    )
    return demo

def webui():
    init_data()
    data_manager.demo = create_ui()
    data_manager.demo.launch(
        share=False,
    )

if __name__ == "__main__":
    webui()