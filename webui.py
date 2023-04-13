import gradio as gr
from gradio import components
from api_models import Txt2ImgModel, ApiType, apiTypeModel
import json
from ui_components import FormRow, FormGroup, ToolButton, FormHTML
# # Load parameters from JSON file
model = Txt2ImgModel()
demo = None
interfaces = []

def update_parameters(model: Txt2ImgModel):
    # Save the model to a JSON file
    model.save('parameters.json')

    return "Parameters saved to parameters.json"

def update_parameter_display(value):
    for _, label, id in interfaces:
        container = demo.get_component('tab_' + id)
        print(container)
        if value == label:
            container.show()
        else:
            container.hide()

def create_ui():
    api_type = ["txt2img", "img2img"]
    with gr.Blocks() as txt2img_parameter:
        with gr.Row():
            with gr.Column(variant='compact', elem_id="txt2img paramater"):
                with FormRow(elem_id=f"Enable HR"):
                    # sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
                    # steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
                    seed = gr.Textbox(label='Seed', value= -1 , elem_id = 'txt2img_seed')

    with gr.Blocks() as img2img_parameter:
        with gr.Row():
            with gr.Column(variant='compact', elem_id="txt2img paramater"):
                with FormRow(elem_id=f"Enable HR"):
                    # sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
                    # steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
                    seed = gr.Textbox(label='Seed2', value= -1 , elem_id = 'img2img_seed')

    interfaces = [
        (txt2img_parameter, "txt2img", "txt2img"),
        (img2img_parameter, "img2img", "img2img"),
    ]

    with gr.Blocks(analytics_enabled=False, title="Stable Diffusion") as demo:
        parameter_type = gr.Dropdown(label='type', elem_id="parameter_type", on_change=update_parameter_display, choices=api_type, value=api_type[0])
        test = gr.Blocks()
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, id in interfaces:
                with gr.TabItem(label, id=id, elem_id='tab_' + id):
                    interface.render()
        # parameter_type.change(update_parameter_display, parameter_type, test)
        update_parameter_display(parameter_type.value)
    return demo

def webui():
    demo = create_ui()
    demo.launch(
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