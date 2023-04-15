import os
import modules.file_util as file_util
import json
from modules.api_models import Txt2ImgModel, ApiType, TemplateBaseModel, Img2ImgModel, to_serializable

work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
work_dir = os.path.join(work_dir, "templates")

#利用file_util中的方法，获取work_dir下所有文件夹
def get_all_templates_folders():
    folders = file_util.get_dirs(work_dir)
    return folders

def get_new_template_folder_name():
    folders = get_all_templates_folders()
    #find template{i} is exist
    i = 1
    while True:
        folder = "template_folder" + str(i)
        if folder not in folders:
            return folder
        i += 1

def get_new_template_name(folder):
    templates = get_templates_from_folder(folder)
    #find template{i} is exist
    i = 1
    while True:
        template = "template" + str(i)
        if template not in templates:
            return template
        i += 1

#利用file_util中的方法，获取某个template文件夹下所有json文件
def get_templates_from_folder(folder):
    json_files = file_util.get_json_files(os.path.join(work_dir, folder))
    return json_files

#将json先解析成apiTypeModel，根据apiTypeModel中得type再决定解析成哪个model
def get_template_model(json_file):
    content = file_util.read_json_file(json_file)
    apiTypeModel:TemplateBaseModel = json.loads(content)
    if apiTypeModel.type == ApiType.img2img:
        apiTypeModel.api_model = Img2ImgModel(**apiTypeModel.api_model)
    elif apiTypeModel.type == ApiType.txt2img:
        apiTypeModel.api_model = Txt2ImgModel(**apiTypeModel.api_model)
    return apiTypeModel

#将apiTypeModel转换成json并存储到template得指定文件夹下
def save_template_model(folder, apiTypeModel:TemplateBaseModel):
    json_file = os.path.join(work_dir, folder, apiTypeModel.template_name + ".json")
    content = json.dumps(apiTypeModel, default= to_serializable, indent=4)
    file_util.write_json_file(json_file, content)
    
def check_templates_folder_is_exist(folder):
    return folder in get_all_templates_folders()

def check_templates_folder_is_exist(folder, template):
    return template in get_templates_from_folder(folder)
