import os
import modules.file_util as file_util
import json
from modules.api_models import Txt2ImgModel, ApiType, TemplateBaseModel, Img2ImgModel


work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
work_dir = os.path.join(work_dir, "templates")

#利用file_util中的方法，获取work_dir下所有文件夹
def get_all_template():
    template = file_util.get_dirs(work_dir)
    return template

#利用file_util中的方法，获取某个template文件夹下所有json文件
def get_models_from_template(template):
    json_files = file_util.get_json_files(os.path.join(work_dir, template))
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
def save_template_model(template, apiTypeModel:TemplateBaseModel):
    json_file = os.path.join(work_dir, template, apiTypeModel.name + ".json")
    content = json.dumps(apiTypeModel, default=lambda o: o.__dict__)
    file_util.write_json_file(json_file, content)
    
def check_template_is_exist(template):
    return template in get_all_template()

def check_model_is_exist(template, file):
    return file in get_models_from_template(template)