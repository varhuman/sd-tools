#the file is to be used for file operations
import os
from modules.log_util import logger

#检查文件夹是否存在，不存在则创建
def check_folder(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

#获取指定路径下所有文件夹
def get_dirs(path):
    dirs = []
    if not os.path.exists(path):
        logger.error(f"get_dirs: the path {path} is not exist")
        return dirs
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            dirs.append(dir)
    return dirs

#获得指定文件下所有后缀为json得文件，支持深度搜索
def get_json_files(path, deep=True):
    files = []
    if not os.path.exists(path):
        return files
    if deep:
        for dir in get_dirs(path):
            files.extend(get_json_files(os.path.join(path, dir), deep))
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.endswith(".json"):
            # files add file without json suffix
            files.append(file[:-5])
    return files

#指定目录下获取新的文件名字，格式为base_name_{i}extension
def get_new_file_name(folder, base_name, extension, index=1):
    name = base_name + "_" + str(index) + '.' + extension
    if name in os.listdir(folder):
        return get_new_file_name(folder, base_name, extension, index+1)
    return name

#获得指定文件夹下每个文件夹中所有得json文件，分别返回
def get_json_files_by_dir(path):
    files = {}
    for dir in get_dirs(path):
        files[dir] = get_json_files(os.path.join(path, dir))
    return files

#检查文件夹下是否存在该文件夹
def check_folder_is_exist(path:str, folder):
    return folder in get_dirs(path)

#把json内容写入指定得json文件中
def write_json_file(path:str, content):
    #先检查是否存在该文件夹
    check_folder(os.path.dirname(path))
    #先检查是否是json文件
    if not path.endswith(".json"):
        logger.error(f"write_json_file: the file {path} is not a json file")
        return False
    
    with open(path, 'w') as f:
        f.write(content)


def read_json_file(path:str):
    #先检查是否是json文件
    if not path.endswith(".json"):
        logger.error(f"read_json_file: the file {path} is not a json file")
        return False
    with open(path, 'r') as f:
        return f.read()

#创建指定文件夹在指定路径
def create_folder(path:str, folder):
    if not check_folder_is_exist(path, folder):
        os.mkdir(os.path.join(path, folder))
        return True
    else:
        logger.error(f"create_folder: the folder {folder} is already exist")
        return False