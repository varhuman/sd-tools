# Stable Diffusion Tool

欢迎使用我为 [stable diffusion](https://github.com/facebookresearch/stable_diffusion) 开发的辅助工具。

本工具的主要目标是解决以下三个问题：

1. **批量生成局限性**：虽然 stable diffusion 支持多样式（styles）的生成，但是这种方式并不尽人意。另外，我们不能精确控制每个样式的多个参数。

2. **样式（styles）保存不够细节**：在这个项目中，我引入了例如controlnet的详细参数，包括图片、mask等，每个样式现在可以单独控制一些参数。

3. **参数共享不便**：在本项目中，我将参数以 JSON 格式存储在本地，这使得在团队合作时，我们可以更方便地共享和使用相同的样式模板。

## 📁 文件目录

- `templates`：用于存储样式模板。
- `logs`：用于存储运行日志（暂未优化）。
- `modules`：存放主要代码。
- `webui.py`：项目主要入口。

## 🚀 安装与启动

1. 运行 `install.bat` 以创建虚拟环境并安装所需的依赖项。
2. 运行 `start.bat` 以启动项目。

> ⚠️ 注意：目前项目的界面尚未优化，后续将会进行改进。

祝你编程愉快！
