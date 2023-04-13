import utils as api

modeltitle = "nightSkyYOZORAStyle_yozoraV1Origin.safetensors [e7bf829cff]"
input_image_url = "C:/Users/Ieunn/Pictures/00013-814328471.jpg"
input_mask_url = ""
setup_args = {
    "prompt": "white hair girl, red eyes, fashionclothes", #for txt2img
    #"init_images": "imgurl", #for img2img
    "steps": 50,
}
controlnet_args = {
    "module": "depth",
    "model": "control_sd15_depth [fef5e48e]",
}

def test_request():
    test = api.Txt2Img()
    test.setup(input_image_url, input_mask_url, setup_args, controlnet_args)
    js = test.send_request()
    return js

import io, base64
import matplotlib.pyplot as plt
from PIL import Image

def test_show_image(js):
    pil_img = Image.open(input_image_url)
    image = Image.open(io.BytesIO(base64.b64decode(js["images"][0])))
    mask_image = Image.open(io.BytesIO(base64.b64decode(js["images"][1])))

    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(pil_img)   
    axarr[1].imshow(image)
    axarr[2].imshow(mask_image)
    plt.show()

#api.utils.set_checkpoints(modeltitle)
js = test_request()
test_show_image(js)