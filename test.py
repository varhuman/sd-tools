api_type = [
    {
        name: "文字转图片",
        "value": "txt2img",
    },
    {
        "name": "Image to Image",
        "value": "img2img",
    },
]

test = {}
#add test.name = "test"
test["name"] = "test"

api_type.append(test)

print(api_type[2].name)
