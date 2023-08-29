import base64
import random
import numpy as np
import torch
from PIL import Image
import time
import simplejpeg
import base64
import threading


LOCK = threading.Lock()

COMPILE_ROUNDS = 2
print("loading torch")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')


print("inferencing hcaptcha classes")
tt = time.time()

# add as many classes as possible,
# even of images that are not 'asked for' via the captcha, but still displayed.
# this will help the model learn to ignore them
# format:
# {hCaptcha's name} : {class name for the model to use}
# e. hCaptcha uses 'a Yoko'  which is a japanese amulet, so rename it to 'an amulet'
classes_text = {
    "a fried egg on a brеaԁ": "a fried egg on a bread",
    "a panda in wаtеr": "a panda in water",
    "an otter in wаtеr": "an otter in water",
    "a panda on a couch": "a panda on a couch",
    "a violin": "a violin",
    "a piano": "a piano",
    "a console cοntroller": "a console cοntroller",
    "a smartphone": "a smartphone",
    "a camera": "a camera",
    "a telescope": "a telescope",
    "a globe": "a globe",
    "a paintbrush": "a paintbrush",
    "a Yoko": "an amulet",
    "a bird": "a bird",
    "a stork": "a stork",
    "a watch": "a watch",
    "an earing": "an earing",
    "a turtle": "a turtle",
    "a timber door": "a timber door",
    "a wardrobe": "a wardrobe",
    "a pencil": "a pencil",
    "a bed": "a bed",
    "a motorcycle": "a motorcycle",
    "a table": "a table",
    "a river": "a river",
    "a helicopter": "a helicopter",
    "a flower": "a flower",
    "a chess table": "a chess table",
    "a microwave": "a microwave",
    "a nintendo switch": "a nintendo switch",
    "a bag": "a bag",
    "a tent": "a tent",
    "a keyboard": "a keyboard",
    "a mouse": "a mouse",
    "a truck": "a truck",
    "a crumpled paper ball": "a crumpled paper ball",
    "a tractor": "a tractor",
    "a robot": "a robot",
    "a bus": "a bus",
    "a car": "a car"
}

text_classes = []
for c in classes_text.values():
    text_classes.append(c)


text_embed = None
def reinfer_text_embeds():
    global text_embed, text_classes

    # prefix with 'a photo of '
    data = [f"a photo of {label}" for label in text_classes]
    print(data)

    text_inputs = tokenizer(data).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embed = model.encode_text(text_inputs).float()
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

    print("reinferenced text embeds", text_embed.shape)


reinfer_text_embeds()


# imgs is a dict in format
# { uid: img }
def predict(text, imgs_map):
    global text_embed

    if text not in classes_text.keys():
        print("got class", text, "which doesn't exist!")
        print(classes_text)
        print("reinferencing text embeds")

        classes_text[text] = text
        text_classes.append(classes_text[text])

        print(classes_text)

        with LOCK:
            reinfer_text_embeds()

    raw_images = {}
    for uid, img_b64 in imgs_map.items():
        img_bytes = base64.b64decode(img_b64)
        raw_images[uid] = preprocess(Image.fromarray(simplejpeg.decode_jpeg(img_bytes)))

    images_input = torch.tensor(np.stack(raw_images.values())).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(), LOCK:
        image_embeds = model.encode_image(images_input).float()


    results = []

    i = 0
    for image_embed in image_embeds:
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        similarity = (text_embed.cpu().numpy() @ image_embed.cpu().numpy().T)

        index = np.argmax(similarity)

        # print highest class probability & image uid
        #print(list(raw_images.keys())[i])
        print(text_classes[index], list(raw_images.keys())[i])
        if text_classes[index] == text:
            results.append(True)
        else:
            results.append(False)

        i+=1

    print(results)
    # combine uid and result
    #results = dict(zip(results))

    print("results")
    print(results)

    return results


import glob


def test():
    images = {}
    for f in glob.iglob("testing/*"):
        raw_img = open(f, "rb").read()
        images[f] = base64.b64encode(raw_img)


    results = predict("a guitar", images)


test()
