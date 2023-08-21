import base64
import random
import numpy as np
import torch
from PIL import Image
import time
import simplejpeg
import base64

import clip

COMPILE_ROUNDS = 2
print("loading torch")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

for i in range(COMPILE_ROUNDS):
    print("compiling round", i + 1)
    t1 = time.time()

    batch_size = 6

    images = [preprocess(Image.open("CLIP.png"))] * batch_size
    image_input = torch.tensor(np.stack(images)).to(device)

    text = tokenizer(["mama sita"]).to(device)

    print("fetching features")

    with torch.no_grad():
        image_embed = model.encode_image(image_input).float()
        text_embed = model.encode_text(text).float()

    image_embed /= image_embed.norm(dim=-1, keepdim=True)
    text_embed /= text_embed.norm(dim=-1, keepdim=True)
    similarity = (text_embed.cpu().numpy() @ image_embed.cpu().numpy().T)[0]

    print(similarity)
    print(np.argmax(similarity))

    print("compiled in", time.time() - t1)

print("\n\ncompiled 10 rounds")

print("inferencing hcaptcha classes")

# imgs is a dict in format
# { uid: img }
def predict(texts, img):
    global text_embed

    data = [f"a photo of {label}" for label in texts]

    text_inputs = tokenizer(data).to(device)
    with torch.no_grad():
        text_embed = model.encode_text(text_inputs).float()
        text_embed /= text_embed.norm(dim=-1, keepdim=True)


    img_bytes = base64.b64decode(img)
    raw_images = [preprocess(Image.fromarray(simplejpeg.decode_jpeg(img_bytes)))]

    images_input = torch.tensor(np.stack(raw_images)).to(device)

    with torch.no_grad():
        image_embeds = model.encode_image(images_input).float()


    for image_embed in image_embeds:
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        similarity = (text_embed.cpu().numpy() @ image_embed.cpu().numpy().T)

        index = np.argmax(similarity)

        # return (text - sim, text - sim, ...)
        return [f"{texts[i]} - {similarity[i]}" for i in range(len(texts))]
