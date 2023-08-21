import base64
import random
import numpy as np
import jax
from PIL import Image
import time
import simplejpeg

import clip_jax

print("loading jax")

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu")

devices = jax.local_devices()

print(f"jax devices: {devices}")

jax_params = jax.device_put_replicated(jax_params, devices)

image_fn = jax.pmap(image_fn)
text_fn = jax.pmap(text_fn)

print("moved jax, compiling")
t1 = time.time()

batch_size = 1

jax_image = np.expand_dims(jax_preprocess(Image.open("CLIP.png")), (0, 1))
jax_image = np.repeat(jax_image, len(devices), axis=0)
jax_image = np.repeat(jax_image, batch_size, axis=1)

jax_text = np.expand_dims(clip_jax.tokenize(["something crazy"]), 0)
jax_text = np.repeat(jax_text, len(devices), axis=0)
jax_text = np.repeat(jax_text, batch_size, axis=1)

_ = image_fn(jax_params, jax_image)
_ = text_fn(jax_params, jax_text)

print("compiled in", time.time() - t1)

text_embed_cache = {}


@jax.jit
def cosine_similarity(image_features, text_features):
    image_features = jax.numpy.expand_dims(image_features, 0)
    image_features_norm = jax.numpy.linalg.norm(image_features, axis=1, keepdims=True)
    text_features_norm = jax.numpy.linalg.norm(text_features, axis=0, keepdims=True)

    # Distance matrix of size (b, n).
    return (
        (image_features @ text_features) / (image_features_norm @ text_features_norm)
    ).T


def predict(text, imgs):
    
    if text in text_embed_cache.keys():
        text_embed = text_embed_cache[text]
    else:
        jax_text = np.expand_dims(clip_jax.tokenize([text]), 0)
        jax_text = np.repeat(jax_text, len(devices), axis=0)
        jax_text = np.repeat(jax_text, batch_size, axis=1)
        text_embed = text_fn(jax_params, jax_text)[0][0]

        # put in cache
        text_embed_cache[text] = text_embed

    results = []

    jax_image = np.expand_dims(jax_preprocess(Image.fromarray(simplejpeg.decode_jpeg(imgs[0]))),
                               (0, 1))
    
    for img in imgs[1:]:
        jax_image = np.append(jax_image, np.expand_dims(jax_preprocess(Image.fromarray(simplejpeg.decode_jpeg(img))), (0, 1)), axis=0)

    jax_image = np.repeat(jax_image, batch_size, axis=1)

    image_embed = image_fn(jax_params, jax_image)
    results.append(random.randint(0, 1000) / 1000.0)

    for i2 in range(batch_size):
        for i1 in range(len(devices)):  # 8*128 =1024
            sim = float(cosine_similarity(image_embed[i1][i2], text_embed)[0])
            # print(sim)

            results.append(sim)

    results = results[:-2]
    results.pop(0)
    # print(results)

    return results.index(max(results))
