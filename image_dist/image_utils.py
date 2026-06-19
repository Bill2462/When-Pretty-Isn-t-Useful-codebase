import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.functional as TF

from tqdm import tqdm

import data_utils
import vendi

from PIL import Image
from data_utils import Example, Group

def get_inception(pretrained=True, pool=True):
    if pretrained:
        weights = Inception_V3_Weights.DEFAULT
    else:
        weights = None
    model = inception_v3(
        weights=weights, transform_input=True
    ).eval()
    if pool:
        model.fc = nn.Identity()
    return model

def inception_transforms():
    return transforms.Compose(
        [
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ]
    )


def get_embeddings(
    images,
    model=None,
    transform=None,
    batch_size=64,
    device=torch.device("cuda"),
):
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()
    uids = []
    embeddings = []
    for batch in tqdm(data_utils.to_batches(images, batch_size)):
        x = torch.stack([transform(Image.open(img).convert("RGB")) for img in batch], 0).to(device)
        with torch.no_grad():
            output = model(x)
        if type(output) == list:
            output = output[0]
        output_arr = output.squeeze().cpu().numpy()
        if output_arr.ndim==1:
            output_arr = output_arr.reshape(1, output_arr.size)
        embeddings.append(output_arr)
    return np.concatenate(embeddings, 0)


def get_pixel_vectors(images, resize=32):
    if resize:
        images = [img.resize((resize, resize)) for img in images]
    return np.stack([np.array(img).flatten() for img in images], 0)


def get_inception_embeddings(images, batch_size=64, device="cuda"):
    if type(device) == str:
        device = torch.device(device)
    model = get_inception(pretrained=True, pool=True).to(device)
    transform = inception_transforms()
    return get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )


def pixel_vendi_score(images, resize=32):
    X = get_pixel_vectors(images)
    n, d = X.shape
    if n < d:
        return vendi.score_X(X)
    return vendi.score_dual(X)


def embedding_vendi_score(
    images, batch_size=64, device="cuda", model=None, transform=None
):
    X = get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )
    n, d = X.shape
    if n < d:
        return vendi.score_X(X)
    return vendi.score_dual(X)

