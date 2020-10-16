import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from .np_transforms import Compose, Scale, CenterCrop, Normalize_mean_std, ToTensor

# https://github.com/bitsauce/Carla-ppo
def preprocess_frame(frame):
    frame = frame.astype(np.float32)/255.0
    return frame

def create_encode_state_fn(Resize, CenterCrop, mean, std):

    def encode_state(env):
        # frame = preprocess_frame(env.observation) # np.ndarray
        frame = Image.fromarray(env.observation)

        preprocess = transforms.Compose([ ## Only transform PIL Image
        transforms.Resize(Resize),
        transforms.CenterCrop(CenterCrop),
        transforms.ToTensor(), # map image to 0 1
        #transforms.Normalize(mean=mean, std=std),
        ])
        encoded_state = preprocess(frame)
        return encoded_state

    return encode_state
