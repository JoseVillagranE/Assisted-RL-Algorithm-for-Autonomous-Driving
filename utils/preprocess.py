import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# https://github.com/bitsauce/Carla-ppo
def preprocess_frame(frame):
    frame = frame.astype(np.float32)/255.0
    return frame


def create_encode_state_fn(Resize, CenterCrop, mean, std):

    def encode_state(env):
        """
        :env.observation (ndarray) [H, W, C]
        :encode_state (Tensor) [C, H, W]
        """
        # frame = preprocess_frame(env.observation) # np.ndarray
        # frame = Image.fromarray(preprocess_frame(env.observation).astype('uint8'), 'RGB')
        # frame = torch.from_numpy(env.observation)
        preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(Resize),
        transforms.CenterCrop(CenterCrop),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean, std=std),
        ])
        encoded_state = preprocess(env.observation)
        return encoded_state

    return encode_state
