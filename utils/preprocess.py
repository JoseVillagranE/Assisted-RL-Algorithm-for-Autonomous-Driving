import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from utils.utils import vector, wp_features_function

# https://github.com/bitsauce/Carla-ppo
def preprocess_frame(frame):
    frame = frame.astype(np.float32)/255.0
    return frame


def create_encode_state_fn(Resize_h, Resize_w, CenterCrop, mean, std,
                           measurement_to_include,
                           vae_encode=None,
                           feat_wp_encode=None):
    
    measure_flags = ["steer" in measurement_to_include,
                     "throttle" in measurement_to_include,
                     "speed" in measurement_to_include,
                     "orientation" in measurement_to_include]
        
    def encode_state(env):
        """
        :env.observation (ndarray) [H, W, C]
        :encode_state (Tensor)
        """
        
        preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Resize_h, Resize_w)),
        #transforms.CenterCrop(CenterCrop),
        transforms.ToTensor()
        ])
        encoded_state = preprocess(env.observation)
        if vae_encode is not None:
            encoded_state = vae_encode(encoded_state.unsqueeze(0)).squeeze().detach().numpy()
            measurements = []
            if measure_flags[0]: measurements.append(env.agent.control.steer)
            if measure_flags[1]: measurements.append(env.agent.control.throttle)
            if measure_flags[2]: measurements.append(env.agent.get_speed()*3.6) # km/h
            if measure_flags[3]: measurements.extend(vector(env.agent.get_forward_vector()))
            #measurements = torch.tensor(measurements).float()
            #encoded_state = torch.cat((encoded_state, measurements))
            encoded_state = np.append(encoded_state, measurements).astype(float)
            
        else:
            encoded_state = encoded_state.numpy() 
            
        if feat_wp_encode:
            feat_angle = wp_features_function(vector(env.agent.get_velocity()),
                                              env.agent.get_list_next_wps(n=5))
            encoded_angle = feat_wp_encode(feat_angle)
            encoded_state = np.append(encoded_state, encoded_angle).astype(float)
        return encoded_state

    return encode_state
