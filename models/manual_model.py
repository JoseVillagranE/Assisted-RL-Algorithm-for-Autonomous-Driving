import numpy as np
import pygame
from pygame.locals import *
from ConvVAE import VAE_Actor, VAE_Critic
from ExperienceReplayMemory import SequentialDequeMemory, \
                                   RandomDequeMemory, \
                                   PrioritizedDequeMemory, \
                                   cat_experience_tuple


class Manual_Model:

    def __init__(self,
                 state_dim=128,
                 action_space=2,
                 n_channel=3,
                 z_dim=128,
                 beta=1,
                 type_RM="random",
                 max_memory_size=10000,
                 rw_weights=[],
                 batch_size=64,
                 wp_encode=False,
                 wp_encoder_size=64,
                 VAE_weights_path="./models/weights/segmodel_expert_samples_sem_all.pt"):

        self.action_space = action_space
        self.action = np.zeros(self.action_space)        
        self.type_RM = type_RM
        
        
        self.actor = VAE_Actor(state_dim,
                               action_space,
                               n_channel,
                               z_dim,
                               VAE_weights_path=VAE_weights_path,
                               beta=beta,
                               wp_encode=wp_encode,
                               wp_encoder_size=wp_encoder_size).float()
        
        
        if type_RM == "sequential":
            self.replay_memory = SequentialDequeMemory(rw_weights)
            self.replay_memory_e = SequentialDequeMemory(rw_weights)
        elif type_RM == "random":
            self.replay_memory = RandomDequeMemory(queue_capacity=max_memory_size,
                                                    rw_weights=rw_weights,
                                                    batch_size=batch_size)
            self.replay_memory_e = RandomDequeMemory(queue_capacity=max_memory_size,
                                                    rw_weights=rw_weights,
                                                    batch_size=batch_size)
            
        elif type_RM == "prioritized":
            self.replay_memory = PrioritizedDequeMemory(queue_capacity=max_memory_size,
                                                        alpha = alpha,
                                                        beta = beta,
                                                        rw_weights=rw_weights,
                                                        batch_size=batch_size)
            self.replay_memory_e = PrioritizedDequeMemory(queue_capacity=max_memory_size,
                                                        alpha = alpha,
                                                        beta = beta,
                                                        rw_weights=rw_weights,
                                                        batch_size=batch_size)
        

    def predict(self, state):

        pygame.event.pump()
        self.keys = pygame.key.get_pressed()

        if self.keys[K_LEFT] or self.keys[K_a]:
            self.action[0] = -0.5
        elif self.keys[K_RIGHT] or self.keys[K_d]:
            self.action[0] = 0.5
        else:
            self.action[0] = 0.0
        self.action[0] = np.clip(self.action[0], -1, 1)
        self.action[1] = 1.0 if self.keys[K_UP] or self.keys[K_w] else 0.0

        return self.action
    
    def load_state_dict(self, path):
        pass
    
    def feat_ext(self, image):
        return self.actor.feat_ext(image)
    
    def wp_encode_fn(self, wp):
        return self.actor.wp_encode_fn(wp)
        
