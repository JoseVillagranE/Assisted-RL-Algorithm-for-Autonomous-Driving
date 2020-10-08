from .DummyModel import DummyModel
from .manual_model import Manual_Model
from .DDPG import DDPG

def init_model(model_name, action_space, h_image_in, w_image_in,
            actor_lr, critic_lr, batch_size, gamma, tau, type_RM, max_memory_size,
            device = 'cpu', rw_weights=None, actor_linear_layers=[]):


    if model_name == "DummyModel":
        model = DummyModel(action_space)

    elif model_name == "manual_model":
        model = Manual_Model(action_space)

    elif model_name == "DDPG":
        model = DDPG(action_space,
                    h_image_in,
                    w_image_in,
                    actor_lr = actor_lr,
                    critic_lr = critic_lr,
                    batch_size = batch_size,
                    gamma = gamma,
                    tau = tau,
                    type_RM = type_RM,
                    max_memory_size = max_memory_size,
                    device=device,
                    rw_weights=rw_weights,
                    actor_linear_layers=actor_linear_layers)

    else:
        NotImplementedError("Dont exist that model that you required")
    return model
