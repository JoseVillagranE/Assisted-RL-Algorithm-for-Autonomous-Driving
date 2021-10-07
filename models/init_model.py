from .DummyModel import DummyModel
from .manual_model import Manual_Model
from .DDPG import DDPG
from .TD3 import TD3
from .PADDPG import PADDPG
from .BC import BC
from .CoL import CoL
from .TD3CoL import TD3CoL


def init_model(config):

    model = None

    if config.run_type == "DummyModel":
        model = DummyModel(config.train.action_space)

    elif config.run_type == "manual_model":
        model = Manual_Model(
            2,
            wp_encode=config.train.wp_encode,
            wp_encoder_size=config.train.wp_encoder_size,
        )

    elif config.run_type == "DDPG":
        model = DDPG(config)

    elif config.run_type == "PADDPG":
        model = PADDPG(config)
        
    elif config.run_type == "TD3":
        model = TD3(config)

    elif config.run_type == "BC":  # for the moment is just for evaluation
        model = BC(
            state_dim=config.train.state_dim,
            action_space=config.train.action_space,
            type_AC=config.model_type,
            VAE_weights_path=config.train.VAE_weights_path,
            wp_encode=config.train.wp_encode,
            wp_encoder_size=config.train.wp_encoder_size,
        )

    elif config.run_type == "CoL":
        model = CoL(config)
    
    elif config.run_type == "TD3CoL":
        model = TD3CoL(config)
    
    else:
        raise NotImplementedError("Dont exist that model that you required")
    return model
