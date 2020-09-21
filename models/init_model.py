from .DummyModel import DummyModel
from .manual_model import Manual_Model
from .DDPG import DDPG

def init_model(model_name, action_space):


    if model_name == "DummyModel":
        model = DummyModel(action_space)

    elif model_name == "manual_model":
        model = Manual_Model(action_space)

    elif model_name == "DDPG":
        model = DDPG()

    else:
        NotImplementedError("Dont exist that model that you required")
    return model
