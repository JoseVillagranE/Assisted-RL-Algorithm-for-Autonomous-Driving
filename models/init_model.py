from DummyModel import DummyModel


def init_model(model_name, action_space):


    if model_name == "DummyModel":
        model = DummyModel(action_space)

    return model
