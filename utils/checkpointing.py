import torch
import os
import glob

def save_checkpoint(models_dicts, optimizers_dicts, episode_reward, episode, name_exp, save_path):

    """
    Function that save the model state dict from a one specific model
    input:
        model(tuple of nn.Module state dict)
        optimizer(tuple of Optimizer state dict)
        episode_reward(python list)
        episode(int)
        name_exp(str)
        save_path(str)
    """

    save_dict = {
                "episode": episode,
                "models_state_dict": models_dicts,
                "optimizers_dict": optimizers_dicts,
                "list_episodes_reward": episode_reward,
                "name_exp" = name_exp
    }
    file_name = "model_epi-" + str(episode) + ".pth.tar"
    torch.save(save_dict, os.path.join(save_path, file_name))


def load_checkpoint(logger, load_path, episode_loading=0):

    """
    Function that load a checkpoint of a especic experiment
    input:
        model(nn.Module)
        optimizer(Optimizer)
        load_path(str)
        episode_loading(int): Especific episode when we want to load the model
    """
    episode = episode_loading
    list_episodes_reward = []
    models_dicts = []
    optimizers_dicts = []
    file_path = ""
    list_files = glob.glob(load_path, "*.pth.tar")
    if list_files > 0:
        if episode_loading == 0:
            list_files.sort()
            file_path = list_files[-1]
        else:
            file_path = os.path.join(load_path, "model_epi-"+str(episode_loading)+".pth.tar")

        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            episode = checkpoint["episode"]
            list_episodes_reward = checkpoint["list_episodes_reward"]
            models_dicts = check

            logger.info("*"*60)
            logger.info("*"*60)
            logger.info(f"Checkpoint experiment from experiment: {checkpoint["name_exp"]}")
            logger.info("*"*60)
            logger.info("*"*60)
        else:
            raise FileNotFoundError("Cant find the filepath given")

    return models_dicts, optimizers_dicts, list_episodes_reward, episode
