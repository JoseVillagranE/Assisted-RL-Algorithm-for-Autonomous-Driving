import os
from config.config import config

def set_working_dir():

    cwd = os.getcwd()
    nwd = cwd[:cwd.find(config.project) + len(config.project)]
    os.chdir(nwd)
    return nwd
