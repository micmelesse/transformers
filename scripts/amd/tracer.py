import os
import torch
import socket
import shutil


def print_var_name(variable):
    for name in globals():
        if eval(name) == variable:
            return name


def init_hostdir():
    global host_name
    host_name = socket.gethostname()
    if os.path.exists(host_name):
        shutil.rmtree(host_name)
    os.mkdir(host_name)


def save_tensor(tensor_to_save, name=None):
    if name == None:
        name = print_var_name(tensor_to_save)

    device_id = tensor_to_save.get_device()

    torch.save(tensor_to_save, os.path.join(host_name, str(name) +"_"+ str(device_id) + '.pt'))
