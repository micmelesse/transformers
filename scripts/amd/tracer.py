import os
import torch
import socket
import shutil


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False


def print_var_name(variable):
    for name in globals():
        if eval(name) == variable:
            return name


def init_hostdir():
    host_name = socket.gethostname()
    if os.path.exists(host_name):
        print(host_name, "exists")
        # if yes_or_no("Do you want to delete existing folder, " + host_name):
        print("deleting existing", host_name, "directory")
        shutil.rmtree(host_name)
        os.mkdir(host_name)
    else:
        os.mkdir(host_name)


def save_tensor(tensor_to_save, name=None):
    host_name = socket.gethostname()
    if name == None:
        name = print_var_name(tensor_to_save)

    # device_id = tensor_to_save.get_device()
    device_id = 0

    torch.save(tensor_to_save, os.path.join(host_name, str(name) + "_" + str(device_id) + '.pt'))
