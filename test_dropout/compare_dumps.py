import os
import torch
import argparse


def get_tensors(dump_dir):
    tensors = {}
    for subdir, _, files in os.walk(dump_dir):
        for file in files:
            try:
                tensors[file] = torch.load(os.path.join(
                    subdir, file), map_location=torch.device('cpu'))
            except:
                print("Failed to load", subdir, file)
                pass

    return tensors


def check_tensor_finite(tensor):
    return torch.isfinite(tensor).all()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('amd_dump')
parser.add_argument('nv_dump')
args = parser.parse_args()

amd_dump = get_tensors(args.amd_dump)
nv_dump = get_tensors(args.nv_dump)
print("AMD tensor dump path:", args.amd_dump)
print("NV tensor dump path:", args.nv_dump)

for tensor_name, amd_tensor in amd_dump.items():
    print(tensor_name, end='')
    nv_tensor = nv_dump[tensor_name]

    diff = torch.dist(
        amd_tensor.float(), nv_tensor.float()).item()
    print(", diff of", diff, end='')

    # if not check_tensor_finite(nv_tensor):
        # print(", not finite on NV", end='')

    # if not check_tensor_finite(amd_tensor):
        # print(", not finite on AMD", end='')

    print("")
