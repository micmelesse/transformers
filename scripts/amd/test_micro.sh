cd ~
git clone https://github.com/ROCmSoftwarePlatform/pytorch-micro-benchmarking
cd pytorch-micro-benchmarking
pip install torchvision --no-dependencies
HSA_FORCE_FINE_GRAIN_PCIE=1 python3.6 micro_benchmarking_pytorch.py --network resnet50 --batch-size 2048 --fp16 1 --dataparallel --device_ids 0,1,2,3,4,5,6,7 --dist-backend nccl