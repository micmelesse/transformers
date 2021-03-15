# rm -rf DeepSpeed
# git clone https://github.com/ROCmSoftwarePlatform/DeepSpeed
cd DeepSpeed/docker
docker build -f Dockerfile.rocm .

# cd DeepSpeed
# bash install.sh -r -t
# DS_BUILD_CUDA=0 DS_BUILD_LAMB=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_CPU_ADAM=1 bash install.sh -r -d

# python -m pip install -U pip #upgrade pip
# STAGE_DIR=/tmp
# git clone https://github.com/ROCmSoftwarePlatform/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
# cd ${STAGE_DIR}/DeepSpeed &&
#     git checkout . &&
#     git checkout master &&
#     ./install.sh --third_party_only --allow_sudo &&
#     DS_BUILD_CUDA=0 DS_BUILD_LAMB=1 ./install.sh --allow_sudo
# rm -rf ${STAGE_DIR}/DeepSpeed
# cd ~ && python -c "import deepspeed; print(deepspeed.__version__)"
