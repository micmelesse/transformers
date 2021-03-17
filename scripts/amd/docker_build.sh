# set path
DOCKERFILE_PATH=scripts/amd/huggingface_zero.Dockerfile

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f 1 -d '.')

# build docker
docker build -f $DOCKERFILE_PATH -t $DOCKERIMAGE_NAME .