nvidia-docker pull carlasim/carla:latest

GPU=$1
shift


nvidia-docker run -it \
              --net=host \
              --gpus $GPU \
              carlasim/carla:latest \
              /bin/bash CarlaUE4.sh Town02\
              -carla-server \
              -fps=10 \
              -carla-no-hud \
              -world-port=2000;
