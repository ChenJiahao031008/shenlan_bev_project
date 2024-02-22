docker run -itd \
    --gpus all \
    --shm-size=2g \
    --network=host \
    -v /home/idriver/learning/data:/home/idriver/learning/data \
    -v /home/idriver/workspace/my_project/bev_docker/my_work:/my_work \
    --name my-bev-docker \
    bev-playground:v1.0 bash
