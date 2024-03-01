docker run -itd \
    --gpus all \
    --shm-size=2g \
    --network=host \
    -v /media/idriver/My\ Passport/dataset:/data \
    -v /home/idriver/workspace/my_work/shenlan_bev_project/my_work:/my_work \
    --name my-bev-docker \
    bev-playground:v1.0 bash
