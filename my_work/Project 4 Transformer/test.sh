# python tools/create_data.py nuscenes \
#     --root-path /data/nuscenes \
#     --out-dir /data/nuscenes \
#     --extra-tag nuscenes \
#     --version v1.0 \
#     --canbus /data/nuscenes

./tools/dist_test.sh \
    ./projects/configs/bevformer/bevformer_tiny.py \
    ./ckpts/bevformer_tiny_epoch_24.pth \
    1
