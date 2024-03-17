# Project 4 Transformer

## 一、推理脚本

```bash
python tools/create_data.py nuscenes \
	--root-path /data/nuscenes \
     --out-dir /data/nuscenes \
     --extra-tag nuscenes \
     --version v1.0 \
     --canbus /data/nuscenes

./tools/dist_test.sh \
    ./projects/configs/bevformer/bevformer_tiny.py \
    ./ckpts/bevformer_tiny_epoch_24.pth \
    1
```

## 二、推理结果

```
Calculating metrics...
Saving metrics to: test/bevformer_tiny/Sun_Mar_17_17_32_07_2024/pts_bbox
mAP: 0.2519
mATE: 0.8984
mASE: 0.2931
mAOE: 0.6507
mAVE: 0.6564
mAAE: 0.2160
NDS: 0.3545
Eval time: 283.7s

Per-class results:
Object Class    AP  ATE ASE AOE AVE AAE
car 0.457   0.650   0.162   0.130   0.571   0.216
truck   0.192   0.885   0.237   0.225   0.570   0.232
bus 0.234   0.972   0.234   0.184   1.469   0.345
trailer 0.066   1.215   0.285   0.725   0.429   0.073
construction_vehicle    0.058   1.081   0.495   1.516   0.233   0.365
pedestrian  0.332   0.814   0.304   0.933   0.694   0.258
motorcycle  0.214   0.871   0.281   0.825   0.829   0.198
bicycle 0.203   0.826   0.292   1.172   0.456   0.043
traffic_cone    0.384   0.798   0.342   nan nan nan
barrier 0.379   0.871   0.299   0.147   nan nan
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.087, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.3341, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.6275, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.781, 'pts_bbox_NuScenes/car_trans_err': 0.6502, 'pts_bbox_NuScenes/car_scale_err': 0.1618, 'pts_bbox_NuScenes/car_orient_err': 0.13, 'pts_bbox_NuScenes/car_vel_err': 0.5709, 'pts_bbox_NuScenes/car_attr_err': 0.2157, 'pts_bbox_NuScenes/mATE': 0.8984, 'pts_bbox_NuScenes/mASE': 0.2931, 'pts_bbox_NuScenes/mAOE': 0.6507, 'pts_bbox_NuScenes/mAVE': 0.6564, 'pts_bbox_NuScenes/mAAE': 0.216, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0018, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.071, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.2561, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.4395, 'pts_bbox_NuScenes/truck_trans_err': 0.8854, 'pts_bbox_NuScenes/truck_scale_err': 0.2368, 'pts_bbox_NuScenes/truck_orient_err': 0.2249, 'pts_bbox_NuScenes/truck_vel_err': 0.5704, 'pts_bbox_NuScenes/truck_attr_err': 0.2316, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0001, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0642, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.1679, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 1.081, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.4947, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.5155, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.233, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.3652, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.0553, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.3178, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.5612, 'pts_bbox_NuScenes/bus_trans_err': 0.9718, 'pts_bbox_NuScenes/bus_scale_err': 0.2335, 'pts_bbox_NuScenes/bus_orient_err': 0.1836, 'pts_bbox_NuScenes/bus_vel_err': 1.4693, 'pts_bbox_NuScenes/bus_attr_err': 0.3446, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.0505, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.215, 'pts_bbox_NuScenes/trailer_trans_err': 1.2154, 'pts_bbox_NuScenes/trailer_scale_err': 0.2846, 'pts_bbox_NuScenes/trailer_orient_err': 0.7249, 'pts_bbox_NuScenes/trailer_vel_err': 0.4293, 'pts_bbox_NuScenes/trailer_attr_err': 0.0726, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.0119, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.2203, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.5854, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.6972, 'pts_bbox_NuScenes/barrier_trans_err': 0.8708, 'pts_bbox_NuScenes/barrier_scale_err': 0.2995, 'pts_bbox_NuScenes/barrier_orient_err': 0.1466, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0025, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.0959, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.2882, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.469, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.871, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.2814, 'pts_bbox_NuScenes/motorcycle_orient_err': 0.8252, 'pts_bbox_NuScenes/motorcycle_vel_err': 0.8286, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.1977, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.007, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.1162, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.2902, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.398, 'pts_bbox_NuScenes/bicycle_trans_err': 0.8264, 'pts_bbox_NuScenes/bicycle_scale_err': 0.2919, 'pts_bbox_NuScenes/bicycle_orient_err': 1.1723, 'pts_bbox_NuScenes/bicycle_vel_err': 0.4561, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0427, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0249, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.198, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.4598, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.6461, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.8143, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.3041, 'pts_bbox_NuScenes/pedestrian_orient_err': 0.933, 'pts_bbox_NuScenes/pedestrian_vel_err': 0.6935, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.2578, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.0338, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.2566, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.5584, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.6859, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.7975, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.3425, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.35449179552635024, 'pts_bbox_NuScenes/mAP': 0.251880970171981}
```

