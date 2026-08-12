# Mini ORB-SLAM

面向 ROS1 Noetic 的单目视觉 SLAM 工程，使用 C++17、OpenCV、Eigen、g2o 和
DBoW2 实现完整的跟踪、局部建图、局部 BA、回环检测与位姿图优化链路。

## 项目亮点

- 完整单目 SLAM 闭环：ORB 特征、两视图初始化、运动模型跟踪、PnP + g2o 位姿
  优化、关键帧管理、三角化、局部 BA、融合、冗余关键帧剔除、BoW 回环和位姿图。
- ROS 工程化：图像与 `CameraInfo` 校准握手、发布端处理 ACK、轨迹导出、离线
  KITTI 播放器，支持可重复的端到端回归。
- Local Mapping 采用显式事务：前端只在映射结果提交并消费后接纳下一关键帧，避免
  关键帧顺序因线程调度窗口漂移而导致地图分叉。
- Local BA 使用不可变地图快照构图，g2o 完成求解后在地图版本校验通过时才原子提交；
  对异常重投影边执行观测删除、共视更新和位姿图测量刷新。
- 性能优化以剖析数据为依据：缓存 MapPoint 关键帧观测、复用描述子距离、消除 BA
  提交阶段的重复遍历，并通过 incident-edge 索引缩小共视约束维护范围。

## 系统架构

```text
Image + CameraInfo
        |
        v
Frontend: feature extraction, initialization, tracking, keyframe admission
        |
        v
Local Mapper: triangulation -> fusion -> Local BA -> culling -> commit
        |                                                    |
        v                                                    v
Map / covisibility graph <------------------------- pose-graph constraints
        |
        v 
Loop Closer: BoW retrieval -> geometric verification -> pose-graph optimization
```

- `frontend.cpp`：跟踪、初始化、关键帧准入和 Local Mapping 结果消费。
- `local_mapper.cpp`：局部建图事务、三角化、融合、关键帧和地图点剔除。
- `pose_optimizer.cpp`：PnP 位姿优化、Local BA、候选校验和原子提交。
- `map.cpp`：地图生命周期、共视关系与增量位姿图测量刷新。
- `loop_closer.cpp`：BoW 候选检索、几何验证和回环位姿图优化。
- `kitti_mono_publisher.cpp`：带 ACK 的 KITTI 单目数据集发布器。

## Build

### Dependencies

- Ubuntu + ROS1 Noetic
- C++17 / catkin tools
- OpenCV, Eigen3, Sophus, g2o
- DBoW2（项目内置）

```bash
cd /workspace/mini_orb
source /opt/ros/noetic/setup.bash
catkin build mini_orb_slam --no-status
source devel/setup.bash
```

词典文件 `vocabulary/ORBvoc.txt` 由 Git LFS 管理。首次构建前请确认其已被正确拉取。

## Run With A Camera

节点订阅单目图像和可选的 `sensor_msgs/CameraInfo`。启用 `CameraInfo` 时，系统会在
第一帧图像前锁定相机内参和畸变模型。

```bash
source /workspace/mini_orb/devel/setup.bash
roslaunch mini_orb_slam mini_orb.launch \
  camera_topic:=/camera/image_raw \
  camera_info_topic:=/camera/camera_info \
  use_camera_info:=true \
  camera_info_required:=true \
  trajectory_output_path:=/tmp/mini_orb_trajectory.tum
```

## Reproduce The KITTI Regression

先启动 SLAM 节点：

```bash
RESULT=/workspace/slam_data/results/kitti00_300f_interview_demo
mkdir -p "$RESULT"

source /workspace/mini_orb/devel/setup.bash
roslaunch mini_orb_slam mini_orb.launch \
  camera_topic:=/kitti/image_raw \
  camera_info_topic:=/kitti/camera_info \
  use_camera_info:=true \
  camera_info_required:=true \
  image_queue_size:=600 \
  trajectory_output_path:="$RESULT/trajectory.tum"
```

在另一终端启动数据集发布器：

```bash
source /workspace/mini_orb/devel/setup.bash
roslaunch mini_orb_slam kitti_mono.launch \
  dataset_root:=/workspace/slam_data/sequences/00/00 \
  max_frames:=300 \
  playback_rate:=10.0 \
  playback_speed:=1.0 \
  wait_for_subscribers:=true \
  wait_for_processed_frames:=true \
  processed_frame_wait_timeout_sec:=240 \
  max_in_flight_frames:=10
```

使用项目脚本评估轨迹：

```bash
python3 src/mini_orb_slam/scripts/evaluate_kitti_outputs.py \
  --poses /workspace/slam_data/data_odometry_poses/dataset/poses/00.txt \
  --times /workspace/slam_data/sequences/00/times.txt \
  --mini "$RESULT/trajectory.tum" \
  --orb2 /workspace/slam_data/results/kitti00_300f_P2-KITTI-ORB2-R02-timing-baseline/orb2_trajectory.tum \
  --output-prefix "$RESULT/evaluation/kitti00_vs_orb2"
```

## Engineering Notes

- 地图更新以 `Map::getVersion()` 为提交前提，避免 BA 在旧快照上求解后覆盖新事务。
- Local BA 只刷新受影响关键帧的非回环位姿图边；回环约束保持独立，由回环线程全量维护。
- 回归验收同时检查 ACK、轨迹行数、LOST/reset、KF/MP、ATE、RPE 和分阶段耗时，避免只看
  单项性能或只针对某个场景调参。
