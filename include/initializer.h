#ifndef MINI_ORB_SLAM_INCLUDE_INITIALIZER_H_
#define MINI_ORB_SLAM_INCLUDE_INITIALIZER_H_

#include <memory>
#include <vector>

#include <opencv4/opencv2/core.hpp>

#include "camera.h"
#include "common.h"
#include "frame.h"

namespace mini_orb_slam
{

class Initializer
{
public:
    explicit Initializer(const std::shared_ptr<Camera>& camera);

    PoseRecoveryResult recoverPoseFromFrames(
        const std::shared_ptr<Frame>& ref_frame,
        const std::shared_ptr<Frame>& cur_frame,
        const std::vector<std::pair<int, int>>& match_indices) const;

    TriangulationResult triangulateFromPose(const PoseRecoveryResult& pose_result) const;

    TriangulationResult triangulateFromMatchedFrames(
        const std::shared_ptr<Frame>& ref_frame,
        const std::shared_ptr<Frame>& cur_frame,
        const std::vector<std::pair<int, int>>& match_indices) const;


private:
    std::shared_ptr<Camera> camera_;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_INITIALIZER_H_
