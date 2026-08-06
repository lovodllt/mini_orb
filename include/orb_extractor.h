#ifndef MINI_ORB_SLAM_INCLUDE_ORB_EXTRACTOR_H_
#define MINI_ORB_SLAM_INCLUDE_ORB_EXTRACTOR_H_

#include <vector>

#include <opencv4/opencv2/features2d.hpp>
#include <ros/ros.h>

namespace mini_orb_slam
{

class ORBExtractor
{
public:
    ORBExtractor() = default;

    bool loadParams(ros::NodeHandle& nh);
    void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    int getFeaturesNum() const { return features_num_; }
    int getLevelsNum() const { return levels_num_; }
    double getScaleFactor() const { return scale_factor_; }

private:
    int features_num_{1000};
    int levels_num_{8};
    double scale_factor_{1.2};
    int init_th_fast_{20};
    int min_th_fast_{7};

    cv::Ptr<cv::ORB> orb_;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_ORB_EXTRACTOR_H_
