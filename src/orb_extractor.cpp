#include "orb_extractor.h"

#include <algorithm>

namespace mini_orb_slam
{

bool ORBExtractor::loadParams(ros::NodeHandle& nh)
{
    nh.param("features_num", features_num_, features_num_);
    nh.param("scale_factor", scale_factor_, scale_factor_);
    nh.param("levels_num", levels_num_, levels_num_);
    nh.param("init_th_fast", init_th_fast_, init_th_fast_);
    nh.param("min_th_fast", min_th_fast_, min_th_fast_);

    const auto create_orb = [this](int feature_count)
    {
        return cv::ORB::create(feature_count,
                               scale_factor_,
                               levels_num_,
                               31,
                               0,
                               2,
                               cv::ORB::HARRIS_SCORE,
                               31,
                               init_th_fast_);
    };

    orb_ = create_orb(features_num_);
    initialization_orb_ = create_orb(std::max(1, 2 * features_num_));

    if (orb_ == nullptr || initialization_orb_ == nullptr)
    {
        ROS_ERROR("Failed to create ORB feature extractor.");
        return false;
    }

    ROS_INFO("ORB feature extractor parameters loaded: features_num=%d, scale_factor=%.2f, levels_num=%d",
             features_num_, scale_factor_, levels_num_);

    return true;
}

void ORBExtractor::extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
                           cv::Mat& descriptors, bool initialization) const
{
    keypoints.clear();
    descriptors.release();

    if (img.empty())
    {
        ROS_WARN("Input image is empty. No features extracted.");
        return;
    }

    const cv::Ptr<cv::ORB>& extractor = initialization ? initialization_orb_ : orb_;
    if (extractor == nullptr)
    {
        ROS_ERROR("ORB extractor is not initialized. Call loadParams() first.");
        return;
    }

    extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

} // namespace mini_orb_slam
