#include "orb_extractor.h"

namespace mini_orb_slam
{

bool ORBExtractor::loadParams(ros::NodeHandle& nh)
{
    nh.param("features_num", features_num_, features_num_);
    nh.param("scale_factor", scale_factor_, scale_factor_);
    nh.param("levels_num", levels_num_, levels_num_);
    nh.param("init_th_fast", init_th_fast_, init_th_fast_);
    nh.param("min_th_fast", min_th_fast_, min_th_fast_);

    orb_ = cv::ORB::create(features_num_,            // 保留特征点数
                           scale_factor_,            // 图像金字塔缩放比例
                           levels_num_,              // 图像金字塔总层数
                           31,                       // 边界宽度
                           0,                        // 金字塔开始层数（原图）
                           2,                        // 计算描述子时比较的点数
                           cv::ORB::HARRIS_SCORE, 
                           31,                       // 计算描述子的邻域窗口宽度
                           init_th_fast_);           // FAST 角点初始阈值

    if (orb_ == nullptr)
    {
        ROS_ERROR("Failed to create ORB feature extractor.");
        return false;
    }

    ROS_INFO("ORB feature extractor parameters loaded: features_num=%d, scale_factor=%.2f, levels_num=%d",
             features_num_, scale_factor_, levels_num_);

    return true;
}

void ORBExtractor::extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
{
    keypoints.clear();
    descriptors.release();

    if (img.empty())
    {
        ROS_WARN("Input image is empty. No features extracted.");
        return;
    }

    if (orb_ == nullptr)
    {
        ROS_ERROR("ORB extractor is not initialized. Call loadParams() first.");
        return;
    }

    orb_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

} // namespace mini_orb_slam