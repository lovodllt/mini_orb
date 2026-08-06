#ifndef MINI_ORB_SLAM_INCLUDE_FEATURE_H_
#define MINI_ORB_SLAM_INCLUDE_FEATURE_H_

#include <memory>
#include <mutex>
#include <opencv4/opencv2/features2d.hpp>

namespace mini_orb_slam
{

class Frame;
class MapPoint;

class Feature
{
public:
    Feature() = default;
    Feature(const std::shared_ptr<Frame>& frame, const cv::KeyPoint& keypoint, int feature_idx = -1)
        : frame_(frame), keypoint_(keypoint), feature_idx_(feature_idx) {}

    const cv::KeyPoint& getKeyPoint() const { return keypoint_; }
    void setKeyPoint(const cv::KeyPoint& keypoint) { keypoint_ = keypoint; }

    int getLevel() const { return keypoint_.octave; }

    int getFeatureIdx() const { return feature_idx_; }
    void setFeatureIdx(int feature_idx) { feature_idx_ = feature_idx; }

    bool isOutlier() const { return is_outlier_; }
    void setOutlier(bool is_outlier) { is_outlier_ = is_outlier; }

    bool hasMapPoint() const
    {
        std::lock_guard<std::mutex> lock(map_point_mutex_);
        return !map_point_.expired();
    }

    std::shared_ptr<MapPoint> getMapPoint() const
    {
        std::lock_guard<std::mutex> lock(map_point_mutex_);
        return map_point_.lock();
    }

    void setMapPoint(const std::shared_ptr<MapPoint>& map_point) 
    {
        std::lock_guard<std::mutex> lock(map_point_mutex_); 
        map_point_ = map_point; 
    }

    std::shared_ptr<Frame> getFrame() const { return frame_.lock(); }

private:
    std::weak_ptr<Frame> frame_;
    std::weak_ptr<MapPoint> map_point_;
    cv::KeyPoint keypoint_;
    int feature_idx_{-1};
    
    bool is_outlier_{false};

    mutable std::mutex map_point_mutex_;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_FEATURE_H_
