#ifndef MINI_ORB_SLAM_INCLUDE_MATCHER_H_
#define MINI_ORB_SLAM_INCLUDE_MATCHER_H_

#include <utility>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <ros/ros.h>

#include "frame.h"

namespace mini_orb_slam
{

class Matcher
{
public:
    Matcher() = default;

    bool loadParams(ros::NodeHandle& nh);

    int getMaxHammingDistance() const { return max_hamming_distance_; }
    float getRatioThreshold() const { return ratio_threshold_; }

    std::vector<std::pair<int, int>> matchFrames(
        const Frame& ref_frame, const Frame& cur_frame) const;

    std::vector<std::pair<int, int>> matchFramesForInitialization(
        const Frame& ref_frame,
        const Frame& cur_frame,
        std::vector<cv::Point2f>& previous_matched,
        int window_size = 100) const;

    std::vector<std::pair<int, int>> matchDescriptors(
        const cv::Mat& query_descriptors, const cv::Mat& train_descriptors) const;

    std::vector<cv::DMatch> matchDescriptorsWithDistance(
        const cv::Mat& query_descriptors, const cv::Mat& train_descriptors) const;

    std::vector<std::pair<int, int>> matchFramesByBoW(const Frame& ref_frame,
                                                     const Frame& cur_frame) const;

    void buildDMatches(const std::vector<std::pair<int, int>>& match_indices, 
                       std::vector<cv::DMatch>& matches) const;

private:
    bool isMatchValid(const cv::DMatch& best_match, const cv::DMatch& second_match) const;

    float ratio_threshold_{0.8f};
    int max_hamming_distance_{50};
};

}  // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_MATCHER_H_
