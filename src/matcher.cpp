#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "matcher.h"

namespace mini_orb_slam
{

namespace 
{

constexpr int kRotationHistBinNum = 30;

int computeRotationBin(float angle_diff_deg)
{
    const float factor = static_cast<float>(kRotationHistBinNum) / 360.0f;
    int bin = cvRound(angle_diff_deg * factor);
    if (bin == kRotationHistBinNum)
        bin = 0;

    return std::max(0, std::min(bin, kRotationHistBinNum - 1));
}

void computeTopThreeBins(const std::array<std::vector<int>, kRotationHistBinNum>& rot_hist, 
                         int& idx1, int& idx2, int& idx3)
{
    idx1 = -1;
    idx2 = -1;
    idx3 = -1;

    std::size_t max1 = 0;
    std::size_t max2 = 0;
    std::size_t max3 = 0;

    for (int i = 0; i < kRotationHistBinNum; i++)
    {
        const std::size_t s = rot_hist[i].size();

        if (s > max1)
        {
            max3 = max2;
            idx3 = idx2;
            max2 = max1;
            idx2 = idx1;
            max1 = s;
            idx1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            idx3 = idx2;
            max2 = s;
            idx2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            idx3 = i;
        }
    }

    if (max2 < 0.1 * static_cast<float>(max1))
    {
        idx2 = -1;
        idx3 = -1;
    }
    else if (max3 < 0.1 * static_cast<float>(max1))
    {
        idx3 = -1;
    }
}

} // namespace

bool Matcher::loadParams(ros::NodeHandle& nh)
{
    nh.param("ratio_threshold", ratio_threshold_, ratio_threshold_);
    nh.param("max_hamming_distance", max_hamming_distance_, max_hamming_distance_);

    return true;
}

std::vector<std::pair<int, int>> Matcher::matchFrames(const Frame& ref_frame, const Frame& cur_frame) const
{
    if (!ref_frame.hasFeatures() || !cur_frame.hasFeatures())
    {
        ROS_WARN("One of the frames has no features. Skipping matching.");
        return {};
    }

    return matchDescriptors(ref_frame.getDescriptors(), cur_frame.getDescriptors());
}

std::vector<std::pair<int, int>> Matcher::matchFramesForInitialization(
    const Frame& ref_frame,
    const Frame& cur_frame,
    std::vector<cv::Point2f>& previous_matched,
    int window_size) const
{
    std::vector<std::pair<int, int>> match_indices;

    if (!ref_frame.hasFeatures() || !cur_frame.hasFeatures() || window_size <= 0)
        return match_indices;

    const std::vector<cv::KeyPoint>& ref_keypoints = ref_frame.getKeypoints();
    const std::vector<cv::KeyPoint>& cur_keypoints = cur_frame.getKeypoints();
    const cv::Mat& ref_descriptors = ref_frame.getDescriptors();
    const cv::Mat& cur_descriptors = cur_frame.getDescriptors();

    if (ref_keypoints.size() != static_cast<std::size_t>(ref_descriptors.rows) ||
        cur_keypoints.size() != static_cast<std::size_t>(cur_descriptors.rows))
    {
        return match_indices;
    }

    if (previous_matched.size() != ref_keypoints.size())
    {
        previous_matched.clear();
        previous_matched.reserve(ref_keypoints.size());
        for (const cv::KeyPoint& keypoint : ref_keypoints)
            previous_matched.push_back(keypoint.pt);
    }

    // ORB-SLAM2 logic reference: SearchForInitialization uses TH_LOW=50 and
    // a 0.9 nearest-neighbour ratio, independently of normal tracking's
    // descriptor acceptance policy.
    constexpr int kInitializationMaxHammingDistance = 50;
    constexpr float kInitializationRatio = 0.9f;

    std::vector<int> matched_ref_for_cur(cur_keypoints.size(), -1);
    std::vector<int> matched_distance_for_cur(cur_keypoints.size(),
                                              std::numeric_limits<int>::max());
    std::array<std::vector<int>, kRotationHistBinNum> rotation_hist;
    for (auto& bin : rotation_hist)
        bin.reserve(16);

    for (int ref_idx = 0; ref_idx < ref_descriptors.rows; ++ref_idx)
    {
        const cv::KeyPoint& ref_keypoint = ref_keypoints[ref_idx];
        const int level = ref_keypoint.octave;
        if (level != 0)
            continue;

        const std::vector<int> candidate_indices = cur_frame.getFeatureIndicesInArea(
            previous_matched[ref_idx], static_cast<float>(window_size), level, level);
        if (candidate_indices.empty())
            continue;

        int best_distance = std::numeric_limits<int>::max();
        int second_best_distance = std::numeric_limits<int>::max();
        int best_cur_idx = -1;

        for (const int cur_idx : candidate_indices)
        {
            if (cur_idx < 0 || cur_idx >= cur_descriptors.rows)
            {
                continue;
            }

            const int distance = static_cast<int>(cv::norm(
                ref_descriptors.row(ref_idx), cur_descriptors.row(cur_idx), cv::NORM_HAMMING));

            if (matched_distance_for_cur[cur_idx] <= distance)
                continue;

            if (distance < best_distance)
            {
                second_best_distance = best_distance;
                best_distance = distance;
                best_cur_idx = cur_idx;
            }
            else if (distance < second_best_distance)
            {
                second_best_distance = distance;
            }
        }

        if (best_cur_idx < 0 || best_distance > kInitializationMaxHammingDistance ||
            best_distance >= kInitializationRatio * second_best_distance)
        {
            continue;
        }

        const int previous_ref_idx = matched_ref_for_cur[best_cur_idx];
        if (previous_ref_idx >= 0)
        {
            for (int previous_cur_idx = 0;
                 previous_cur_idx < static_cast<int>(matched_ref_for_cur.size());
                 ++previous_cur_idx)
            {
                if (matched_ref_for_cur[previous_cur_idx] == previous_ref_idx)
                {
                    matched_ref_for_cur[previous_cur_idx] = -1;
                    matched_distance_for_cur[previous_cur_idx] =
                        std::numeric_limits<int>::max();
                    break;
                }
            }

            for (auto& bin : rotation_hist)
            {
                bin.erase(std::remove(bin.begin(), bin.end(), previous_ref_idx), bin.end());
            }
        }

        matched_ref_for_cur[best_cur_idx] = ref_idx;
        matched_distance_for_cur[best_cur_idx] = best_distance;

        float angle_diff = ref_keypoint.angle - cur_keypoints[best_cur_idx].angle;
        if (angle_diff < 0.0f)
            angle_diff += 360.0f;

        rotation_hist[computeRotationBin(angle_diff)].push_back(ref_idx);
    }

    int best_bin_1 = -1;
    int best_bin_2 = -1;
    int best_bin_3 = -1;
    computeTopThreeBins(rotation_hist, best_bin_1, best_bin_2, best_bin_3);

    std::vector<bool> keep_reference(ref_keypoints.size(), false);
    const auto mark_bin = [&rotation_hist, &keep_reference](int bin_idx)
    {
        if (bin_idx < 0)
            return;

        for (const int ref_idx : rotation_hist[bin_idx])
            keep_reference[ref_idx] = true;
    };

    mark_bin(best_bin_1);
    mark_bin(best_bin_2);
    mark_bin(best_bin_3);

    match_indices.reserve(ref_keypoints.size());
    for (int cur_idx = 0; cur_idx < static_cast<int>(matched_ref_for_cur.size()); ++cur_idx)
    {
        const int ref_idx = matched_ref_for_cur[cur_idx];
        if (ref_idx < 0 || !keep_reference[ref_idx])
            continue;

        match_indices.emplace_back(ref_idx, cur_idx);
        previous_matched[ref_idx] = cur_keypoints[cur_idx].pt;
    }

    return match_indices;
}

std::vector<std::pair<int, int>> Matcher::matchDescriptors(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors) const
{
    std::vector<std::pair<int, int>> match_indices;

    if (query_descriptors.empty() || train_descriptors.empty())
        return match_indices;

    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    matcher.knnMatch(query_descriptors, train_descriptors, knn_matches, 2);

    match_indices.reserve(knn_matches.size());

    for (const auto& candidates : knn_matches)
    {
        if (candidates.size() < 2)
            continue;

        const cv::DMatch& best_match = candidates[0];
        const cv::DMatch& second_match = candidates[1];

        if (!isMatchValid(best_match, second_match))
            continue;

        match_indices.emplace_back(best_match.queryIdx, best_match.trainIdx);
    }

    return match_indices;
}

std::vector<std::pair<int, int>> Matcher::matchFramesByBoW(const Frame& ref_frame, 
                                                           const Frame& cur_frame) const
{
    std::vector<std::pair<int, int>> match_indices;

    if (!ref_frame.hasFeatures() || !cur_frame.hasFeatures() ||
        !ref_frame.hasBoW() || !cur_frame.hasBoW())
    {
        return match_indices;
    }

    const FeatureVector& ref_fv = ref_frame.getFeatureVector();
    const FeatureVector& cur_fv = cur_frame.getFeatureVector();
    if (ref_fv.empty() || cur_fv.empty())
        return match_indices;

    const cv::Mat& ref_descriptors = ref_frame.getDescriptors();
    const cv::Mat& cur_descriptors = cur_frame.getDescriptors();
    const std::vector<cv::KeyPoint>& ref_keypoints = ref_frame.getKeypoints();
    const std::vector<cv::KeyPoint>& cur_keypoints = cur_frame.getKeypoints();

    std::vector<int> best_ref_for_cur(cur_descriptors.rows, -1);
    std::vector<int> best_dist_for_cur(cur_descriptors.rows, std::numeric_limits<int>::max());
    std::vector<float> best_angle_diff_for_cur(cur_descriptors.rows, -1.0f);

    auto ref_it = ref_fv.begin();
    auto cur_it = cur_fv.begin();

    while (ref_it != ref_fv.end() && cur_it != cur_fv.end())
    {
        if (ref_it->first < cur_it->first)
        {
            ref_it++;
            continue;
        }

        if (cur_it->first < ref_it->first)
        {
            cur_it++;
            continue;
        }

        const std::vector<unsigned int>& ref_indices = ref_it->second;
        const std::vector<unsigned int>& cur_indices = cur_it->second;

        for (const unsigned int ref_idx_u : ref_indices)
        {
            const int ref_idx = static_cast<int>(ref_idx_u);
            if (ref_idx < 0 || ref_idx >= ref_descriptors.rows ||
                ref_idx >= static_cast<int>(ref_keypoints.size()))
            {
                continue;
            }

            int best_cur_idx = -1;
            int best_dist = std::numeric_limits<int>::max();
            int second_best_dist = std::numeric_limits<int>::max();

            for (const unsigned int cur_idx_u : cur_indices)
            {
                const int cur_idx = static_cast<int>(cur_idx_u);
                if (cur_idx < 0 || cur_idx >= cur_descriptors.rows ||
                    cur_idx >= static_cast<int>(cur_keypoints.size()))
                {
                    continue;
                }

                const int dist = static_cast<int>(
                    cv::norm(ref_descriptors.row(ref_idx), 
                             cur_descriptors.row(cur_idx),
                             cv::NORM_HAMMING));
                
                if (dist < best_dist)
                {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_cur_idx = cur_idx;
                }
                else if (dist < second_best_dist)
                {
                    second_best_dist = dist;
                }
            }

            if (best_dist > max_hamming_distance_ || best_cur_idx < 0)
                continue;

            if (second_best_dist != std::numeric_limits<int>::max() &&
                best_dist >= ratio_threshold_ * second_best_dist)
            {
                continue;
            }

            if (best_dist < best_dist_for_cur[best_cur_idx])
            {
                float angle_diff = ref_keypoints[ref_idx].angle - cur_keypoints[best_cur_idx].angle;
                if (angle_diff < 0.0f)
                    angle_diff += 360.0f;

                best_dist_for_cur[best_cur_idx] = best_dist;
                best_ref_for_cur[best_cur_idx] = ref_idx;
                best_angle_diff_for_cur[best_cur_idx] = angle_diff;
            }
        }

        ref_it++;
        cur_it++;
    }

    std::array<std::vector<int>, kRotationHistBinNum> rot_hist;
    for (auto& bin : rot_hist)
        bin.reserve(16);

    for (int cur_idx = 0; cur_idx < static_cast<int>(best_ref_for_cur.size()); cur_idx++)
    {
        if (best_ref_for_cur[cur_idx] < 0 || best_angle_diff_for_cur[cur_idx] < 0.0f)
            continue;

        const int bin = computeRotationBin(best_angle_diff_for_cur[cur_idx]);
        rot_hist[bin].push_back(cur_idx);
    }

    int idx1, idx2, idx3;
    computeTopThreeBins(rot_hist, idx1, idx2, idx3);

    std::vector<bool> keep_match(best_ref_for_cur.size(), false);

    const auto mark_bin = [&rot_hist, &keep_match](int bin_idx)
    {
        if (bin_idx < 0)
            return;

        for (const int cur_idx : rot_hist[bin_idx])
            keep_match[cur_idx] = true;
    };

    mark_bin(idx1);
    mark_bin(idx2);
    mark_bin(idx3);

    match_indices.reserve(cur_descriptors.rows);

    for (int cur_idx = 0; cur_idx < best_ref_for_cur.size(); cur_idx++)
    {
        const int ref_idx = best_ref_for_cur[cur_idx];
        if (ref_idx >= 0 && keep_match[cur_idx])
            match_indices.emplace_back(ref_idx, cur_idx);
    }

    return match_indices;
}

void Matcher::buildDMatches(const std::vector<std::pair<int, int>>& match_indices, std::vector<cv::DMatch>& matches) const
{
    matches.clear();
    matches.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        cv::DMatch match;
        match.queryIdx = match_idx.first;
        match.trainIdx = match_idx.second;
        match.distance = 0.0f;
        matches.push_back(match);
    }
}

bool Matcher::isMatchValid(const cv::DMatch& best_match, const cv::DMatch& second_match) const
{
    if (best_match.distance > max_hamming_distance_)
        return false;

    if (best_match.distance >= ratio_threshold_ * second_match.distance)
        return false;

    return true;
}
 
}
