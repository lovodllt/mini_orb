#include <unordered_set>
#include <algorithm>
#include <limits.h>
#include <cmath>
#include <sstream>
#include <ros/ros.h>

#include "local_mapper.h"

namespace mini_orb_slam
{

namespace
{

double pointNorm(const cv::Point3d& p)
{
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

double dotPoint(const cv::Point3d& a, const cv::Point3d& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool computeFundamentalMatrix(const std::shared_ptr<Frame>& ref_keyframe,
                              const std::shared_ptr<Frame>& cur_keyframe,
                              cv::Matx33d& F_21)
{
    if (ref_keyframe == nullptr || cur_keyframe == nullptr ||
        ref_keyframe->getCamera() == nullptr)
    {
        return false;
    }

    cv::Mat R_ref, t_ref;
    cv::Mat R_cur, t_cur;

    ref_keyframe->copyPose(R_ref, t_ref);
    cur_keyframe->copyPose(R_cur, t_cur);

    if (R_ref.empty() || t_ref.empty() || R_cur.empty() || t_cur.empty())
        return false;

    cv::Mat K;
    ref_keyframe->getCamera()->getK().convertTo(K, CV_64F);
    if (K.rows != 3 || K.cols != 3)
        return false;

    const cv::Mat R_21 = R_cur * R_ref.t();
    const cv::Mat t_21 = t_cur - R_21 * t_ref;
    if (cv::norm(t_21) <= 1e-6)
        return false;

    const double tx = t_21.at<double>(0, 0);
    const double ty = t_21.at<double>(1, 0);
    const double tz = t_21.at<double>(2, 0);
    const cv::Mat t_21_skew = (cv::Mat_<double>(3, 3) <<
        0.0, -tz, ty,
        tz, 0.0, -tx,
        -ty, tx, 0.0);

    const cv::Mat K_inv = K.inv();
    const cv::Mat F = K_inv.t() * t_21_skew * R_21 * K_inv;
    if (!cv::checkRange(F))
        return false;

    cv::Mat F_64;
    F.convertTo(F_64, CV_64F);
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
            F_21(row, col) = F_64.at<double>(row, col);
    }

    return true;
}

bool passesEpipolarConstraint(const cv::Matx33d& F_21,
                              const cv::Point2f& ref_point,
                              const cv::Point2f& cur_point,
                              int cur_level,
                              double scale_factor)
{
    const cv::Vec3d line = F_21 * cv::Vec3d(ref_point.x, ref_point.y, 1.0);
    const double line_norm2 = line[0] * line[0] + line[1] * line[1];
    if (line_norm2 <= 1e-12)
        return false;

    const double numerator = line[0] * cur_point.x + line[1] * cur_point.y + line[2];
    const double distance2 = numerator * numerator / line_norm2;
    const int level = std::max(0, cur_level);
    const double valid_scale = scale_factor > 1.0 ? scale_factor : 1.2;
    const double sigma2 = std::pow(valid_scale, 2 * level);

    return distance2 <= 3.84 * sigma2;
}

bool passesScaleConsistency(const std::shared_ptr<Frame>& ref_keyframe,
                            const std::shared_ptr<Frame>& cur_keyframe,
                            const cv::Point3d& point_world,
                            int ref_level,
                            int cur_level,
                            double scale_factor)
{
    if (ref_keyframe == nullptr || cur_keyframe == nullptr)
        return false;
    
    const cv::Point3d ref_view = point_world - ref_keyframe->getCameraCenter();
    const cv::Point3d cur_view = point_world - cur_keyframe->getCameraCenter();

    const double dist_ref = pointNorm(ref_view);
    const double dist_cur = pointNorm(cur_view);

    if (!std::isfinite(dist_ref) || !std::isfinite(dist_cur) ||
        dist_ref <= 1e-6 || dist_cur <= 1e-6)
    {
        return false;
    }

    const double valid_scale = scale_factor > 1.0 ? scale_factor : 1.2;
    const double ratio_dist = dist_cur / dist_ref;
    const double ratio_level = std::pow(valid_scale, static_cast<double>(cur_level - ref_level));
    const double ratio_factor = 1.5 * valid_scale;

    return ratio_dist * ratio_factor >= ratio_level &&
           ratio_dist <= ratio_level * ratio_factor;
}

bool getFeatureDescriptor(const std::shared_ptr<Feature>& feature,
                          cv::Mat& descriptor)
{
    descriptor.release();

    if (feature == nullptr)
        return false;

    const std::shared_ptr<Frame> frame = feature->getFrame();
    if (frame == nullptr || frame->getDescriptors().empty())
        return false;

    const int feature_idx = feature->getFeatureIdx();
    if (feature_idx < 0 || feature_idx >= static_cast<int>(frame->getFeatures().size()))
        return false;

    descriptor = frame->getDescriptors().row(feature_idx).clone();
    return true;
}

bool getMapPointDescriptor(const std::shared_ptr<MapPoint>& map_point,
                          cv::Mat& descriptor)
{
    descriptor.release();

    if (map_point == nullptr || map_point->isBad())
        return false;

    if (map_point->hasRepresentativeDescriptor())
    {
        descriptor = map_point->getRepresentativeDescriptor().clone();
        return true;
    }

    return getFeatureDescriptor(map_point->selectRefFeatureCandidate(), descriptor);
}

constexpr int kTriangulationRotationHistBinNum = 30;

int computeTriangulationRotationBin(float angle_diff_deg)
{
    const float factor = static_cast<float>(kTriangulationRotationHistBinNum) / 360.0f;
    int bin = cvRound(angle_diff_deg * factor);
    if (bin == kTriangulationRotationHistBinNum)
        bin = 0;

    return std::max(0, std::min(bin, kTriangulationRotationHistBinNum - 1));
}

void computeTopThreeTrianglationBins(
    const std::array<std::vector<int>, kTriangulationRotationHistBinNum>& rot_hist,
    int& idx1,
    int& idx2,
    int& idx3)
{
    idx1 = -1;
    idx2 = -1;
    idx3 = -1;

    std::size_t max1 = 0;
    std::size_t max2 = 0;
    std::size_t max3 = 0;

    for (int i = 0; i < kTriangulationRotationHistBinNum; i++)
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

    if (max2 < 0.1 * max1)
    {
        idx2 = -1;
        idx3 = -1;
    }
    else if (max3 < 0.1 * max1)
    {
        idx3 = -1;
    }
}

std::vector<std::pair<int, int>> collectTriangulationMatches(
    const Matcher& matcher,
    const std::shared_ptr<Frame>& ref_keyframe,
    const std::shared_ptr<Frame>& cur_keyframe,
    const cv::Matx33d& F_21,
    double scale_factor)
{
    std::vector<std::pair<int, int>> candidate_matches;

    if (ref_keyframe == nullptr || cur_keyframe == nullptr ||
        !ref_keyframe->hasFeatures() || !cur_keyframe->hasFeatures())
    {
        return candidate_matches;
    }

    const std::vector<std::pair<int, int>> raw_matches = 
        matcher.matchFrames(*ref_keyframe, *cur_keyframe);
    if (raw_matches.empty())
        return candidate_matches;

    const std::vector<std::shared_ptr<Feature>>& ref_features = ref_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_keyframe->getFeatures();
    const std::vector<cv::KeyPoint> ref_keypoints = ref_keyframe->getKeypoints();
    const std::vector<cv::KeyPoint> cur_keypoints = cur_keyframe->getKeypoints();
    const cv::Mat& ref_descriptor = ref_keyframe->getDescriptors();
    const cv::Mat& cur_descroptor = cur_keyframe->getDescriptors();

    std::vector<int> best_ref_for_cur(cur_features.size(), -1);
    std::vector<int> best_dist_for_cur(cur_features.size(), std::numeric_limits<int>::max());
    std::vector<float> best_angle_diff_for_cur(cur_features.size(), -1.0f);

    for (const auto& match_idx : raw_matches)
    {
        const int ref_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (ref_idx < 0 || ref_idx >= ref_features.size() ||
            cur_idx < 0 || cur_idx >= cur_features.size() ||
            ref_idx >= ref_descriptor.rows || cur_idx >= cur_descroptor.rows ||
            ref_idx >= ref_keypoints.size() || cur_idx >= cur_keypoints.size())
        {
            continue;
        }

        const std::shared_ptr<Feature>& ref_feature = ref_features[ref_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (ref_feature == nullptr || cur_feature == nullptr)
            continue;


        if (ref_feature->hasMapPoint() || cur_feature->hasMapPoint())
            continue;

        if (!passesEpipolarConstraint(F_21, 
                                      ref_feature->getKeyPoint().pt, 
                                      cur_feature->getKeyPoint().pt, 
                                      cur_feature->getLevel(), 
                                      scale_factor))
        {
            continue;
        }

        const int dist = static_cast<int>(
            cv::norm(ref_descriptor.row(ref_idx),
                     cur_descroptor.row(cur_idx),
                     cv::NORM_HAMMING));

        if (dist < best_dist_for_cur[cur_idx])
        {
            float angle_diff = ref_keypoints[ref_idx].angle - cur_keypoints[cur_idx].angle;
            if (angle_diff < 0.0f)
                angle_diff += 360.0f;

            best_dist_for_cur[cur_idx] = dist;
            best_ref_for_cur[cur_idx] = ref_idx;
            best_angle_diff_for_cur[cur_idx] = angle_diff;
        }
    }

    std::array<std::vector<int>, kTriangulationRotationHistBinNum> rot_hist;
    for (auto& bin : rot_hist)
        bin.reserve(16);

    for (int cur_idx = 0; cur_idx < best_ref_for_cur.size(); cur_idx++)
    {
        if (best_ref_for_cur[cur_idx] < 0 || best_angle_diff_for_cur[cur_idx] < 0.0f)
            continue;

        const int bin = computeTriangulationRotationBin(best_angle_diff_for_cur[cur_idx]);
        rot_hist[bin].push_back(cur_idx);
    }

    int idx1, idx2, idx3;
    computeTopThreeTrianglationBins(rot_hist, idx1, idx2, idx3);

    std::vector<bool> keep_match(best_ref_for_cur.size(), false);

    const auto mark_bin = [&rot_hist, &keep_match](int bin_idx)
    {
        if (bin_idx < 0 || bin_idx >= kTriangulationRotationHistBinNum)
            return;

        for (const int cur_idx : rot_hist[bin_idx])
            keep_match[cur_idx] = true;
    };

    mark_bin(idx1);
    mark_bin(idx2);
    mark_bin(idx3);

    candidate_matches.reserve(raw_matches.size());
    for (int cur_idx = 0; cur_idx < best_ref_for_cur.size(); cur_idx++)
    {
        if (best_ref_for_cur[cur_idx] < 0 || !keep_match[cur_idx])
            continue;

        candidate_matches.emplace_back(best_ref_for_cur[cur_idx], cur_idx);
    }

    return candidate_matches;
}

std::vector<std::shared_ptr<MapPoint>> collectUniqueMapPointsFromKeyframe(
    const std::shared_ptr<Frame>& keyframe)
{
    std::vector<std::shared_ptr<MapPoint>> map_points;
    if (keyframe == nullptr)
        return map_points;

    std::unordered_set<std::size_t> ids;
    ids.reserve(keyframe->getFeatureNum());
    map_points.reserve(keyframe->getFeatureNum());

    for (const auto& feature : keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!ids.insert(map_point->getId()).second)
            continue;

        map_points.push_back(map_point);
    }

    return map_points;
}

float computeFusionSearchRadius(int pred_level,
                                double scale_factor,
                                int levels_num)
{
    constexpr float kBaseSearchRadius = 15.0f;

    if (levels_num <= 1)
        return kBaseSearchRadius;

    const int clamped_level = std::max(0, std::min(pred_level, levels_num - 1));
    return kBaseSearchRadius * std::pow(scale_factor, clamped_level);
}

} // namespace

LocalMapper::LocalMapper(const std::shared_ptr<Initializer>& initializer,
                         const Matcher& matcher,
                         const std::shared_ptr<PoseOptimizer>& pose_optimizer,
                         double scale_factor,
                         int levels_num) :
    initializer_(initializer),
    matcher_(matcher),
    pose_optimizer_(pose_optimizer),
    scale_factor_(scale_factor > 1.0 ? scale_factor : 1.2),
    levels_num_(levels_num > 0 ? levels_num : 8) {}

LocalMapper::~LocalMapper()
{
    requestFinish();
    join();
}

void LocalMapper::start()
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (worker_started_)
        return;

    finish_requested_ = false;
    finished_ = false;
    stop_requested_ = false;
    stopped_ = false;
    accept_keyframes_ = true;
    worker_started_ = true;

    worker_thread_ = std::thread(&LocalMapper::run, this);
}

void LocalMapper::requestFinish()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        finish_requested_ = true;
        accept_keyframes_ = false;
    }

    queue_cv_.notify_all();
}

void LocalMapper::join()
{
    if (worker_thread_.joinable())
        worker_thread_.join();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    worker_started_ = false;
}

bool LocalMapper::insertKeyframe(const LocalMappingInput& input)
{
    if (input.map == nullptr || input.ref_keyframe == nullptr || input.cur_keyframe == nullptr)
        return false;

    if (!input.cur_keyframe->isKeyframe())
        return false;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!accept_keyframes_ || finish_requested_)
            return false;

        pending_keyframes_.push_back(input);
    }

    queue_cv_.notify_one();
    return true;
}

bool LocalMapper::hasPendingKeyframe() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !pending_keyframes_.empty();
}

bool LocalMapper::tryPopFinishedResult(LocalMappingOutput& output)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (finished_results_.empty())
        return false;

    output = finished_results_.front();
    finished_results_.pop_front();
    return true;
}

bool LocalMapper::acceptKeyframe() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return accept_keyframes_ &&
           !finish_requested_ &&
           !stop_requested_ &&
           pending_keyframes_.empty() &&
           !processing_new_keyframe_.load(std::memory_order_acquire);
}

bool LocalMapper::isStopped() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return stopped_;
}

bool LocalMapper::stopRequested() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return stop_requested_;
}

void LocalMapper::run()
{
    while (true)
    {
        LocalMappingInput input;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]()
            {
                return finish_requested_ || !pending_keyframes_.empty();
            });

            if (finish_requested_ && pending_keyframes_.empty())
                break;

            accept_keyframes_ = false;
            stopped_ = false;

            input = pending_keyframes_.front();
            pending_keyframes_.pop_front();
        }

        LocalMappingOutput output;
        output.input = input;
        output.result = processNewKeyframe(input.map,
                                           input.ref_keyframe,
                                           input.cur_keyframe,
                                           input.tracking_seed);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            finished_results_.push_back(std::move(output));
            accept_keyframes_ = true;
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stopped_ = true;
        finished_ = true;
        accept_keyframes_ = false;
    }
}

LocalMappingResult LocalMapper::processNewKeyframe(const std::shared_ptr<Map>& map, 
                                                   const std::shared_ptr<Frame>& ref_keyframe, 
                                                   const std::shared_ptr<Frame>& cur_keyframe,
                                                   const PnPResult& tracking_seed) const
{
    LocalMappingResult result;

    if (map == nullptr || ref_keyframe == nullptr || cur_keyframe == nullptr)
        return result;

    if (!cur_keyframe->isKeyframe())
        return result;

    bool expected = false;
    if (!processing_new_keyframe_.compare_exchange_strong(expected, 
                                                          true, 
                                                          std::memory_order_acquire, 
                                                          std::memory_order_relaxed))
    {
        return result;
    }

    struct ProcessingFlagGuard
    {
        std::atomic_bool& flag;

        ~ProcessingFlagGuard()
        {
            flag.store(false, std::memory_order_release);
        }
    } process_flag_guard{processing_new_keyframe_};

    std::lock_guard<std::mutex> map_lock(map->getMutex());

    if (recent_map_.lock() != map)
    {
        recent_map_ = map;
        recent_added_map_points_.clear();
        local_mapping_generation_ = 0;
    }

    map->addKeyframe(cur_keyframe);

    const std::size_t current_local_mapping_generation = local_mapping_generation_++;

    processCurrentKeyframeMapPoints(cur_keyframe);
    updateCovisibilityGraph(map, cur_keyframe);

    result.culled_map_point_num = cullMapPoints(map, current_local_mapping_generation);

    result.new_map_point_num = 
        growMapByKeyFrames(map, 
                           ref_keyframe, 
                           cur_keyframe, 
                           current_local_mapping_generation);

    updateCovisibilityGraph(map, cur_keyframe);

    const std::size_t fused_map_point_num = searchInNeighbors(map, cur_keyframe);

    updateCovisibilityGraph(map, cur_keyframe);
    if (ref_keyframe != cur_keyframe)
        updateCovisibilityGraph(map, ref_keyframe);

    constexpr std::size_t kMinKeyframesForLocalBA = 3;
    result.local_ba_called = (pose_optimizer_ != nullptr) &&
                             (map->getKeyframeNum() > kMinKeyframesForLocalBA);

    LocalBAResult local_ba_result;
    if (result.local_ba_called)
        local_ba_result = pose_optimizer_->optimizeLocalMap(map, cur_keyframe, tracking_seed);

    result.culled_keyframe_num = cullKeyframes(map, cur_keyframe);

    result.local_ba_solver_success = local_ba_result.solver_success;
    result.local_ba_success = local_ba_result.accepted;
    result.local_ba_rejected = local_ba_result.solver_success && !local_ba_result.accepted;

    result.local_ba_edge_num = local_ba_result.edge_num;
    result.local_ba_rejected_edge_num = local_ba_result.rejected_edge_num;
    result.local_ba_seed_reproj_error = local_ba_result.seed_reproj_error;
    result.local_ba_candidate_seed_reproj_error = local_ba_result.candidate_seed_reproj_error;

    result.success = (result.new_map_point_num > 0) ||
                     (fused_map_point_num > 0) ||
                     (result.culled_map_point_num > 0) ||
                     (result.culled_keyframe_num > 0) ||
                     result.local_ba_success;

    return result;
}

void LocalMapper::processCurrentKeyframeMapPoints(const std::shared_ptr<Frame>& keyframe) const
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return;

    std::unordered_set<std::size_t> updated_map_point_ids;

    for (const auto& feature : keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!updated_map_point_ids.insert(map_point->getId()).second)
            continue;

        refreshMapPointState(map_point);
    }
}

void LocalMapper::registerRecentMapPoint(const std::shared_ptr<MapPoint>& map_point) const
{
    if (map_point == nullptr || map_point->isBad())
        return;

    recent_added_map_points_.push_back(map_point);
}

void LocalMapper::refreshMapPointState(const std::shared_ptr<MapPoint>& map_point) const
{
    if (map_point == nullptr || map_point->isBad())
        return;

    map_point->updateViewStatistics(scale_factor_, levels_num_);
    map_point->updateRepresentativeDescriptor();
}

void LocalMapper::refreshMapPointStates(const std::vector<std::shared_ptr<MapPoint>>& map_points) const
{
    std::unordered_set<std::size_t> refreshed_ids;
    refreshed_ids.reserve(map_points.size());

    for (const auto& map_point : map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!refreshed_ids.insert(map_point->getId()).second)
            continue;

        refreshMapPointState(map_point);
    }
}

void LocalMapper::updateCovisibilityGraph(const std::shared_ptr<Map>& map, 
                                          const std::shared_ptr<Frame>& keyframe) const
{
    if (map == nullptr || keyframe == nullptr || !keyframe->isKeyframe())
        return;
    
    keyframe->updateConnections();

    const std::vector<std::shared_ptr<Frame>> neighbors = 
        keyframe->getConnectedKeyframes(5);

    for (const auto& neighbor : neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe())
            continue;

        neighbor->updateConnections();
        map->recordCovisibilityConstraints(neighbor);
    }

    map->recordCovisibilityConstraints(keyframe);
}

double LocalMapper::computeMedianSceneDepth(const std::shared_ptr<Frame>& keyframe) const
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return -1.0;

    cv::Mat R_cw;
    cv::Mat t_cw;
    keyframe->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return -1.0;

    std::vector<double> depths;
    depths.reserve(keyframe->getFeatureNum());

    for (const auto& feature : keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        const cv::Point3d& point_world = map_point->getPos();
        const cv::Mat point_w = 
            (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);
        const cv::Mat point_c = R_cw * point_w + t_cw;

        const double depth = point_c.at<double>(2, 0);
        if (std::isfinite(depth) && depth > 1e-6)
            depths.push_back(depth);
    }

    if (depths.empty())
        return -1.0;

    const std::size_t mid = depths.size() / 2;
    std::nth_element(depths.begin(), depths.begin() + mid, depths.end());
    return depths[mid];
}

std::size_t LocalMapper::countReliableMapPointFeatures(const std::shared_ptr<Frame>& keyframe) const
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return 0;

    std::size_t reliable_num = 0;
    std::unordered_set<std::size_t> seen_ids;
    seen_ids.reserve(keyframe->getFeatureNum());

    for (const auto& feature : keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!seen_ids.insert(map_point->getId()).second)
            continue;

        if (map_point->getKeyframeObservationCount() < 3)
            continue;

        reliable_num++;
    }

    return reliable_num;
}

bool LocalMapper::isTriangulationPartnerValid(const std::shared_ptr<Frame>& keyframe,
                                              const std::shared_ptr<Frame>& cur_keyframe) const
{
    if (keyframe == nullptr || cur_keyframe == nullptr)
        return false;

    if (!keyframe->isKeyframe() || !cur_keyframe->isKeyframe())
        return false;

    if (keyframe == cur_keyframe)
        return false;

    cv::Mat keyframe_R, keyframe_t;
    cv::Mat cur_R, cur_t;

    keyframe->copyPose(keyframe_R, keyframe_t);
    cur_keyframe->copyPose(cur_R, cur_t);

    if (keyframe_R.empty() || keyframe_t.empty() || cur_R.empty() || cur_t.empty())
        return false;

    constexpr int kMinTriangulationWeight = 15;
    constexpr std::size_t kMinReliableMapFeatures = 30;

    if (keyframe->getConnectionWeight(cur_keyframe->getId()) < kMinTriangulationWeight)
        return false;

    if (countReliableMapPointFeatures(keyframe) < kMinReliableMapFeatures)
        return false;

    return true;
}

std::vector<std::shared_ptr<Frame>> LocalMapper::collectTriangulationKeyframes(
    const std::shared_ptr<Frame>& ref_keyframe,
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<std::shared_ptr<Frame>> result;
    if (cur_keyframe == nullptr)
        return result;

    constexpr std::size_t kMaxTriangulationNeighbors = 10;

    std::unordered_set<std::size_t> added_ids;
    result.reserve(kMaxTriangulationNeighbors + 1);

    auto append_unique = [&](const std::shared_ptr<Frame>& keyframe, bool force_keep = false)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe())
            return;

        if (keyframe == cur_keyframe)
            return;

        if (!force_keep && !isTriangulationPartnerValid(keyframe, cur_keyframe))
            return;

        if (!added_ids.insert(keyframe->getId()).second)
            return;

        result.push_back(keyframe);
    };

    append_unique(ref_keyframe, true);

    const std::vector<std::shared_ptr<Frame>> covisible_keyframes = 
        cur_keyframe->getBestCovisibilityKeyframes(kMaxTriangulationNeighbors, 1);

    for (const auto& keyframe : covisible_keyframes)
    {
        append_unique(keyframe);
        if (result.size() >= kMaxTriangulationNeighbors)
            break;
    }

    return result;
}

std::size_t LocalMapper::growMapByKeyFrames(const std::shared_ptr<Map>& map, 
                                            const std::shared_ptr<Frame>& ref_keyframe, 
                                            const std::shared_ptr<Frame>& cur_keyframe,
                                            std::size_t current_local_mapping_generation) const
{   
    if (map == nullptr || ref_keyframe == nullptr || cur_keyframe == nullptr || initializer_ == nullptr)
        return 0;

    const std::vector<std::shared_ptr<Frame>> triangulation_keyframes = 
        collectTriangulationKeyframes(ref_keyframe, cur_keyframe);

    std::size_t created_map_points_num = 0;

    for (const auto& neighbpr_keyframe : triangulation_keyframes)
    {
        created_map_points_num += growMapByKeyFramePair(map, 
                                                        neighbpr_keyframe, 
                                                        cur_keyframe, 
                                                        current_local_mapping_generation);
    }

    return created_map_points_num;
}

std::size_t LocalMapper::growMapByKeyFramePair(const std::shared_ptr<Map>& map,
                                               const std::shared_ptr<Frame>& ref_keyframe,
                                               const std::shared_ptr<Frame>& cur_keyframe,
                                               std::size_t current_local_mapping_generation) const
{
    if (map == nullptr || ref_keyframe == nullptr || cur_keyframe == nullptr || initializer_ == nullptr)
        return 0;

    const double baseline = 
        cv::norm(ref_keyframe->getCameraCenter() - cur_keyframe->getCameraCenter());
    if (baseline < 0.05)
        return 0;

    const double median_depth = computeMedianSceneDepth(ref_keyframe);
    if (median_depth <= 1e-6)
        return 0;

    constexpr double kMinBaselineDepthRatio = 0.01;
    const double baseline_depth_ratio = baseline / median_depth;
    if (baseline_depth_ratio < kMinBaselineDepthRatio)
        return 0;

    cv::Matx33d F_21;
    if (!computeFundamentalMatrix(ref_keyframe, cur_keyframe, F_21))
        return 0;

    const std::vector<std::shared_ptr<Feature>>& ref_features = ref_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_keyframe->getFeatures();

    const std::vector<std::pair<int, int>> candidate_matches = 
        collectTriangulationMatches(matcher_, ref_keyframe, cur_keyframe, F_21, scale_factor_);

    if(candidate_matches.size() < 10)
        return 0;

    const TriangulationResult triangulation_result =
        initializer_->triangulateFromMatchedFrames(ref_keyframe, 
                                                   cur_keyframe, 
                                                   candidate_matches);

    if (triangulation_result.points_3d.empty())
        return 0;

    std::size_t created_map_points_num = 0;

    for (std::size_t i = 0; i < triangulation_result.points_3d.size(); i++)
    {
        const int ref_idx = triangulation_result.ref_feature_indices[i];
        const int cur_idx = triangulation_result.cur_feature_indices[i];

        if (ref_idx < 0 || ref_idx >= static_cast<int>(ref_features.size()) ||
            cur_idx < 0 || cur_idx >= static_cast<int>(cur_features.size()))
        {
            continue;
        }

        const std::shared_ptr<Feature>& ref_feature = ref_features[ref_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (ref_feature == nullptr || cur_feature == nullptr)
            continue;

        if (ref_feature->hasMapPoint() || cur_feature->hasMapPoint())
            continue;

        if (!passesScaleConsistency(ref_keyframe, 
                                    cur_keyframe, 
                                    triangulation_result.points_3d[i], 
                                    ref_feature->getLevel(), 
                                    cur_feature->getLevel(), 
                                    scale_factor_))
        {
            continue;
        }

        std::shared_ptr<MapPoint> map_point = 
            std::make_shared<MapPoint>(map->allocateMapPointId(), 
                                       triangulation_result.points_3d[i]);

        map_point->setFirstKeyframeId(cur_keyframe->getId());
        map_point->setFirstLocalMappingGeneration(current_local_mapping_generation);

        map_point->setRefFeature(ref_feature);
        map_point->setCurFeature(cur_feature);
        map_point->addObservation(ref_feature);
        map_point->addObservation(cur_feature);

        ref_feature->setMapPoint(map_point);
        cur_feature->setMapPoint(map_point);

        map_point->updateViewStatistics(scale_factor_, levels_num_);
        map_point->updateRepresentativeDescriptor();

        registerRecentMapPoint(map_point);

        map->addMapPoint(map_point);
        created_map_points_num++;
    }

    return created_map_points_num;
}

std::vector<std::shared_ptr<Frame>> LocalMapper::collectFusionKeyframes(
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<std::shared_ptr<Frame>> result;
    if (cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return result;

    constexpr std::size_t kMaxDirectNeighbors = 10;
    constexpr std::size_t kMaxSecondaryNeighbors = 5;

    std::unordered_set<std::size_t> added_ids;
    added_ids.reserve(kMaxDirectNeighbors * (kMaxSecondaryNeighbors + 1) + 1);

    const std::vector<std::shared_ptr<Frame>> direct_neighbors = 
        cur_keyframe->getBestCovisibilityKeyframes(kMaxDirectNeighbors, 1);

    for (const auto& neighbor : direct_neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe() || neighbor == cur_keyframe)
            continue;

        if (!added_ids.insert(neighbor->getId()).second)
            continue;

        result.push_back(neighbor);
    }

    for (const auto& neighbor : direct_neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe() || neighbor == cur_keyframe)
            continue;

        const std::vector<std::shared_ptr<Frame>> secondary_neighbors = 
            neighbor->getBestCovisibilityKeyframes(kMaxSecondaryNeighbors, 1);

        for (const auto& secondary_neighbor : secondary_neighbors)
        {
            if (secondary_neighbor == nullptr || 
                !secondary_neighbor->isKeyframe() ||
                secondary_neighbor == cur_keyframe)
            {
                continue;
            }

            if (!added_ids.insert(secondary_neighbor->getId()).second)
                continue;

            result.push_back(secondary_neighbor);
        }
    }

    return result;
}

bool LocalMapper::projectionMapPointToFrame(const std::shared_ptr<MapPoint>& map_point,
                                            const std::shared_ptr<Frame>& frame,
                                            cv::Point2f& projected_pixel,
                                            double camera_distance,
                                            int& pred_level) const
{
    projected_pixel = cv::Point2f(0.0f, 0.0f);
    camera_distance = 0.0;
    pred_level = 0;

    if (map_point == nullptr || map_point->isBad() || frame == nullptr)
        return false;

    const std::shared_ptr<Camera> camera = frame->getCamera();
    if (camera == nullptr)
        return false;

    cv::Mat R_cw, t_cw;
    frame->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return false;

    const cv::Point3d& pw = map_point->getPos();
    const cv::Mat point_w = (cv::Mat_<double>(3, 1) << pw.x, pw.y, pw.z);

    const cv::Mat point_c = R_cw * point_w + t_cw;

    const double x = point_c.at<double>(0, 0);
    const double y = point_c.at<double>(1, 0);
    const double z = point_c.at<double>(2, 0);
    if (z <= 1e-6)
        return false;

    const cv::Point3d camera_center = frame->getCameraCenter();
    const cv::Point3d view = pw - camera_center;
    camera_distance = pointNorm(view);
    if (camera_distance <= 1e-6)
        return false;

    if (map_point->hasValidViewStatistics())
    {
        if (camera_distance < map_point->getMinDistance() || 
            camera_distance > map_point->getMaxDistance())
        {
            return false;
        }

        const cv::Point3d& normal = map_point->getNormalVector();
        const double normal_norm = pointNorm(normal);
        if (normal_norm <= 1e-6)
            return false;

        const double view_cos = dotPoint(view, normal) / (camera_distance * normal_norm);
        if (view_cos < 0.5)
            return false;
    }

    const Eigen::Vector3d pc(x, y, z);
    const Eigen::Vector2d pixel = camera->Camera2Pixel(pc);

    projected_pixel = cv::Point2f(pixel(0), pixel(1));
    
    const int border = 10;
    if (projected_pixel.x < border || projected_pixel.x >= frame->getImg().cols - border ||
        projected_pixel.y < border || projected_pixel.y >= frame->getImg().rows - border)
    {
        return false;
    }

    pred_level = 
        map_point->predictScaleLevel(camera_distance, scale_factor_, levels_num_);
    return true;
}

int LocalMapper::findFuseMatchInKeyframe(const std::shared_ptr<MapPoint>& map_point,
                                         const std::shared_ptr<Frame>& keyframe,
                                         const cv::Point2f& projected_pixel,
                                         int pred_level,
                                         const std::unordered_set<int>& used_feature_indices) const
{
    if (map_point == nullptr || keyframe == nullptr)
        return -1;

    cv::Mat map_descriptor;
    if (!getMapPointDescriptor(map_point, map_descriptor))
        return -1;

    const int min_level = std::max(0, pred_level - 1);
    const int max_level = std::min(levels_num_ - 1, pred_level + 1);
    const float search_radius = computeFusionSearchRadius(pred_level, scale_factor_, levels_num_);

    const std::vector<int> candidate_indices = 
        keyframe->getFeatureIndicesInArea(projected_pixel, search_radius, min_level, max_level);

    if (candidate_indices.empty())
        return -1;

    int best_idx = -1;
    int best_distance = std::numeric_limits<int>::max();
    int second_best_distance = std::numeric_limits<int>::max();
    int best_level = -1;
    int second_best_level = -1;

    const cv::Mat& descriptors = keyframe->getDescriptors();
    const std::vector<std::shared_ptr<Feature>>& features = keyframe->getFeatures();

    for (const int feature_idx : candidate_indices)
    {
        if (used_feature_indices.count(feature_idx) > 0)
            continue;

        if (feature_idx < 0 || feature_idx >= descriptors.rows ||
            feature_idx >= static_cast<int>(features.size()))
        {
            continue;
        }

        const std::shared_ptr<Feature>& feature = features[feature_idx];
        if (feature == nullptr)
            continue;

        const double pixel_distance = cv::norm(feature->getKeyPoint().pt - projected_pixel);
        if (pixel_distance > search_radius)
            continue;

        const int descriptor_distance = static_cast<int>(
            cv::norm(descriptors.row(feature_idx), map_descriptor, cv::NORM_HAMMING));

        if (descriptor_distance > matcher_.getMaxHammingDistance())
            continue;

        const int feature_level = feature->getLevel();

        if (descriptor_distance < best_distance)
        {
            second_best_distance = best_distance;
            second_best_level = best_level;

            best_distance = descriptor_distance;
            best_level = feature_level;
            best_idx = feature_idx;
        }
        else if (descriptor_distance < second_best_distance)
        {
            second_best_distance = descriptor_distance;
            second_best_level = feature_level;
        }
    }

    if (best_idx < 0)
        return -1;

    constexpr float kFuseRatio = 0.9f;
    if (second_best_distance != std::numeric_limits<int>::max() &&
        best_level == second_best_level &&
        static_cast<float>(best_distance) >= kFuseRatio * static_cast<float>(second_best_distance))
    {
        return -1;
    }

    return best_idx;
}

std::shared_ptr<MapPoint> LocalMapper::chooseDominantMapPoint(
    const std::shared_ptr<MapPoint>& lhs,
    const std::shared_ptr<MapPoint>& rhs) const
{
    if (lhs == nullptr || lhs->isBad())
        return rhs;
    if (rhs == nullptr || rhs->isBad())
        return lhs;

    const std::size_t lhs_kf_obs = lhs->getKeyframeObservationCount();
    const std::size_t rhs_kf_obs = rhs->getKeyframeObservationCount();
    if (lhs_kf_obs != rhs_kf_obs)
        return lhs_kf_obs > rhs_kf_obs ? lhs : rhs;

    const std::size_t lhs_obs = lhs->getObservationCount();
    const std::size_t rhs_obs = rhs->getObservationCount();
    if (lhs_obs != rhs_obs)
        return lhs_obs > rhs_obs ? lhs : rhs;

    if (lhs->getFoundTimes() != rhs->getFoundTimes())
        return lhs->getFoundTimes() > rhs->getFoundTimes() ? lhs : rhs;

    return lhs->getId() < rhs->getId() ? lhs : rhs;
}

void LocalMapper::mergeMapPoints(const std::shared_ptr<MapPoint>& keep_point,
                                 const std::shared_ptr<MapPoint>& remove_point) const
{
    if (keep_point == nullptr || remove_point == nullptr || keep_point == remove_point)
        return;

    remove_point->replaceWith(keep_point);
}

std::size_t LocalMapper::fuseMapPointsIntoKeyframe(
    const std::vector<std::shared_ptr<MapPoint>>& source_map_points,
    const std::shared_ptr<Frame>& target_keyframe) const
{
    if (target_keyframe == nullptr || !target_keyframe->isKeyframe() || source_map_points.empty())
        return 0;

    const std::vector<std::shared_ptr<Feature>>& target_features = target_keyframe->getFeatures();
    if (target_features.empty() || target_keyframe->getDescriptors().empty())
        return 0;

    std::unordered_set<int> used_feature_indices;
    used_feature_indices.reserve(target_features.size());

    std::unordered_set<std::size_t> touched_ids;
    std::vector<std::shared_ptr<MapPoint>> touched_points;
    touched_ids.reserve(source_map_points.size());

    auto touch_point = [&](const std::shared_ptr<MapPoint>& map_point)
    {
        if (map_point == nullptr || map_point->isBad())
            return;

        if (!touched_ids.insert(map_point->getId()).second)
            return;

        touched_points.push_back(map_point);
    };

    std::size_t fused_num = 0;

    for (const auto& source_map_point : source_map_points)
    {
        if (source_map_point == nullptr || source_map_point->isBad())
            continue;

        cv::Point2f projected_pixel;
        double camera_distance = 0.0;
        int pred_level = 0;
        if (!projectionMapPointToFrame(source_map_point, 
                                        target_keyframe, 
                                        projected_pixel, 
                                        camera_distance, 
                                        pred_level))
        {
            continue;
        }

        const int matched_feature_idx = findFuseMatchInKeyframe(source_map_point, 
                                                                target_keyframe, 
                                                                projected_pixel, 
                                                                pred_level, 
                                                                used_feature_indices);
        if (matched_feature_idx < 0 || matched_feature_idx >= target_features.size())
            continue;
        
        const std::shared_ptr<Feature>& matched_feature = target_features[matched_feature_idx];
        if (matched_feature == nullptr)
            continue;

        used_feature_indices.insert(matched_feature_idx);

        const std::shared_ptr<MapPoint> target_map_point = matched_feature->getMapPoint();

        if (target_map_point == nullptr || target_map_point->isBad())
        {
            matched_feature->setMapPoint(source_map_point);
            source_map_point->addObservation(matched_feature);
            touch_point(source_map_point);
            fused_num++;
            continue;
        }

        if (target_map_point == source_map_point)
            continue;

        const std::shared_ptr<MapPoint> keep_point = 
            chooseDominantMapPoint(source_map_point, target_map_point);
        const std::shared_ptr<MapPoint> remove_point =
            (keep_point == source_map_point) ? target_map_point : source_map_point;

        mergeMapPoints(keep_point, remove_point);
        touch_point(keep_point);
        fused_num++;
    }
    
    refreshMapPointStates(touched_points);

    return fused_num;
}

std::size_t LocalMapper::searchInNeighbors(const std::shared_ptr<Map>& map,
                                           const std::shared_ptr<Frame>& cur_frame) const
{
    if (map == nullptr || cur_frame == nullptr || !cur_frame->isKeyframe())
        return 0;

    const std::vector<std::shared_ptr<Frame>> neighbor_keyframes = 
        collectFusionKeyframes(cur_frame);

    if (neighbor_keyframes.empty())
        return 0;

    const std::vector<std::shared_ptr<MapPoint>> cur_map_points = 
        collectUniqueMapPointsFromKeyframe(cur_frame);

    std::vector<std::vector<std::shared_ptr<MapPoint>>> neighbor_map_points_sets;
    neighbor_map_points_sets.reserve(neighbor_keyframes.size());

    for (const auto& neighbor_keyframe : neighbor_keyframes)
    {
        if (neighbor_keyframe == nullptr || !neighbor_keyframe->isKeyframe())
        {
            neighbor_map_points_sets.emplace_back();
            continue;
        }

        neighbor_map_points_sets.push_back(collectUniqueMapPointsFromKeyframe(neighbor_keyframe));
    }

    std::size_t fused_num = 0;

    for (const auto& neighbor_keyframe : neighbor_keyframes)
        fused_num += fuseMapPointsIntoKeyframe(cur_map_points, neighbor_keyframe);

    for (int i = 0; i < neighbor_keyframes.size(); i++)
    {
        fused_num += fuseMapPointsIntoKeyframe(neighbor_map_points_sets[i], cur_frame);
    }

    updateCovisibilityGraph(map, cur_frame);

    for (const auto& neighbor_keyframe : neighbor_keyframes)
        updateCovisibilityGraph(map, neighbor_keyframe);

    return fused_num;
}

std::size_t LocalMapper::cullMapPoints(const std::shared_ptr<Map>& map,
                                       std::size_t current_local_mapping_generation) const
{
    if (map == nullptr)
        return 0;

    std::size_t culled_num = 0;

    for (auto it = recent_added_map_points_.begin(); it != recent_added_map_points_.end();)
    {
        const std::shared_ptr<MapPoint>& point = *it;
        if (point == nullptr || point->isBad())
        {
            it = recent_added_map_points_.erase(it);
            continue;
        }

        const std::size_t first_local_mapping_generation = point->getFirstLocalMappingGeneration();
        const std::size_t age = current_local_mapping_generation >= first_local_mapping_generation
            ? current_local_mapping_generation - first_local_mapping_generation
            : 0;

        const bool low_found_ratio = 
            point->getVisibleTimes() >= 3 && point->getFoundRatio() < 0.25;
        const bool insufficient_observations = 
            age >= 2 && point->getKeyframeObservationCount() <= 2;

        if (low_found_ratio || insufficient_observations)
        {
            point->setBad(true);
            it = recent_added_map_points_.erase(it);
            culled_num++;
        }
        else if (age >= 3)
        {
            it = recent_added_map_points_.erase(it);
        }
        else
        {
            it++;
        }
    }

    map->removeBadMapPoints();
    return culled_num;
}

LocalMapper::KeyframeRedundancyStats LocalMapper::evaluateKeyframeRedundancy(
    const std::shared_ptr<Frame>& keyframe, 
    const std::shared_ptr<Map>& map) const
{
    KeyframeRedundancyStats stats;

    if (keyframe == nullptr || map == nullptr)
        return stats;

    constexpr std::size_t kMinRedundantObservations = 3;

    for (const auto& feature : keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (map_point->getKeyframeObservationCount() < kMinRedundantObservations)
            continue;

        stats.total_map_features++;

        const int current_level = feature->getLevel();
        std::size_t support_observation_num = 0;

        const std::vector<std::shared_ptr<Feature>> observations = map_point->getObservations();
        for (const auto& observation : observations)
        {
            if (observation == nullptr || observation == feature)
                continue;

            const std::shared_ptr<Frame> observation_keyframe = observation->getFrame();
            if (observation_keyframe == nullptr || 
                observation_keyframe == keyframe ||
                !observation_keyframe->isKeyframe())
            {
                continue;
            }

            if (observation->getLevel() <= current_level + 1)
                support_observation_num++;

            if (support_observation_num >= kMinRedundantObservations)
            {
                stats.redundant_map_features++;
                break;
            }
        }
    }

    if (stats.total_map_features > 0)
    {
        stats.redundant_ratio = 
            static_cast<float>(stats.redundant_map_features) / 
            static_cast<float>(stats.total_map_features);
    }

    return stats;
}

bool LocalMapper::isKeyframeRedundant(const std::shared_ptr<Frame>& keyframe,
                                      const std::shared_ptr<Map>& map,
                                      KeyframeRedundancyStats* stats) const
{
    if (keyframe == nullptr || map == nullptr)
        return false;

    const std::vector<std::shared_ptr<Frame>>& keyframes = map->getKeyframes();
    if (keyframes.size() <= 2)
        return false;

    if (keyframe == keyframes.front() || keyframe == keyframes.back())
        return false;

    const KeyframeRedundancyStats evaluated_stats = evaluateKeyframeRedundancy(keyframe, map);
    if (stats != nullptr)
    {
        *stats = evaluated_stats;
    }

    if (evaluated_stats.total_map_features < 20)
        return false;

    return evaluated_stats.redundant_ratio > 0.9;
}

std::vector<std::shared_ptr<Frame>> LocalMapper::collectKeyframeCullingCandidates(
    const std::shared_ptr<Frame>& cur_frame) const
{
    std::vector<std::shared_ptr<Frame>> candidates;
    if (cur_frame == nullptr || !cur_frame->isKeyframe())
        return candidates;

    constexpr std::size_t kMaxCullingNeighbors = 10;
    constexpr int kMinCullingWeight = 15;

    const std::vector<std::shared_ptr<Frame>> neighbors = 
        cur_frame->getBestCovisibilityKeyframes(kMaxCullingNeighbors, kMinCullingWeight);

    std::unordered_set<std::size_t> seen_ids;
    seen_ids.reserve(neighbors.size());

    for (const auto& neighbor : neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe() || neighbor == cur_frame)
            continue;

        if (!seen_ids.insert(neighbor->getId()).second)
            continue;

        candidates.push_back(neighbor);
    }

    return candidates;
}

std::size_t LocalMapper::cullKeyframes(const std::shared_ptr<Map>& map,
                                       const std::shared_ptr<Frame>& cur_keyframe) const
{
    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return 0;

    const std::vector<std::shared_ptr<Frame>> candidate = 
        collectKeyframeCullingCandidates(cur_keyframe);

    if (candidate.empty())
        return 0;

    std::vector<std::shared_ptr<Frame>> redundant_keyframes;
    redundant_keyframes.reserve(candidate.size());

    const std::vector<std::shared_ptr<Frame>>& map_keyframes = map->getKeyframes();

    for (const auto& keyframe : candidate)
    {
        if (isKeyframeRedundant(keyframe, map))
            redundant_keyframes.push_back(keyframe);
    }

    for (const auto& keyframe : redundant_keyframes)
        map->removeKeyframe(keyframe);

    return redundant_keyframes.size();
}

} // namespace mini_orb_slam
