#include <unordered_set>
#include <algorithm>
#include <limits.h>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <ros/ros.h>

#include "local_mapper.h"

#include "bow_vocabulary.h"
#include "keyframe_database.h"
#include "loop_closer.h"

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

double minDistanceInvariance(double min_distance)
{
    return 0.8 * min_distance;
}

double maxDistanceInvariance(double max_distance)
{
    return 1.2 * max_distance;
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

    if (map_point->getRepresentativeDescriptorView(descriptor))
        return true;

    return getFeatureDescriptor(map_point->selectRefFeatureCandidate(), descriptor);
}

int descriptorDistanceFast(const cv::Mat& lhs, const cv::Mat& rhs)
{
    if (lhs.rows != 1 || rhs.rows != 1 || lhs.cols != rhs.cols ||
        lhs.type() != CV_8U || rhs.type() != CV_8U || lhs.cols != 32 ||
        !lhs.isContinuous() || !rhs.isContinuous())
    {
        return static_cast<int>(cv::norm(lhs, rhs, cv::NORM_HAMMING));
    }

    const auto* lhs_words = reinterpret_cast<const std::uint32_t*>(lhs.ptr<unsigned char>());
    const auto* rhs_words = reinterpret_cast<const std::uint32_t*>(rhs.ptr<unsigned char>());
    int distance = 0;
    for (int i = 0; i < 8; ++i)
        distance += __builtin_popcount(lhs_words[i] ^ rhs_words[i]);
    return distance;
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

struct TriangulationMatchDiagnostics
{
    std::size_t raw_matches{0};
    std::size_t occupied_feature_matches{0};
    std::size_t epipolar_rejected_matches{0};
    std::size_t rotation_rejected_matches{0};
    std::size_t candidate_matches{0};
};

std::vector<std::pair<int, int>> collectTriangulationMatches(
    const std::vector<cv::DMatch>& raw_matches,
    const std::shared_ptr<Frame>& ref_keyframe,
    const std::shared_ptr<Frame>& cur_keyframe,
    const cv::Matx33d& F_21,
    double scale_factor,
    struct TriangulationMatchDiagnostics* diagnostics = nullptr)
{
    std::vector<std::pair<int, int>> candidate_matches;

    if (diagnostics != nullptr)
        *diagnostics = {};

    if (ref_keyframe == nullptr || cur_keyframe == nullptr ||
        !ref_keyframe->hasFeatures() || !cur_keyframe->hasFeatures())
    {
        return candidate_matches;
    }

    if (diagnostics != nullptr)
        diagnostics->raw_matches = raw_matches.size();
    if (raw_matches.empty())
        return candidate_matches;

    const std::vector<std::shared_ptr<Feature>>& ref_features = ref_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_keyframe->getFeatures();
    const std::vector<cv::KeyPoint>& ref_keypoints = ref_keyframe->getKeypoints();
    const std::vector<cv::KeyPoint>& cur_keypoints = cur_keyframe->getKeypoints();
    const cv::Mat& ref_descriptors = ref_keyframe->getDescriptors();
    const cv::Mat& cur_descriptors = cur_keyframe->getDescriptors();

    std::vector<int> best_ref_for_cur(cur_features.size(), -1);
    std::vector<int> best_dist_for_cur(cur_features.size(), std::numeric_limits<int>::max());
    std::vector<float> best_angle_diff_for_cur(cur_features.size(), -1.0f);

    for (const cv::DMatch& raw_match : raw_matches)
    {
        const int ref_idx = raw_match.queryIdx;
        const int cur_idx = raw_match.trainIdx;

        if (ref_idx < 0 || ref_idx >= ref_features.size() ||
            cur_idx < 0 || cur_idx >= cur_features.size() ||
            ref_idx >= ref_descriptors.rows || cur_idx >= cur_descriptors.rows ||
            ref_idx >= ref_keypoints.size() || cur_idx >= cur_keypoints.size())
        {
            continue;
        }

        const std::shared_ptr<Feature>& ref_feature = ref_features[ref_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (ref_feature == nullptr || cur_feature == nullptr)
            continue;


        if (ref_feature->hasMapPoint() || cur_feature->hasMapPoint())
        {
            if (diagnostics != nullptr)
                diagnostics->occupied_feature_matches++;
            continue;
        }

        if (!passesEpipolarConstraint(F_21, 
                                      ref_feature->getKeyPoint().pt, 
                                      cur_feature->getKeyPoint().pt, 
                                      cur_feature->getLevel(), 
                                      scale_factor))
        {
            if (diagnostics != nullptr)
                diagnostics->epipolar_rejected_matches++;
            continue;
        }

        const int dist = cvRound(raw_match.distance);

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
        {
            if (best_ref_for_cur[cur_idx] >= 0 && diagnostics != nullptr)
                diagnostics->rotation_rejected_matches++;
            continue;
        }

        candidate_matches.emplace_back(best_ref_for_cur[cur_idx], cur_idx);
    }

    if (diagnostics != nullptr)
        diagnostics->candidate_matches = candidate_matches.size();

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
        if (feature == nullptr)
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
    constexpr float kBaseSearchRadius = 3.0f;

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

void LocalMapper::setKeyframeDatabase(
    const std::shared_ptr<BoWVocabulary>& vocabulary,
    const std::shared_ptr<KeyframeDatabase>& database)
{
    bow_vocabulary_ = vocabulary;
    keyframe_database_ = database;
}

void LocalMapper::setLoopCloser(LoopCloser* loop_closer)
{
    loop_closer_ = loop_closer;
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
    latest_scheduled_map_.reset();
    latest_scheduled_keyframe_.reset();
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

void LocalMapper::requestStop()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_requested_ = true;
        accept_keyframes_ = false;
    }
    queue_cv_.notify_all();
}

void LocalMapper::release()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_requested_ = false;
        stopped_ = false;
        accept_keyframes_ = !finish_requested_;
    }
    queue_cv_.notify_all();
}

bool LocalMapper::insertKeyframe(const LocalMappingInput& input)
{
    if (input.map == nullptr || input.ref_keyframe == nullptr || input.cur_keyframe == nullptr)
        return false;

    if (!input.cur_keyframe->isKeyframe())
        return false;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!accept_keyframes_ || finish_requested_ || stop_requested_)
            return false;

        if (pending_keyframes_.size() >= kMaxPendingKeyframes)
            return false;

        pending_keyframes_.push_back(input);
        latest_scheduled_map_ = input.map;
        latest_scheduled_keyframe_ = input.cur_keyframe;
    }

    queue_cv_.notify_one();
    return true;
}

std::shared_ptr<Frame> LocalMapper::getLatestScheduledKeyframe(
    const std::shared_ptr<Map>& map) const
{
    if (map == nullptr)
        return nullptr;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (latest_scheduled_map_.lock() != map)
        return nullptr;

    return latest_scheduled_keyframe_.lock();
}

bool LocalMapper::hasPendingKeyframe() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !pending_keyframes_.empty();
}

bool LocalMapper::waitUntilIdle()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]()
    {
        return finish_requested_ || stop_requested_ ||
               (pending_keyframes_.empty() && !processing_new_keyframe_);
    });

    return !finish_requested_ && !stop_requested_;
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

bool LocalMapper::waitPopFinishedResult(LocalMappingOutput& output)
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]()
    {
        return finish_requested_ || !finished_results_.empty();
    });

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
           pending_keyframes_.size() < kMaxPendingKeyframes;
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

bool LocalMapper::finishRequested() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return finish_requested_;
}

void LocalMapper::run()
{
    std::size_t processed_keyframes = 0;
    std::size_t fusion_calls = 0;
    std::size_t local_ba_calls = 0;
    std::size_t keyframe_cull_calls = 0;
    std::size_t convergence_yields = 0;
    double total_mapping_ms = 0.0;
    double total_ba_ms = 0.0;

    while (true)
    {
        LocalMappingInput input;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]()
            {
                return finish_requested_ || stop_requested_ || !pending_keyframes_.empty();
            });

            if (finish_requested_ && pending_keyframes_.empty())
                break;

            if (stop_requested_)
            {
                stopped_ = true;
                accept_keyframes_ = false;
                queue_cv_.wait(lock, [this]()
                {
                    return finish_requested_ || !stop_requested_;
                });
                if (finish_requested_ && pending_keyframes_.empty())
                    break;
                continue;
            }

            stopped_ = false;

            input = pending_keyframes_.front();
            pending_keyframes_.pop_front();
            processing_new_keyframe_ = true;
        }

        LocalMappingOutput output;
        output.input = input;
        output.result = processNewKeyframe(input.map,
                                           input.ref_keyframe,
                                           input.cur_keyframe,
                                           input.tracking_seed);

        processed_keyframes++;
        fusion_calls += output.result.fusion_called ? 1 : 0;
        local_ba_calls += output.result.local_ba_called ? 1 : 0;
        keyframe_cull_calls += output.result.keyframe_cull_called ? 1 : 0;
        convergence_yields +=
            output.result.convergence_skipped_for_pending_keyframe ? 1 : 0;
        total_mapping_ms += output.result.total_duration_ms;
        total_ba_ms += output.result.local_ba_duration_ms;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            finished_results_.push_back(std::move(output));
            processing_new_keyframe_ = false;
        }
        queue_cv_.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stopped_ = true;
        finished_ = true;
        accept_keyframes_ = false;
    }

    ROS_INFO_STREAM("P2-SLAM-SCHED summary processed=" << processed_keyframes
                    << " fusion=" << fusion_calls
                    << " local_ba=" << local_ba_calls
                    << " keyframe_cull=" << keyframe_cull_calls
                    << " yielded_convergence=" << convergence_yields
                    << " local_mapping_total_ms=" << total_mapping_ms
                    << " local_mapping_mean_ms="
                    << (processed_keyframes == 0 ? 0.0 : total_mapping_ms / processed_keyframes)
                    << " local_ba_total_ms=" << total_ba_ms
                    << " local_ba_mean_ms="
                    << (local_ba_calls == 0 ? 0.0 : total_ba_ms / local_ba_calls));
}

LocalMappingResult LocalMapper::processNewKeyframe(const std::shared_ptr<Map>& map, 
                                                   const std::shared_ptr<Frame>& ref_keyframe, 
                                                   const std::shared_ptr<Frame>& cur_keyframe,
                                                   const PnPResult& tracking_seed)
{
    LocalMappingResult result;
    const auto processing_start = std::chrono::steady_clock::now();

    if (map == nullptr || ref_keyframe == nullptr || cur_keyframe == nullptr)
        return result;

    if (!cur_keyframe->isKeyframe())
        return result;

    double map_lock_wait_ms = 0.0;
    double keyframe_commit_ms = 0.0;
    double map_point_cull_ms = 0.0;
    double triangulation_ms = 0.0;
    double fusion_and_graph_ms = 0.0;
    double fusion_pre_graph_ms = 0.0;
    double fusion_reference_graph_ms = 0.0;
    double keyframe_cull_ms = 0.0;
    std::size_t fused_map_point_num = 0;
    std::size_t current_local_mapping_generation = 0;
    std::vector<std::shared_ptr<Frame>> triangulation_keyframes;
    {
        const auto map_lock_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> map_lock(map->getMutex());
        map_lock_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - map_lock_start).count();
        if (recent_map_.lock() != map)
        {
            recent_map_ = map;
            recent_added_map_points_.clear();
            local_mapping_generation_ = 0;
        }

        const auto keyframe_commit_start = std::chrono::steady_clock::now();
        map->addKeyframe(cur_keyframe);

        current_local_mapping_generation = local_mapping_generation_++;

        processCurrentKeyframeMapPoints(cur_keyframe);
        updateCovisibilityGraph(map, cur_keyframe);
        keyframe_commit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - keyframe_commit_start).count();

        const auto map_point_cull_start = std::chrono::steady_clock::now();
        result.culled_map_point_num = cullMapPoints(map, current_local_mapping_generation);
        map_point_cull_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - map_point_cull_start).count();

        triangulation_keyframes = collectTriangulationKeyframes(ref_keyframe, cur_keyframe);
    }

    const auto descriptor_match_start = std::chrono::steady_clock::now();
    const std::vector<TriangulationMatchCache> triangulation_match_cache =
        collectTriangulationMatchCache(triangulation_keyframes, cur_keyframe);
    result.triangulation_match_duration_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - descriptor_match_start).count();

    {
        const auto map_lock_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> map_lock(map->getMutex());
        map_lock_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - map_lock_start).count();
        const auto triangulation_start = std::chrono::steady_clock::now();
        result.new_map_point_num =
            growMapByKeyFrames(map,
                               triangulation_match_cache,
                               cur_keyframe,
                               current_local_mapping_generation,
                               &result);
        triangulation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - triangulation_start).count();
    }

    LocalBAResult local_ba_result;
    if (!hasPendingKeyframe())
    {
        {
            const auto map_lock_start = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> map_lock(map->getMutex());
            map_lock_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - map_lock_start).count();
            const auto fusion_and_graph_start = std::chrono::steady_clock::now();
            const auto pre_graph_start = std::chrono::steady_clock::now();
            updateCovisibilityGraph(map, cur_keyframe);
            fusion_pre_graph_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pre_graph_start).count();
            fused_map_point_num = searchInNeighbors(map, cur_keyframe, &result);

            if (ref_keyframe != cur_keyframe)
            {
                const auto reference_graph_start = std::chrono::steady_clock::now();
                updateCovisibilityGraph(map, ref_keyframe);
                fusion_reference_graph_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - reference_graph_start).count();
            }
            fusion_and_graph_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fusion_and_graph_start).count();
            result.fusion_called = true;
        }

        bool run_convergence = false;
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            if (!pending_keyframes_.empty())
            {
                result.convergence_skipped_for_pending_keyframe = true;
            }
            else if (!stop_requested_ && !finish_requested_)
            {
                run_convergence = true;
            }
        }

        if (run_convergence)
        {
            {
                const auto map_lock_start = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> map_lock(map->getMutex());
                map_lock_wait_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - map_lock_start).count();
                constexpr std::size_t kMinKeyframesForLocalBA = 3;
                result.local_ba_called = (pose_optimizer_ != nullptr) &&
                                         (map->getKeyframeNum() > kMinKeyframesForLocalBA);
            }

            if (result.local_ba_called)
            {
                const auto local_ba_start = std::chrono::steady_clock::now();
                local_ba_result = pose_optimizer_->optimizeLocalMap(map,
                                                                    cur_keyframe,
                                                                    tracking_seed);
                result.local_ba_duration_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - local_ba_start).count();
            }

            const auto map_lock_start = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> map_lock(map->getMutex());
            map_lock_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - map_lock_start).count();
            const auto keyframe_cull_start = std::chrono::steady_clock::now();
            result.culled_keyframe_num = cullKeyframes(map, cur_keyframe, &result);
            keyframe_cull_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - keyframe_cull_start).count();
            result.keyframe_cull_called = true;
            result.active_keyframe_num = map->getKeyframeNum();
            result.active_map_point_num = map->getMapPointNum();
        }
    }
    else
    {
        result.convergence_skipped_for_pending_keyframe = true;
    }

    if (!result.keyframe_cull_called)
    {
        const auto map_lock_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> map_lock(map->getMutex());
        map_lock_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - map_lock_start).count();
        result.active_keyframe_num = map->getKeyframeNum();
        result.active_map_point_num = map->getMapPointNum();
    }

    result.local_ba_solver_success = local_ba_result.solver_success;
    result.local_ba_success = local_ba_result.accepted;
    result.local_ba_rejected = local_ba_result.solver_success && !local_ba_result.accepted;
    result.local_ba_edge_num = local_ba_result.edge_num;
    result.local_ba_rejected_edge_num = local_ba_result.rejected_edge_num;
    result.local_ba_seed_reproj_error = local_ba_result.seed_reproj_error;
    result.local_ba_candidate_seed_reproj_error = local_ba_result.candidate_seed_reproj_error;
    result.local_ba_rejection_reason = local_ba_result.rejection_reason;

    handOffCommittedKeyframe(map, cur_keyframe, result);

    result.success = (result.new_map_point_num > 0) ||
                     (fused_map_point_num > 0) ||
                     (result.culled_map_point_num > 0) ||
                     (result.culled_keyframe_num > 0) ||
                     result.local_ba_success;

    const auto processing_end = std::chrono::steady_clock::now();
    const double processing_ms = std::chrono::duration<double, std::milli>(
        processing_end - processing_start).count();
    result.total_duration_ms = processing_ms;
    result.keyframe_commit_duration_ms = keyframe_commit_ms;
    result.map_point_cull_duration_ms = map_point_cull_ms;
    result.triangulation_duration_ms = triangulation_ms;
    result.fusion_duration_ms = fusion_and_graph_ms;
    result.fusion_pre_graph_duration_ms = fusion_pre_graph_ms;
    result.fusion_reference_graph_duration_ms = fusion_reference_graph_ms;
    result.keyframe_cull_duration_ms = keyframe_cull_ms;
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-SLAM-PERF local_mapping keyframe=" << cur_keyframe->getId()
        << " map_lock_wait_ms=" << map_lock_wait_ms
        << " total_ms=" << processing_ms
        << " ba_ms=" << result.local_ba_duration_ms
        << " ba_called=" << (result.local_ba_called ? 1 : 0)
        << " ba_accepted=" << (result.local_ba_success ? 1 : 0));
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-SLAM-PERF local_mapping_sections keyframe_commit_ms="
        << keyframe_commit_ms
        << " map_point_cull_ms=" << map_point_cull_ms
        << " triangulation_ms=" << triangulation_ms
        << " fusion_and_graph_ms=" << fusion_and_graph_ms
        << " ba_ms=" << result.local_ba_duration_ms
        << " keyframe_cull_ms=" << keyframe_cull_ms);
    ROS_INFO_STREAM(
        "P2-SLAM-PERF fusion_phases keyframe=" << cur_keyframe->getId()
        << " pre_graph_ms=" << fusion_pre_graph_ms
        << " context_ms=" << result.fusion_context_duration_ms
        << " forward_ms=" << result.fusion_forward_duration_ms
        << " reverse_ms=" << result.fusion_reverse_duration_ms
        << " post_graph_ms=" << result.fusion_post_graph_duration_ms
        << " reference_graph_ms=" << fusion_reference_graph_ms
        << " total_ms=" << fusion_and_graph_ms);
    ROS_INFO_STREAM(
        "P2-SLAM-PERF cull_phases keyframe=" << cur_keyframe->getId()
        << " candidate_collection_ms="
        << result.keyframe_cull_candidate_collection_duration_ms
        << " evaluation_ms=" << result.keyframe_cull_evaluation_duration_ms
        << " removal_ms=" << result.keyframe_cull_removal_duration_ms
        << " total_ms=" << keyframe_cull_ms);
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-SLAM-PERF local_mapping_work keyframe=" << cur_keyframe->getId()
        << " tri_partners=" << result.triangulation_partner_num
        << " tri_pairs=" << result.triangulation_pair_num
        << " tri_candidates=" << result.triangulation_candidate_match_num
        << " tri_svd=" << result.triangulation_svd_num
        << " tri_valid_points=" << result.triangulation_valid_point_num
        << " fusion_targets=" << result.fusion_target_keyframe_num
        << " fusion_sources=" << result.fusion_source_map_point_num
        << " fusion_projection_attempts=" << result.fusion_projection_attempt_num
        << " fusion_projection_valid=" << result.fusion_projection_valid_num
        << " fusion_descriptor_queries=" << result.fusion_descriptor_query_num
        << " fusion_feature_candidates=" << result.fusion_feature_candidate_num
        << " fusion_descriptor_comparisons=" << result.fusion_descriptor_comparison_num
        << " cull_candidates=" << result.keyframe_cull_candidate_num
        << " cull_features=" << result.keyframe_cull_feature_num
        << " cull_observations=" << result.keyframe_cull_observation_num);
    ROS_INFO_STREAM(
        "P2-SLAM-PERF triangulation_phases keyframe=" << cur_keyframe->getId()
        << " match_ms=" << result.triangulation_match_duration_ms
        << " geometry_ms=" << result.triangulation_geometry_duration_ms
        << " commit_ms=" << result.triangulation_commit_duration_ms
        << " total_ms=" << triangulation_ms
        << " pairs=" << result.triangulation_pair_num
        << " candidates=" << result.triangulation_candidate_match_num
        << " valid_points=" << result.triangulation_valid_point_num
        << " created_points=" << result.new_map_point_num);
    ROS_INFO_STREAM(
        "P2-SLAM-PERF triangulation_match_policy keyframe=" << cur_keyframe->getId()
        << " global_bf=1 distance_reuse=1");
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-SLAM-PERF local_ba_profile keyframe=" << cur_keyframe->getId()
        << " local_kf=" << local_ba_result.local_keyframe_num
        << " fixed_kf=" << local_ba_result.fixed_keyframe_num
        << " local_mp=" << local_ba_result.local_map_point_num
        << " observations=" << local_ba_result.observation_num
        << " mp_ba_edges_2=" << local_ba_result.map_points_with_2_ba_edges
        << " mp_ba_edges_3=" << local_ba_result.map_points_with_3_ba_edges
        << " mp_ba_edges_4plus="
        << local_ba_result.map_points_with_4_or_more_ba_edges
        << " edges=" << local_ba_result.edge_num
        << " context_ms=" << local_ba_result.context_build_ms
        << " graph_ms=" << local_ba_result.graph_build_ms
        << " solve_ms=" << local_ba_result.solve_ms
        << " validation_ms=" << local_ba_result.validation_ms
        << " view_stats_ms=" << local_ba_result.view_statistics_ms
        << " descriptor_refresh_ms=" << local_ba_result.descriptor_refresh_ms
        << " descriptor_refresh_num=" << local_ba_result.descriptor_refresh_num
        << " commit_ms=" << local_ba_result.commit_ms
        << " accepted=" << (local_ba_result.accepted ? 1 : 0)
        << " reason=" << local_ba_result.rejection_reason);

    return result;
}

void LocalMapper::handOffCommittedKeyframe(
    const std::shared_ptr<Map>& map,
    const std::shared_ptr<Frame>& cur_keyframe,
    LocalMappingResult& result) const
{
    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return;

    if (keyframe_database_ != nullptr)
    {
        if (!cur_keyframe->hasBoW() && bow_vocabulary_ != nullptr)
            cur_keyframe->computeBoW(bow_vocabulary_);

        result.keyframe_database_registered =
            keyframe_database_->addKeyframe(cur_keyframe);
    }

    if (loop_closer_ != nullptr)
    {
        LoopClosingInput input;
        input.map = map;
        input.cur_keyframe = cur_keyframe;
        result.loop_keyframe_queued = loop_closer_->insertKeyframe(input);
    }
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

        map_point->addObservation(feature);
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

std::vector<LocalMapper::TriangulationMatchCache>
LocalMapper::collectTriangulationMatchCache(
    const std::vector<std::shared_ptr<Frame>>& triangulation_keyframes,
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<TriangulationMatchCache> match_cache;
    if (cur_keyframe == nullptr || !cur_keyframe->hasFeatures())
        return match_cache;

    match_cache.reserve(triangulation_keyframes.size());
    for (const std::shared_ptr<Frame>& ref_keyframe : triangulation_keyframes)
    {
        if (ref_keyframe == nullptr || !ref_keyframe->hasFeatures())
            continue;

        TriangulationMatchCache cache;
        cache.ref_keyframe = ref_keyframe;
        cache.raw_matches = matcher_.matchDescriptorsWithDistance(
            ref_keyframe->getDescriptors(), cur_keyframe->getDescriptors());
        match_cache.push_back(std::move(cache));
    }

    return match_cache;
}

std::size_t LocalMapper::growMapByKeyFrames(const std::shared_ptr<Map>& map, 
                                            const std::vector<TriangulationMatchCache>& match_cache,
                                            const std::shared_ptr<Frame>& cur_keyframe,
                                            std::size_t current_local_mapping_generation,
                                            LocalMappingResult* result) const
{   
    if (map == nullptr || cur_keyframe == nullptr || initializer_ == nullptr)
        return 0;

    if (result != nullptr)
        result->triangulation_partner_num += match_cache.size();

    std::size_t created_map_points_num = 0;

    for (const TriangulationMatchCache& cache : match_cache)
    {
        created_map_points_num += growMapByKeyFramePair(map, 
                                                        cache.ref_keyframe,
                                                        cur_keyframe, 
                                                        cache.raw_matches,
                                                        current_local_mapping_generation,
                                                        result);
    }

    return created_map_points_num;
}

std::size_t LocalMapper::growMapByKeyFramePair(const std::shared_ptr<Map>& map,
                                               const std::shared_ptr<Frame>& ref_keyframe,
                                               const std::shared_ptr<Frame>& cur_keyframe,
                                               const std::vector<cv::DMatch>& raw_matches,
                                               std::size_t current_local_mapping_generation,
                                               LocalMappingResult* result) const
{
    if (map == nullptr || ref_keyframe == nullptr || cur_keyframe == nullptr || initializer_ == nullptr)
        return 0;

    if (result != nullptr)
        result->triangulation_pair_num++;

    const double baseline = 
        cv::norm(ref_keyframe->getCameraCenter() - cur_keyframe->getCameraCenter());

    const double median_depth = computeMedianSceneDepth(ref_keyframe);
    if (median_depth <= 1e-6)
    {
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                        << " pair_cur=" << cur_keyframe->getId()
                        << " stage=triangulation_gate baseline=" << baseline
                        << " median_depth=" << median_depth
                        << " ratio=NA candidates=0 created=0 reason=median_depth");
        return 0;
    }

    constexpr double kMinBaselineDepthRatio = 0.01;
    const double baseline_depth_ratio = baseline / median_depth;
    if (baseline_depth_ratio < kMinBaselineDepthRatio)
    {
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                        << " pair_cur=" << cur_keyframe->getId()
                        << " stage=triangulation_gate baseline=" << baseline
                        << " median_depth=" << median_depth
                        << " ratio=" << baseline_depth_ratio
                        << " candidates=0 created=0 reason=baseline_depth_ratio");
        return 0;
    }

    cv::Matx33d F_21;
    if (!computeFundamentalMatrix(ref_keyframe, cur_keyframe, F_21))
    {
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                        << " pair_cur=" << cur_keyframe->getId()
                        << " stage=triangulation_gate baseline=" << baseline
                        << " median_depth=" << median_depth
                        << " ratio=" << baseline_depth_ratio
                        << " candidates=0 created=0 reason=fundamental");
        return 0;
    }

    const std::vector<std::shared_ptr<Feature>>& ref_features = ref_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_keyframe->getFeatures();

    TriangulationMatchDiagnostics match_diagnostics;
    const auto match_start = std::chrono::steady_clock::now();
    const std::vector<std::pair<int, int>> candidate_matches = 
        collectTriangulationMatches(raw_matches, ref_keyframe, cur_keyframe, F_21,
                                    scale_factor_, &match_diagnostics);
    if (result != nullptr)
    {
        result->triangulation_match_duration_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - match_start).count();
    }

    if (result != nullptr)
    {
        result->triangulation_candidate_match_num += candidate_matches.size();
    }

    if(candidate_matches.size() < 10)
    {
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                        << " pair_cur=" << cur_keyframe->getId()
                        << " stage=triangulation_matches baseline=" << baseline
                        << " median_depth=" << median_depth
                        << " ratio=" << baseline_depth_ratio
                        << " raw=" << match_diagnostics.raw_matches
                        << " occupied=" << match_diagnostics.occupied_feature_matches
                        << " epipolar_rejected=" << match_diagnostics.epipolar_rejected_matches
                        << " rotation_rejected=" << match_diagnostics.rotation_rejected_matches
                        << " candidates=" << match_diagnostics.candidate_matches
                        << " created=0 reason=insufficient_candidates");
        return 0;
    }

    const auto geometry_start = std::chrono::steady_clock::now();
    const TriangulationResult triangulation_result =
        initializer_->triangulateFromMatchedFrames(ref_keyframe, 
                                                   cur_keyframe, 
                                                   candidate_matches);
    if (result != nullptr)
    {
        result->triangulation_geometry_duration_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - geometry_start).count();
    }

    if (result != nullptr)
    {
        result->triangulation_svd_num++;
        result->triangulation_valid_point_num += triangulation_result.points_3d.size();
    }

    if (triangulation_result.points_3d.empty())
    {
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                        << " pair_cur=" << cur_keyframe->getId()
                        << " stage=triangulation_result baseline=" << baseline
                        << " median_depth=" << median_depth
                        << " ratio=" << baseline_depth_ratio
                        << " raw=" << match_diagnostics.raw_matches
                        << " occupied=" << match_diagnostics.occupied_feature_matches
                        << " epipolar_rejected=" << match_diagnostics.epipolar_rejected_matches
                        << " rotation_rejected=" << match_diagnostics.rotation_rejected_matches
                        << " candidates=" << match_diagnostics.candidate_matches
                        << " triangulated=0 created=0 reason=no_valid_points");
        return 0;
    }

    const auto commit_start = std::chrono::steady_clock::now();
    std::size_t created_map_points_num = 0;
    std::size_t invalid_result_pairs = 0;
    std::size_t occupied_result_pairs = 0;
    std::size_t scale_rejected_pairs = 0;

    for (std::size_t i = 0; i < triangulation_result.points_3d.size(); i++)
    {
        const int ref_idx = triangulation_result.ref_feature_indices[i];
        const int cur_idx = triangulation_result.cur_feature_indices[i];

        if (ref_idx < 0 || ref_idx >= static_cast<int>(ref_features.size()) ||
            cur_idx < 0 || cur_idx >= static_cast<int>(cur_features.size()))
        {
            invalid_result_pairs++;
            continue;
        }

        const std::shared_ptr<Feature>& ref_feature = ref_features[ref_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (ref_feature == nullptr || cur_feature == nullptr)
        {
            invalid_result_pairs++;
            continue;
        }

        if (ref_feature->hasMapPoint() || cur_feature->hasMapPoint())
        {
            occupied_result_pairs++;
            continue;
        }

        if (!passesScaleConsistency(ref_keyframe, 
                                    cur_keyframe, 
                                    triangulation_result.points_3d[i], 
                                    ref_feature->getLevel(), 
                                    cur_feature->getLevel(), 
                                    scale_factor_))
        {
            scale_rejected_pairs++;
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
    if (result != nullptr)
    {
        result->triangulation_commit_duration_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - commit_start).count();
    }

    ROS_DEBUG_STREAM("P2-SLAM-DEBUG pair_ref=" << ref_keyframe->getId()
                    << " pair_cur=" << cur_keyframe->getId()
                    << " stage=triangulation_result baseline=" << baseline
                    << " median_depth=" << median_depth
                    << " ratio=" << baseline_depth_ratio
                    << " raw=" << match_diagnostics.raw_matches
                    << " occupied=" << match_diagnostics.occupied_feature_matches
                    << " epipolar_rejected=" << match_diagnostics.epipolar_rejected_matches
                    << " rotation_rejected=" << match_diagnostics.rotation_rejected_matches
                    << " candidates=" << match_diagnostics.candidate_matches
                    << " triangulated=" << triangulation_result.points_3d.size()
                    << " invalid=" << invalid_result_pairs
                    << " occupied_result=" << occupied_result_pairs
                    << " scale_rejected=" << scale_rejected_pairs
                    << " created=" << created_map_points_num
                    << " reason=" << (created_map_points_num > 0 ? "created" : "filtered"));

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
                                            const FusionProjectionContext& context,
                                            cv::Point2f& projected_pixel,
                                            double& camera_distance,
                                            int& pred_level) const
{
    projected_pixel = cv::Point2f(0.0f, 0.0f);
    camera_distance = 0.0;
    pred_level = 0;

    if (map_point == nullptr || map_point->isBad() || context.camera == nullptr ||
        context.R_cw.empty() || context.t_cw.empty())
        return false;

    const cv::Point3d pw = map_point->getPos();
    const double* R = context.R_cw.ptr<double>();
    const double* t = context.t_cw.ptr<double>();
    const double x = R[0] * pw.x + R[1] * pw.y + R[2] * pw.z + t[0];
    const double y = R[3] * pw.x + R[4] * pw.y + R[5] * pw.z + t[1];
    const double z = R[6] * pw.x + R[7] * pw.y + R[8] * pw.z + t[2];
    if (z <= 1e-6)
        return false;

    const cv::Point3d view = pw - context.camera_center;
    camera_distance = pointNorm(view);
    if (camera_distance <= 1e-6)
        return false;

    cv::Point3d normal;
    double min_distance = 0.0;
    double max_distance = 0.0;
    if (map_point->getViewStatistics(normal, min_distance, max_distance))
    {
        if (camera_distance < minDistanceInvariance(min_distance) ||
            camera_distance > maxDistanceInvariance(max_distance))
        {
            return false;
        }

        const double normal_norm = pointNorm(normal);
        if (normal_norm <= 1e-6)
            return false;

        const double view_cos = dotPoint(view, normal) / (camera_distance * normal_norm);
        if (view_cos < 0.5)
            return false;
    }

    double u = 0.0;
    double v = 0.0;
    if (!context.camera->projectCameraPoint(x, y, z, u, v))
        return false;

    projected_pixel = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
    
    const int border = 10;
    if (projected_pixel.x < border || projected_pixel.x >= context.image_size.width - border ||
        projected_pixel.y < border || projected_pixel.y >= context.image_size.height - border)
    {
        return false;
    }

    pred_level = 
        map_point->predictScaleLevel(camera_distance, scale_factor_, levels_num_);
    return true;
}

int LocalMapper::findFuseMatchInKeyframe(const std::shared_ptr<MapPoint>& map_point,
                                         const std::shared_ptr<Frame>& keyframe,
                                         const cv::Mat& map_descriptor,
                                         const cv::Point2f& projected_pixel,
                                         int pred_level,
                                         const std::unordered_set<int>& used_feature_indices,
                                         std::vector<int>& candidate_indices,
                                         LocalMappingResult* result) const
{
    if (map_point == nullptr || keyframe == nullptr)
        return -1;

    if (map_descriptor.empty())
        return -1;

    const int min_level = std::max(0, pred_level - 1);
    const int max_level = std::min(levels_num_ - 1, pred_level);
    const float search_radius = computeFusionSearchRadius(pred_level, scale_factor_, levels_num_);

    keyframe->appendFeatureIndicesInArea(projected_pixel, search_radius,
                                         min_level, max_level, candidate_indices);

    if (result != nullptr)
        result->fusion_feature_candidate_num += candidate_indices.size();

    if (candidate_indices.empty())
        return -1;

    int best_idx = -1;
    int best_distance = std::numeric_limits<int>::max();

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

        const int descriptor_distance = descriptorDistanceFast(
            descriptors.row(feature_idx), map_descriptor);

        if (result != nullptr)
            result->fusion_descriptor_comparison_num++;

        constexpr int kFuseMaxHammingDistance = 50;
        if (descriptor_distance > kFuseMaxHammingDistance)
            continue;

        if (descriptor_distance < best_distance)
        {
            best_distance = descriptor_distance;
            best_idx = feature_idx;
        }
    }

    if (best_idx < 0)
        return -1;

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
    const std::shared_ptr<Frame>& target_keyframe,
    LocalMappingResult* result) const
{
    if (target_keyframe == nullptr || !target_keyframe->isKeyframe() || source_map_points.empty())
        return 0;

    const std::vector<std::shared_ptr<Feature>>& target_features = target_keyframe->getFeatures();
    if (target_features.empty() || target_keyframe->getDescriptors().empty())
        return 0;

    FusionProjectionContext projection_context;
    projection_context.camera = target_keyframe->getCamera();
    target_keyframe->copyPose(projection_context.R_cw, projection_context.t_cw);
    projection_context.camera_center = target_keyframe->getCameraCenter();
    projection_context.image_size = target_keyframe->getImg().size();
    if (projection_context.camera == nullptr || projection_context.R_cw.empty() ||
        projection_context.t_cw.empty() || projection_context.image_size.width <= 0 ||
        projection_context.image_size.height <= 0)
    {
        return 0;
    }

    std::unordered_set<int> used_feature_indices;
    used_feature_indices.reserve(target_features.size());
    std::vector<int> candidate_indices;
    candidate_indices.reserve(32);

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
        if (result != nullptr)
            result->fusion_projection_attempt_num++;
        if (!projectionMapPointToFrame(source_map_point, 
                                        projection_context,
                                        projected_pixel, 
                                        camera_distance, 
                                        pred_level))
        {
            continue;
        }

        if (result != nullptr)
        {
            result->fusion_projection_valid_num++;
            result->fusion_descriptor_query_num++;
        }

        cv::Mat map_descriptor;
        if (!getMapPointDescriptor(source_map_point, map_descriptor))
            continue;

        const int matched_feature_idx = findFuseMatchInKeyframe(source_map_point, 
                                                                target_keyframe, 
                                                                map_descriptor,
                                                                projected_pixel, 
                                                                pred_level, 
                                                                used_feature_indices,
                                                                candidate_indices,
                                                                result);
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
                                           const std::shared_ptr<Frame>& cur_frame,
                                           LocalMappingResult* result) const
{
    if (map == nullptr || cur_frame == nullptr || !cur_frame->isKeyframe())
        return 0;

    const auto context_start = std::chrono::steady_clock::now();
    const std::vector<std::shared_ptr<Frame>> neighbor_keyframes = 
        collectFusionKeyframes(cur_frame);

    if (result != nullptr)
        result->fusion_target_keyframe_num += neighbor_keyframes.size() * 2;

    if (neighbor_keyframes.empty())
        return 0;

    const std::vector<std::shared_ptr<MapPoint>> cur_map_points = 
        collectUniqueMapPointsFromKeyframe(cur_frame);

    if (result != nullptr)
        result->fusion_source_map_point_num += cur_map_points.size();

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
        if (result != nullptr)
            result->fusion_source_map_point_num += neighbor_map_points_sets.back().size();
    }

    const double fusion_context_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - context_start).count();
    if (result != nullptr)
        result->fusion_context_duration_ms = fusion_context_ms;

    std::size_t fused_num = 0;

    const auto forward_fusion_start = std::chrono::steady_clock::now();
    for (const auto& neighbor_keyframe : neighbor_keyframes)
        fused_num += fuseMapPointsIntoKeyframe(cur_map_points, neighbor_keyframe, result);
    const double forward_fusion_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - forward_fusion_start).count();

    const auto reverse_fusion_start = std::chrono::steady_clock::now();
    for (int i = 0; i < neighbor_keyframes.size(); i++)
    {
        fused_num += fuseMapPointsIntoKeyframe(neighbor_map_points_sets[i], cur_frame, result);
    }
    const double reverse_fusion_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - reverse_fusion_start).count();
    if (result != nullptr)
        result->fusion_forward_duration_ms = forward_fusion_ms;
    if (result != nullptr)
        result->fusion_reverse_duration_ms = reverse_fusion_ms;

    const auto graph_refresh_start = std::chrono::steady_clock::now();
    updateCovisibilityGraph(map, cur_frame);
    const double graph_refresh_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - graph_refresh_start).count();
    if (result != nullptr)
        result->fusion_post_graph_duration_ms = graph_refresh_ms;

    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-SLAM-PERF fusion_sections keyframe=" << cur_frame->getId()
        << " neighbors=" << neighbor_keyframes.size()
        << " current_map_points=" << cur_map_points.size()
        << " context_ms=" << fusion_context_ms
        << " forward_ms=" << forward_fusion_ms
        << " reverse_ms=" << reverse_fusion_ms
        << " graph_refresh_ms=" << graph_refresh_ms
        << " fused=" << fused_num);

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
    const std::shared_ptr<Map>& map,
    LocalMappingResult* result) const
{
    KeyframeRedundancyStats stats;

    if (keyframe == nullptr || map == nullptr)
        return stats;

    constexpr std::size_t kMinRedundantObservations = 3;

    for (const auto& feature : keyframe->getFeatures())
    {
        if (result != nullptr)
            result->keyframe_cull_feature_num++;

        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::vector<std::shared_ptr<Feature>> observations =
            map_point->getKeyframeObservations();

        if (result != nullptr)
            result->keyframe_cull_observation_num += observations.size();

        stats.total_map_features++;

        if (observations.size() <= kMinRedundantObservations)
            continue;

        const int current_level = feature->getLevel();
        std::size_t support_observation_num = 0;

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
                                      KeyframeRedundancyStats* stats,
                                      LocalMappingResult* result) const
{
    if (keyframe == nullptr || map == nullptr)
        return false;

    const std::vector<std::shared_ptr<Frame>>& keyframes = map->getKeyframes();
    if (keyframes.size() <= 2)
        return false;

    if (keyframe == keyframes.front() || keyframe == keyframes.back())
        return false;

    const KeyframeRedundancyStats evaluated_stats =
        evaluateKeyframeRedundancy(keyframe, map, result);
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
                                       const std::shared_ptr<Frame>& cur_keyframe,
                                       LocalMappingResult* result) const
{
    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return 0;

    const auto candidate_collection_start = std::chrono::steady_clock::now();
    const std::vector<std::shared_ptr<Frame>> candidate =
        collectKeyframeCullingCandidates(cur_keyframe);
    const double candidate_collection_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - candidate_collection_start).count();

    if (result != nullptr)
    {
        result->keyframe_cull_candidate_num += candidate.size();
        result->keyframe_cull_candidate_collection_duration_ms = candidate_collection_ms;
    }


    if (candidate.empty())
        return 0;

    constexpr std::size_t kMinActiveKeyframes = 10;
    std::size_t culled_keyframe_num = 0;

    for (const auto& keyframe : candidate)
    {
        if (map->getKeyframeNum() <= kMinActiveKeyframes)
        {
            break;
        }

        KeyframeRedundancyStats redundancy_stats;
        const auto evaluation_start = std::chrono::steady_clock::now();
        const bool redundant = isKeyframeRedundant(keyframe, map, &redundancy_stats, result);
        if (result != nullptr)
        {
            result->keyframe_cull_evaluation_duration_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - evaluation_start).count();
        }
        ROS_DEBUG_STREAM("P2-SLAM-DEBUG mini_kfcull current=" << cur_keyframe->getId()
                        << " candidate=" << keyframe->getId()
                        << " map_keyframes=" << map->getKeyframeNum()
                        << " total_features=" << redundancy_stats.total_map_features
                        << " redundant_features=" << redundancy_stats.redundant_map_features
                        << " duplicate_keyframe_observations="
                        << redundancy_stats.duplicate_keyframe_observations
                        << " redundant_ratio=" << redundancy_stats.redundant_ratio
                        << " redundant=" << (redundant ? 1 : 0));
        if (!redundant)
            continue;

        const auto removal_start = std::chrono::steady_clock::now();
        map->removeKeyframe(keyframe);
        if (result != nullptr)
        {
            result->keyframe_cull_removal_duration_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - removal_start).count();
        }
        culled_keyframe_num++;
    }

    return culled_keyframe_num;
}

} // namespace mini_orb_slam
