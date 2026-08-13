#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core.hpp>

#include "loop_closer.h"
#include "local_mapper.h"

namespace mini_orb_slam
{

LoopCloser::LoopCloser(const std::shared_ptr<KeyframeDatabase>& keyframe_database,
                       const Matcher& matcher,
                       const std::shared_ptr<PoseOptimizer>& pose_optimizer,
                       LocalMapper* local_mapper,
                       double scale_factor,
                       int levels_num)
    : keyframe_database_(keyframe_database),
      matcher_(matcher),
      pose_optimizer_(pose_optimizer),
      local_mapper_(local_mapper),
      scale_factor_(scale_factor),
      levels_num_(levels_num) {}

LoopCloser::~LoopCloser()
{
    requestFinish();
    join();
}

void LoopCloser::start()
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

    worker_thread_ = std::thread(&LoopCloser::run, this);
}

void LoopCloser::requestFinish()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        finish_requested_ = true;
        accept_keyframes_ = false;
    }

    queue_cv_.notify_all();
}

void LoopCloser::join()
{
    if (worker_thread_.joinable())
        worker_thread_.join();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    worker_started_ = false;
}

bool LoopCloser::insertKeyframe(const LoopClosingInput& input)
{
    if (input.map == nullptr || input.cur_keyframe == nullptr || !input.cur_keyframe->isKeyframe())
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

bool LoopCloser::tryPopFinishedResult(LoopClosingOutput& output)
{
    LoopClosingOutput ready_output;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (finished_results_.empty())
            return false;

        ready_output = std::move(finished_results_.front());
        finished_results_.pop_front();
    }

    output = std::move(ready_output);
    return true;
}

bool LoopCloser::acceptKeyframe() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return accept_keyframes_ &&
           !finish_requested_ &&
           !stop_requested_ &&
           pending_keyframes_.empty();
}

bool LoopCloser::isStopped() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return stopped_;
}

bool LoopCloser::stopRequested() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return stop_requested_;
}

void LoopCloser::run()
{
    while (true)
    {
        LoopClosingInput input;

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

        LoopClosingOutput output;
        output.input = input;

        if (input.map != nullptr && input.cur_keyframe != nullptr && input.cur_keyframe->isKeyframe())
        {
            output.verification_result = detectAndVerifyLoop(input.map, input.cur_keyframe);

            if (output.verification_result.success &&
                output.verification_result.candidate_keyframe != nullptr)
            {
                if (local_mapper_ != nullptr)
                {
                    local_mapper_->requestStop();
                    while (!local_mapper_->isStopped() && !finish_requested_)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                {
                    std::lock_guard<std::mutex> map_lock(input.map->getMutex());
                    const std::size_t refreshed_constraints =
                        input.map->refreshPoseGraphMeasurements();
                    ROS_INFO_STREAM("P2-SLAM-GRAPH graph_measurements_refreshed="
                                    << refreshed_constraints
                                    << " loop_keyframe=" << input.cur_keyframe->getId());
                    output.correction_result =
                        applyVerifiedLoop(input.map, input.cur_keyframe, output.verification_result);
                    if (output.correction_result.success &&
                        output.correction_result.registered_loop_edge &&
                        pose_optimizer_ != nullptr)
                    {
                        const std::vector<std::shared_ptr<Frame>> map_keyframes =
                            input.map->getKeyframes();
                        const std::vector<std::shared_ptr<MapPoint>> map_points =
                            input.map->getMapPoints();
                        const std::vector<PoseGraphConstraint> constraints =
                            input.map->getPoseGraphConstraints();
                        std::shared_ptr<Frame> anchor_keyframe;

                        for (const auto& keyframe : map_keyframes)
                        {
                            if (keyframe != nullptr && keyframe->isKeyframe())
                            {
                                anchor_keyframe = keyframe;
                                break;
                            }
                        }

                        if (anchor_keyframe != nullptr)
                        {
                            output.graph_optimized =
                                pose_optimizer_->optimizeEssentialGraph(map_keyframes,
                                                                        map_points,
                                                                        constraints,
                                                                        anchor_keyframe);
                            if (output.graph_optimized)
                            {
                                input.map->refreshPoseGraphMeasurements();
                                input.map->markModified();
                            }
                        }
                    }
                }

                if (local_mapper_ != nullptr)
                    local_mapper_->release();
            }
        }

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
    
std::unordered_set<std::size_t> LoopCloser::collectConnectedKeyframeIds(
    const std::shared_ptr<Frame>& keyframe) const
{
    std::unordered_set<std::size_t> excluded_ids;
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return excluded_ids;

    excluded_ids.reserve(16);
    excluded_ids.insert(keyframe->getId());

    const std::vector<std::shared_ptr<Frame>> neighbors =
        keyframe->copyConnectedKeyframes(1);

    for (const auto& neighbor : neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe())
            continue;

        excluded_ids.insert(neighbor->getId());
    }

    return excluded_ids;
}

bool LoopCloser::buildLoopPnPInput(const std::shared_ptr<Frame>& candidate_keyframe,
                                  const std::shared_ptr<Frame>& cur_keyframe,
                                  PnPResult& pnp_input,
                                  int& raw_match_num) const
{
    pnp_input = {};
    raw_match_num = 0;

    if (candidate_keyframe == nullptr || cur_keyframe == nullptr ||
        !candidate_keyframe->hasFeatures() || !cur_keyframe->hasFeatures())
    {
        return false;
    }

    const std::vector<std::pair<int, int>> match_indices = 
        matcher_.matchFramesByBoW(*candidate_keyframe, *cur_keyframe);

    raw_match_num = match_indices.size();
    if (raw_match_num < 15)
        return false;

    const std::vector<std::shared_ptr<Feature>>& candidate_features = 
        candidate_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = 
        cur_keyframe->getFeatures();

    std::unordered_set<std::size_t> used_map_point_ids;
    used_map_point_ids.reserve(raw_match_num);

    pnp_input.object_points.reserve(match_indices.size());
    pnp_input.img_points.reserve(match_indices.size());
    pnp_input.candidate_map_points.reserve(match_indices.size());
    pnp_input.candidate_features.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int candidate_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (candidate_idx < 0 ||
            candidate_idx >= candidate_features.size() ||
            cur_idx < 0 ||
            cur_idx >= cur_features.size())
        {
            continue;
        }

        const std::shared_ptr<Feature>& candidate_feature = candidate_features[candidate_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (candidate_feature == nullptr || 
            cur_feature == nullptr || 
            !candidate_feature->hasMapPoint())
        {
            continue;
        }

        const std::shared_ptr<MapPoint> map_point = candidate_feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!used_map_point_ids.insert(map_point->getId()).second)
            continue;

        pnp_input.object_points.push_back(map_point->getPos());
        pnp_input.img_points.push_back(cur_feature->getKeyPoint().pt);
        pnp_input.candidate_map_points.push_back(map_point);
        pnp_input.candidate_features.push_back(candidate_feature);
    }

    if (pnp_input.object_points.size() < 4)
        return false;

    return setPoseGuessFromFrame(cur_keyframe, pnp_input);
}

bool LoopCloser::setPoseGuessFromFrame(const std::shared_ptr<Frame>& frame,
                                       PnPResult& pnp_input) const
{
    if (frame == nullptr)
        return false;

    cv::Mat R_cw;
    cv::Mat t_cw;
    frame->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return false;

    cv::Mat rvec;
    cv::Rodrigues(R_cw, rvec);

    rvec.convertTo(pnp_input.rvec, CV_64F);
    t_cw.convertTo(pnp_input.tvec, CV_64F);
    return true;
}

bool LoopCloser::isLoopCandidateAccepted(const PnPResult& pnp_result) const
{
    if (!pnp_result.success)
        return false;

    if (pnp_result.inlier_num < 20)
        return false;

    const std::size_t candidate_num = pnp_result.object_points.size();
    if (candidate_num == 0)
        return false;
    
    const double inlier_ratio = static_cast<double>(pnp_result.inlier_num) / candidate_num;
    if (inlier_ratio < 0.3)
        return false;

    const double reproj_error = 
        (pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0)
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;
    if (reproj_error <= 0.0 || reproj_error > 8.0)
        return false;

    return true;
}

double LoopCloser::computeLoopCandidateScore(const PnPResult& pnp_result) const
{
    const std::size_t candidate_num = pnp_result.object_points.size();
    if (candidate_num == 0)
        return 0.0;

    const double inlier_ratio = static_cast<double>(pnp_result.inlier_num) / candidate_num;

    const double reproj_error = 
        (pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0)
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;

    if (reproj_error <= 0.0)
        return 0.0;

    return static_cast<double>(pnp_result.inlier_num) * (1.0 + inlier_ratio) / reproj_error;
}

std::vector<std::shared_ptr<Frame>> LoopCloser::detectLoopCandidates(
    const std::shared_ptr<Map>& map, 
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<std::shared_ptr<Frame>> candidates;

    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe() ||
        keyframe_database_ == nullptr || !cur_keyframe->hasBoW())
    {
        consistent_loop_groups_.clear();
        return candidates;
    }

    constexpr std::size_t kMaxSeedCandiates = 20;
    constexpr std::size_t kMaxLoopCandidates = 10;
    constexpr std::size_t kMaxNeighborNum = 10;
    constexpr double kMinBowScore = 0.015;
    constexpr double kNeighborScoreWeight = 0.75;
    constexpr double kKeepBestGroupRatio = 0.75;

    const std::unordered_set<std::size_t> excluded_ids = 
        collectConnectedKeyframeIds(cur_keyframe);

    const std::vector<KeyframeQueryResult> bow_results = 
        keyframe_database_->query(cur_keyframe, kMaxSeedCandiates, kMinBowScore);

    if (bow_results.empty())
    {
        consistent_loop_groups_.clear();
        return candidates;
    }

    std::unordered_map<std::size_t, double> seed_scores;
    seed_scores.reserve(bow_results.size() * 2 + 1);

    for (const auto& result : bow_results)
    {
        if (result.keyframe == nullptr || !result.keyframe->isKeyframe())
            continue;

        if (excluded_ids.count(result.keyframe->getId()) > 0)
            continue;

        seed_scores[result.keyframe->getId()] = result.score;
    }

    if (seed_scores.empty())
    {
        consistent_loop_groups_.clear();
        return candidates;
    }

    std::unordered_map<std::size_t, LoopCandidateGroup> grouped_candidates;
    grouped_candidates.reserve(seed_scores.size() * 2 + 1);

    double best_group_score = 0.0;

    for (const auto& result : bow_results)
    {
        if (result.keyframe == nullptr || !result.keyframe->isKeyframe())
            continue;

        if (excluded_ids.count(result.keyframe->getId()) > 0)
            continue;

        std::shared_ptr<Frame> best_group_keyframe = result.keyframe;
        double best_group_bow_score = result.score;
        double group_score = result.score;
        int support_num = 0;

        const std::vector<std::shared_ptr<Frame>> neighbors = 
            result.keyframe->copyBestCovisibilityKeyframes(kMaxNeighborNum, 1);

        for (const auto& neighbor : neighbors)
        {
            if (neighbor == nullptr || !neighbor->isKeyframe())
                continue;

            if (excluded_ids.count(neighbor->getId()) > 0)
                continue;

            const auto it = seed_scores.find(neighbor->getId());
            if (it == seed_scores.end())
                continue;

            group_score += kNeighborScoreWeight * it->second;
            support_num++;

            if (it->second > best_group_bow_score)
            {
                best_group_bow_score = it->second;
                best_group_keyframe = neighbor;
            }
        }

        if (best_group_keyframe == nullptr)
            continue;

        auto group_it = grouped_candidates.find(best_group_keyframe->getId());
        if (group_it == grouped_candidates.end() ||
            group_score > group_it->second.group_score)
        {
            grouped_candidates[best_group_keyframe->getId()] = 
            {
                best_group_keyframe,
                best_group_bow_score,
                group_score,
                support_num
            };
        }

        if (group_score > best_group_score)
            best_group_score = group_score;
    }

    std::vector<LoopCandidateGroup> ranked_groups;
    ranked_groups.reserve(grouped_candidates.size());

    for (const auto& item : grouped_candidates)
        ranked_groups.push_back(item.second);

    std::vector<ConsistentLoopGroup> current_consistent_groups;
    current_consistent_groups.reserve(ranked_groups.size());
    for (auto& group : ranked_groups)
    {
        if (group.representative == nullptr)
            continue;

        ConsistentLoopGroup current_group;
        current_group.keyframe_ids.insert(group.representative->getId());
        for (const auto& neighbor :
             group.representative->copyBestCovisibilityKeyframes(kMaxNeighborNum, 1))
        {
            if (neighbor != nullptr && neighbor->isKeyframe())
                current_group.keyframe_ids.insert(neighbor->getId());
        }

        int previous_consistency = 0;
        for (const auto& previous_group : consistent_loop_groups_)
        {
            bool overlaps = false;
            for (const std::size_t id : current_group.keyframe_ids)
            {
                if (previous_group.keyframe_ids.count(id) > 0)
                {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps)
                previous_consistency = std::max(previous_consistency,
                                                previous_group.consistency + 1);
        }

        current_group.consistency = previous_consistency;
        group.consistency = current_group.consistency;
        current_consistent_groups.push_back(std::move(current_group));
    }
    consistent_loop_groups_ = std::move(current_consistent_groups);

    std::sort(ranked_groups.begin(), ranked_groups.end(),
                [](const LoopCandidateGroup& a, const LoopCandidateGroup& b) 
                {
                    if (a.group_score != b.group_score)
                        return a.group_score > b.group_score;

                    if (a.bow_score != b.bow_score)
                        return a.bow_score > b.bow_score;

                    if (a.support_num != b.support_num)
                        return a.support_num > b.support_num;

                    return a.representative->getId() > b.representative->getId();
                });

    const double min_keep_score = best_group_score * kKeepBestGroupRatio;

    for (const auto& group : ranked_groups)
    {
        if (group.representative == nullptr)
            continue;

        if (group.group_score < min_keep_score)
            break;

        if (group.consistency < 2)
            continue;

        candidates.push_back(group.representative);

        if (candidates.size() >= kMaxLoopCandidates)
            break;
    }

    return candidates;
}

bool LoopCloser::worldPointToCameraPoint(const std::shared_ptr<Frame>& frame,
                                         const cv::Point3d& point_world,
                                         cv::Point3d& point_camera) const
{
    point_camera = cv::Point3d(0.0, 0.0, 0.0);

    if (frame == nullptr)
        return false;

    cv::Mat R_cw;
    cv::Mat t_cw;
    frame->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return false;

    const cv::Mat point_world_mat = 
    (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);

    const cv::Mat point_camera_mat = R_cw * point_world_mat + t_cw;

    point_camera.x = point_camera_mat.at<double>(0, 0);
    point_camera.y = point_camera_mat.at<double>(1, 0);
    point_camera.z = point_camera_mat.at<double>(2, 0);

    return point_camera.z > 1e-6;
}

std::vector<LoopSim3Correspondence> LoopCloser::collectLoopSim3Correspondence(
    const std::shared_ptr<Frame>& candidate_keyframe,
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<LoopSim3Correspondence> correspondences;

    if (candidate_keyframe == nullptr || cur_keyframe == nullptr ||
        !candidate_keyframe->hasFeatures() || !cur_keyframe->hasFeatures())
    {
        return correspondences;
    }

    const std::vector<std::pair<int, int>> match_indices = 
        matcher_.matchFramesByBoW(*candidate_keyframe, *cur_keyframe);

    if (match_indices.empty())
        return correspondences;

    const std::vector<std::shared_ptr<Feature>>& candidate_features = 
        candidate_keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = 
        cur_keyframe->getFeatures();

    std::unordered_set<std::size_t> used_candidate_ids;
    std::unordered_set<std::size_t> used_cur_ids;
    used_candidate_ids.reserve(match_indices.size());
    used_cur_ids.reserve(match_indices.size());

    correspondences.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int candidate_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (candidate_idx < 0 || candidate_idx >= candidate_features.size() ||
            cur_idx < 0 || cur_idx >= cur_features.size())
        {
            continue;
        }

        const std::shared_ptr<Feature>& candidate_feature = candidate_features[candidate_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (candidate_feature == nullptr || cur_feature == nullptr || 
            !candidate_feature->hasMapPoint() || !cur_feature->hasMapPoint())
        {
            continue;
        }

        const std::shared_ptr<MapPoint> candidate_map_point = candidate_feature->getMapPoint();
        const std::shared_ptr<MapPoint> cur_map_point = cur_feature->getMapPoint();

        if (candidate_map_point == nullptr || cur_map_point == nullptr ||
            candidate_map_point->isBad() || cur_map_point->isBad())
        {
            continue;
        }

        if (!used_candidate_ids.insert(candidate_map_point->getId()).second ||
            !used_cur_ids.insert(cur_map_point->getId()).second)
        {
            continue;
        }

        cv::Point3d candidate_pos_camera;
        cv::Point3d cur_pos_camera;

        if (!worldPointToCameraPoint(candidate_keyframe, 
                                     candidate_map_point->getPos(), 
                                     candidate_pos_camera))
        {
            continue;
        }

        if (!worldPointToCameraPoint(cur_keyframe, 
                                     cur_map_point->getPos(), 
                                     cur_pos_camera))
        {
            continue;
        }

        correspondences.push_back({
            candidate_map_point,
            cur_map_point,
            candidate_map_point->getPos(),
            cur_map_point->getPos(),
            candidate_pos_camera,
            cur_pos_camera,
            candidate_idx,
            cur_idx
        });
    }

    return correspondences;
}

bool LoopCloser::estimateSim3ByUmeyama(const std::vector<cv::Point3d>& src_points,
                                       const std::vector<cv::Point3d>& dst_points,
                                       double& scale,
                                       cv::Mat& R,
                                       cv::Mat& t) const
{
    scale = 1.0;
    R = cv::Mat();
    t = cv::Mat();

    if (src_points.size() < 3 || src_points.size() != dst_points.size())
        return false;

    cv::Point3d mean_src(0.0, 0.0, 0.0);
    cv::Point3d mean_dst(0.0, 0.0, 0.0);

    for (std::size_t i = 0; i < src_points.size(); i++)
    {
        mean_src += src_points[i];
        mean_dst += dst_points[i];
    }

    const double inv_n = 1.0 / static_cast<double>(src_points.size());
    mean_src *= inv_n;
    mean_dst *= inv_n;

    cv::Mat Sigma = cv::Mat::zeros(3, 3, CV_64F);
    double src_var = 0.0;

    for (std::size_t i = 0; i < src_points.size(); i++)
    {
        const cv::Vec3d xs(src_points[i].x - mean_src.x,
                           src_points[i].y - mean_src.y,
                           src_points[i].z - mean_src.z);
        const cv::Vec3d yd(dst_points[i].x - mean_dst.x,
                           dst_points[i].y - mean_dst.y,
                           dst_points[i].z - mean_dst.z);

        Sigma += (cv::Mat_<double>(3, 3) <<
                    yd[0] * xs[0], yd[0] * xs[1], yd[0] * xs[2],
                    yd[1] * xs[0], yd[1] * xs[1], yd[1] * xs[2],
                    yd[2] * xs[0], yd[2] * xs[1], yd[2] * xs[2]);

        src_var += xs.dot(xs);
    }

    Sigma *= inv_n;
    src_var *= inv_n;

    if (src_var <= 1e-12)
        return false;

    cv::Mat W, U, Vt;
    cv::SVD::compute(Sigma, W, U, Vt);

    cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
    if (cv::determinant(U * Vt) < 0.0)
        S.at<double>(2, 2) = -1.0;

    R = U * S * Vt;

    const double trace_num = 
        W.at<double>(0, 0) * S.at<double>(0, 0) + 
        W.at<double>(1, 0) * S.at<double>(1, 1) +
        W.at<double>(2, 0) * S.at<double>(2, 2);

    scale = trace_num / src_var;
    if (scale <= 0.0)
        return false;

    const cv::Mat mean_src_mat = (cv::Mat_<double>(3, 1) << mean_src.x, mean_src.y, mean_src.z);
    const cv::Mat mean_dst_mat = (cv::Mat_<double>(3, 1) << mean_dst.x, mean_dst.y, mean_dst.z);

    t = mean_dst_mat - scale * R * mean_src_mat;
    return true;
}

double LoopCloser::computeSim3PointError(const LoopSim3Correspondence& correspondence,
                                         const double scale,
                                         const cv::Mat& R,
                                         const cv::Mat& t) const
{
    const cv::Mat src = 
        (cv::Mat_<double>(3, 1) << correspondence.candidate_point_camera.x,
                                   correspondence.candidate_point_camera.y,
                                   correspondence.candidate_point_camera.z);

    const cv::Mat dst =
        (cv::Mat_<double>(3, 1) << correspondence.current_point_camera.x,
                                   correspondence.current_point_camera.y,
                                   correspondence.current_point_camera.z);

    const cv::Mat pred = scale * R * src + t;
    return cv::norm(pred - dst);
}

LoopSim3Result LoopCloser::estimateLoopSim3(
    const std::vector<LoopSim3Correspondence>& correspondences) const
{
    LoopSim3Result best_result;

    constexpr int kRansacIterations = 100;
    constexpr double kInlierErrorThreshold = 0.3;
    constexpr int kMinSim3Inliers = 10;

    if (correspondences.size() < kMinSim3Inliers)
        return best_result;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, correspondences.size() - 1);

    double best_mean_error = 1e18;

    for (int iter = 0; iter < kRansacIterations; iter++)
    {
        std::set<int> sample_ids;
        while (sample_ids.size() < 3)
            sample_ids.insert(dist(rng));

        std::vector<cv::Point3d> sample_src;
        std::vector<cv::Point3d> sample_dst;
        sample_src.reserve(3);
        sample_dst.reserve(3);

        for (const int idx : sample_ids)
        {
            sample_src.push_back(correspondences[idx].candidate_point_camera);
            sample_dst.push_back(correspondences[idx].current_point_camera);
        }

        double scale;
        cv::Mat R, t;
        if (!estimateSim3ByUmeyama(sample_src, sample_dst, scale, R, t))
            continue;

        std::vector<int> inlier_indices;
        inlier_indices.reserve(correspondences.size());

        double total_error = 0.0;

        for (int i = 0; i < correspondences.size(); i++)
        {
            const double error = computeSim3PointError(correspondences[i], scale, R, t);
            if (error > kInlierErrorThreshold)
                continue;

            inlier_indices.push_back(i);
            total_error += error;
        }

        if (inlier_indices.size() < kMinSim3Inliers)
            continue;

        const double mean_error = total_error / inlier_indices.size();

        if (!best_result.success || 
            inlier_indices.size() > best_result.inlier_num || 
            inlier_indices.size() == best_result.inlier_num &&
            mean_error < best_mean_error)
        {
            best_result.success = true;
            best_result.scale = scale;
            best_result.R = R.clone();
            best_result.t = t.clone();
            best_result.inlier_num = inlier_indices.size();
            best_result.inlier_indices = inlier_indices;
            best_result.mean_error = mean_error;
            best_mean_error = mean_error;
        }
    }

    if (!best_result.success)
        return best_result;

    std::vector<cv::Point3d> refine_src;
    std::vector<cv::Point3d> refine_dst;
    refine_src.reserve(best_result.inlier_num);
    refine_dst.reserve(best_result.inlier_num);

    for (const int idx : best_result.inlier_indices)
    {
        refine_src.push_back(correspondences[idx].candidate_point_camera);
        refine_dst.push_back(correspondences[idx].current_point_camera);
    }

    double refine_scale = 1.0;
    cv::Mat refine_R, refine_t;
    if (estimateSim3ByUmeyama(refine_src, refine_dst, refine_scale, refine_R, refine_t))
    {
        double total_error = 0.0;
        for (const int idx : best_result.inlier_indices)
            total_error += computeSim3PointError(correspondences[idx], refine_scale, refine_R, refine_t);

        best_result.scale = refine_scale;
        best_result.R = refine_R.clone();
        best_result.t = refine_t.clone();
        best_result.mean_error = total_error / best_result.inlier_num;
    }

    best_result.inlier_ratio = static_cast<double>(best_result.inlier_num) / correspondences.size();

    return best_result;
}

bool LoopCloser::isLoopSim3Accepted(const LoopSim3Result& sim3_result,
                                    std::size_t correspondence_num) const
{
    if (!sim3_result.success)
        return false;

    if (sim3_result.inlier_num < 20)
        return false;

    if (correspondence_num == 0)
        return false;

    if (sim3_result.inlier_ratio < 0.35)
        return false;

    if (sim3_result.mean_error <= 0.0 || sim3_result.mean_error > 0.3)
        return false;

    return true;
}

double LoopCloser::computeLoopSim3Score(const LoopSim3Result& sim3_result,
                                             std::size_t correspondence_num) const
{
    if (!sim3_result.success || correspondence_num == 0 || sim3_result.mean_error <= 0.0)
        return 0.0;

    return static_cast<double>(sim3_result.inlier_num) * 
        (1.0 + sim3_result.inlier_ratio) / sim3_result.mean_error;
}

LoopVerificationResult LoopCloser::detectAndVerifyLoop(const std::shared_ptr<Map>& map, 
                                                       const std::shared_ptr<Frame>& cur_keyframe) const
{
    LoopVerificationResult best_result;

    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return best_result;

    const std::vector<std::shared_ptr<Frame>> candidates = detectLoopCandidates(map, cur_keyframe);

    double best_score = 0.0;

    for (const auto& candidate : candidates)
    {
        if (candidate == nullptr || candidate == cur_keyframe)
            continue;

        PnPResult pnp_input;
        int raw_match_num = 0;
        if (!buildLoopPnPInput(candidate, cur_keyframe, pnp_input, raw_match_num))
            continue;

        const std::vector<LoopSim3Correspondence> sim3_correspondences = 
            collectLoopSim3Correspondence(candidate, cur_keyframe);

        if (sim3_correspondences.size() < 10)
            continue;

        const PnPResult optimized_result = pose_optimizer_->optimize(pnp_input);
        if (!isLoopCandidateAccepted(optimized_result))
            continue;

        const LoopSim3Result sim3_result = estimateLoopSim3(sim3_correspondences);
        if (!isLoopSim3Accepted(sim3_result, sim3_correspondences.size()))
            continue;

        const double candidate_score = 
            computeLoopSim3Score(sim3_result, sim3_correspondences.size());

        if (!best_result.success || candidate_score > best_score)
        {
            best_score = candidate_score;
            best_result.success = true;
            best_result.candidate_keyframe = candidate;
            best_result.raw_match_num = raw_match_num;
            best_result.pnp_inlier_num = optimized_result.inlier_num;
            best_result.inlier_ratio = optimized_result.object_points.empty() 
                                            ? 0.0
                                            : static_cast<double>(optimized_result.inlier_num) / 
                                                optimized_result.object_points.size();
            best_result.reproj_error = (optimized_result.optimized &&
                                        optimized_result.optimized_reproj_error > 0.0)
                                            ? optimized_result.optimized_reproj_error
                                            : optimized_result.ransac_reproj_error;
            best_result.pnp_result = optimized_result;
            best_result.sim3_correspondences = sim3_correspondences;
            best_result.sim3_result = sim3_result;
        }
    }

    return best_result;
}
    
std::shared_ptr<MapPoint> LoopCloser::chooseDominantMapPoint
    (const std::shared_ptr<MapPoint>& lhs,
     const std::shared_ptr<MapPoint>& rhs) const
{
    if (lhs == nullptr || lhs->isBad())
        return rhs;

    if (rhs == nullptr || rhs->isBad())
        return lhs;

    const std::size_t lhs_kf_obs = lhs->getKeyframeObservationCount();
    const std::size_t rhs_kf_obs = rhs->getKeyframeObservationCount();
    if (lhs_kf_obs != rhs_kf_obs)
        return (lhs_kf_obs > rhs_kf_obs) ? lhs : rhs;

    const std::size_t lhs_obs = lhs->getObservationCount();
    const std::size_t rhs_obs = rhs->getObservationCount();
    if (lhs_obs != rhs_obs)
        return (lhs_obs > rhs_obs) ? lhs : rhs;

    if (lhs->getFoundTimes() != rhs->getFoundTimes())
        return (lhs->getFoundTimes() > rhs->getFoundTimes()) ? lhs : rhs;

    return (lhs->getId() < rhs->getId()) ? lhs : rhs;
}

void LoopCloser::mergeMapPoints(const std::shared_ptr<MapPoint>& keep_point,
                                const std::shared_ptr<MapPoint>& remove_point) const
{
    if (keep_point == nullptr || remove_point == nullptr || keep_point == remove_point)
        return;

    const std::vector<std::shared_ptr<Feature>> observations = remove_point->getObservations();
    for (const auto& feature : observations)
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<MapPoint> current_map_point = feature->getMapPoint();
        if (current_map_point != nullptr &&
            current_map_point != keep_point &&
            current_map_point != remove_point)
        {
            continue;
        }

        feature->setMapPoint(keep_point);
        keep_point->addObservation(feature);
    }

    if (keep_point->getRefFeature() == nullptr)
        keep_point->setRefFeature(remove_point->getRefFeature());

    if (keep_point->getCurFeature() == nullptr)
        keep_point->setCurFeature(remove_point->getCurFeature());

    remove_point->setBad(true);
}

bool LoopCloser::computeRelativePoseBetweenFrames(const std::shared_ptr<Frame>& from_keyframe,
                                                  const std::shared_ptr<Frame>& to_keyframe,
                                                  cv::Mat& R_21, 
                                                  cv::Mat& t_21) const
{
    R_21 = cv::Mat();
    t_21 = cv::Mat();

    if (from_keyframe == nullptr || to_keyframe == nullptr)
        return false;

    cv::Mat R_from, t_from;
    cv::Mat R_to, t_to;
    from_keyframe->copyPose(R_from, t_from);
    to_keyframe->copyPose(R_to, t_to);

    if (R_from.empty() || t_from.empty() || R_to.empty() || t_to.empty())
        return false;

    R_21 = R_to * R_from.t();
    t_21 = t_to - R_21 * t_from;
    return true;
}

bool LoopCloser::buildLoopPoseGraphConstraint(const LoopVerificationResult& loop_result,
                                              cv::Mat& R_21,
                                              cv::Mat& t_21,
                                              double& scale) const
{
    R_21.release();
    t_21.release();
    scale = 1.0;

    const LoopSim3Result& sim3 = loop_result.sim3_result;

    if (!loop_result.success || !sim3.success || 
        sim3.scale <= 1e-8 ||
        sim3.R.rows != 3 || sim3.R.cols != 3 ||
        sim3.t.rows != 3 || sim3.t.cols != 1)
    {
        return false;
    }

    scale = 1.0 / sim3.scale;
    R_21 = sim3.R.t();
    t_21 = -scale * R_21 * sim3.t;

    return true;
}

LoopCorrectionResult LoopCloser::applyVerifiedLoop(const std::shared_ptr<Map>& map,
                                                   const std::shared_ptr<Frame>& cur_keyframe,
                                                   const LoopVerificationResult& loop_result) const
{
    LoopCorrectionResult result;

    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe() ||
        !loop_result.success || loop_result.candidate_keyframe == nullptr)
    {
        return result;
    }

    double world_scale = 1.0;
    cv::Mat world_R, world_t;

    std::unordered_set<std::size_t> merged_remove_ids;
    std::unordered_set<std::size_t> touched_point_ids;
    std::unordered_set<std::size_t> touched_keyframe_ids;

    merged_remove_ids.reserve(loop_result.sim3_correspondences.size()* 2 + 1);
    touched_point_ids.reserve(loop_result.sim3_correspondences.size()* 2 + 1);
    touched_keyframe_ids.reserve(loop_result.sim3_correspondences.size()* 2 + 1);

    std::vector<std::shared_ptr<MapPoint>> touched_points;
    std::vector<std::shared_ptr<Frame>> touched_keyframes;

    auto touchPoint = [&](const std::shared_ptr<MapPoint>& map_point)
    {
        if (map_point == nullptr || map_point->isBad())
            return;

        if (!touched_point_ids.insert(map_point->getId()).second)
            return;

        touched_points.push_back(map_point);
    };

    auto touchKeyframe = [&](const std::shared_ptr<Frame>& keyframe)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe())
            return;

        if (!touched_keyframe_ids.insert(keyframe->getId()).second)
            return;

        touched_keyframes.push_back(keyframe);
    };

    touchKeyframe(cur_keyframe);
    touchKeyframe(loop_result.candidate_keyframe);

    for (const int idx : loop_result.sim3_result.inlier_indices)
    {
        if (idx < 0 || idx >= loop_result.sim3_correspondences.size())
            continue;

        const LoopSim3Correspondence corr = loop_result.sim3_correspondences[idx];
        
        const std::shared_ptr<MapPoint> candidate_point = corr.candidate_map_point;
        const std::shared_ptr<MapPoint> current_point = corr.current_map_point;

        if (candidate_point == nullptr || current_point == nullptr ||
            candidate_point->isBad() || current_point->isBad())
        {
            continue;
        }

        if (candidate_point == current_point)
        {
            touchPoint(candidate_point);

            const std::vector<std::shared_ptr<Feature>> observations = 
                candidate_point->getKeyframeObservations();

            for (const auto& obs : observations)
            {
                if (obs != nullptr)
                    touchKeyframe(obs->getFrame());
            }

            continue;
        }

        const std::shared_ptr<MapPoint> keep_point = 
            chooseDominantMapPoint(candidate_point, current_point);
        const std::shared_ptr<MapPoint> remove_point = 
            (keep_point == candidate_point) ? current_point : candidate_point;

        if (keep_point == nullptr || remove_point == nullptr || keep_point == remove_point)
            continue;

        if (!merged_remove_ids.insert(remove_point->getId()).second)
            continue;

        mergeMapPoints(keep_point, remove_point);
        touchPoint(keep_point);

        const std::vector<std::shared_ptr<Feature>> observations = 
            keep_point->getKeyframeObservations();

        for (const auto& obs : observations)
        {
            if (obs != nullptr)
                touchKeyframe(obs->getFrame());
        }

        result.fused_map_point_num++;
    }

    for (const auto& map_point : touched_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        map_point->updateViewStatistics(scale_factor_, levels_num_);
        map_point->updateRepresentativeDescriptor();
    }

    for (const auto& keyframe : touched_keyframes)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe())
            continue;

        keyframe->updateConnections();
    }

    map->removeBadMapPoints();

    result.updated_keyframe_num = touched_keyframes.size();
    result.registered_loop_edge = false;

    cv::Mat R_cand_cur;
    cv::Mat t_cand_cur;
    double scale_cand_cur = 1.0;
    
    if (buildLoopPoseGraphConstraint(loop_result, R_cand_cur, t_cand_cur, scale_cand_cur))
    {
        constexpr int kLoopEdgeWeight = 1000;

        const bool added = map->addPoseGraphConstraint(
            cur_keyframe, 
            loop_result.candidate_keyframe, 
            kLoopEdgeWeight, 
            R_cand_cur, 
            t_cand_cur, 
            scale_cand_cur, 
            PoseGraphConstraintKind::LOOP);

        result.registered_loop_edge = 
            added ||
            map->hasPoseGraphConstraint(cur_keyframe, 
                                        loop_result.candidate_keyframe, 
                                        PoseGraphConstraintKind::LOOP); 
    }

    result.success = result.registered_loop_edge;
    return result;
}

} // namespace mini_orb_slam
