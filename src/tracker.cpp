#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

#include "tracker.h"

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

} // namespace

Tracker::Tracker(const std::shared_ptr<Camera>& camera, 
                const Matcher& matcher, 
                const std::shared_ptr<PoseOptimizer>& pose_optimizer,
                double scale_factor,
                int levels_num,
                float base_projection_search_radius) 
    : camera_(camera), 
      matcher_(matcher), 
      pose_optimizer_(pose_optimizer),
      scale_factor_(scale_factor > 1.0 ? scale_factor : 1.2),
      levels_num_(levels_num > 0 ? levels_num : 8),
      base_projection_search_radius_(base_projection_search_radius > 0.0f ? base_projection_search_radius : 15.0f) {}

PnPResult Tracker::estimatePoseByPnP(const InitializationResult& init_result) const
{
    PnPResult result;

    if (!init_result.success || camera_ == nullptr)
        return result;

    result.object_points.reserve(init_result.map_points.size());
    result.img_points.reserve(init_result.map_points.size());
    result.candidate_map_points.reserve(init_result.map_points.size());
    result.candidate_features.reserve(init_result.map_points.size());

    for (const auto& map_point : init_result.map_points)
    {
        if (map_point == nullptr)
            continue;

        const std::shared_ptr<Feature>& cur_feature = map_point->getCurFeature();
        if (cur_feature == nullptr)
            continue;

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(cur_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(cur_feature);
    }

    if (pose_optimizer_ == nullptr)
        return {};

    return pose_optimizer_->optimize(result);
}

std::shared_ptr<Feature> Tracker::selectRefFeature(const std::shared_ptr<MapPoint>& map_point) const
{
    if (map_point == nullptr)
        return nullptr;

    const std::shared_ptr<Feature> ref_feature = map_point->getRefFeature();
    if (ref_feature != nullptr)
        return ref_feature;

    return map_point->selectRefFeatureCandidate();
}

bool Tracker::getFeatureDescriptor(const std::shared_ptr<Feature>& feature, cv::Mat& descriptor) const
{
    descriptor.release();

    if (feature == nullptr)
        return false;

    const std::shared_ptr<Frame>& frame = feature->getFrame();
    if (frame == nullptr || frame->getDescriptors().empty())
        return false;

    const int feature_idx = feature->getFeatureIdx();
    if (feature_idx < 0 || feature_idx >= frame->getDescriptors().rows)
        return false;

    descriptor = frame->getDescriptors().row(feature_idx).clone();
    return true;
}

bool Tracker::getMapPointDescriptor(const std::shared_ptr<MapPoint>& map_point, cv::Mat& descriptor) const
{
    descriptor.release();

    if (map_point == nullptr)
        return false;

    if (map_point->hasRepresentativeDescriptor())
    {
        descriptor = map_point->getRepresentativeDescriptor().clone();
        return true;
    }

    const std::shared_ptr<Feature> ref_feature = selectRefFeature(map_point);
    return getFeatureDescriptor(ref_feature, descriptor);
}

bool Tracker::setInitialPoseGuessFromFrame(const std::shared_ptr<Frame>& frame,
                                           PnPResult& result) const
{
    if (frame == nullptr || frame->getRcw().empty() || frame->getTcw().empty())
        return false;

    cv::Mat rvec;
    cv::Rodrigues(frame->getRcw(), rvec);

    rvec.convertTo(result.rvec, CV_64F);
    frame->getTcw().convertTo(result.tvec, CV_64F);

    return true;
}

bool Tracker::isProjectionMatchReliable(int best_distance,
                                        int second_best_distance,
                                        int best_level,
                                        int second_best_level) const
{
    if (best_distance > matcher_.getMaxHammingDistance())
        return false;

    if (second_best_distance == std::numeric_limits<int>::max())
        return true;

    constexpr float kProjectionRatio = 0.9f;

    if (best_level == second_best_level &&
        static_cast<float>(best_distance) >= kProjectionRatio * static_cast<float>(second_best_distance))
    {
        return false;
    }

    return true;
}

float Tracker::computeSearchRadius(int predicted_level) const
{
    if (levels_num_ <= 1)
        return base_projection_search_radius_;

    const int clamped_level = std::max(0, std::min(predicted_level, levels_num_ - 1));

    return base_projection_search_radius_ * std::pow(scale_factor_, clamped_level);
}

bool Tracker::projectMapPointToFrame(const std::shared_ptr<MapPoint>& map_point, 
                                     const std::shared_ptr<Frame>& frame, 
                                     cv::Point2f& projected_pixel,
                                     double& depth,
                                     double& camera_distance) const
{
    projected_pixel = cv::Point2f(0.0f, 0.0f);
    depth = 0.0;
    camera_distance = 0.0;

    if (camera_ == nullptr || map_point == nullptr || frame == nullptr)
        return false;

    const cv::Point3d& pw = map_point->getPos();
    const cv::Mat point_w = (cv::Mat_<double>(3, 1) << pw.x, pw.y, pw.z);
    const cv::Mat point_c = frame->getRcw() * point_w + frame->getTcw();

    depth = point_c.at<double>(2, 0);
    if (depth <= 0.0)
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

    Eigen::Vector3d pc(point_c.at<double>(0, 0), 
                       point_c.at<double>(1, 0), 
                       point_c.at<double>(2, 0));

    const Eigen::Vector2d pixel = camera_->Camera2Pixel(pc);
    projected_pixel.x = static_cast<float>(pixel(0));
    projected_pixel.y = static_cast<float>(pixel(1));

    const int img_cols = frame->getImg().cols;
    const int img_rows = frame->getImg().rows;
    const int border = 10;

    if (projected_pixel.x < border || projected_pixel.x >= img_cols - border ||
        projected_pixel.y < border || projected_pixel.y >= img_rows - border)
    {
        return false;
    }

    return true;
}

int Tracker::findBestFeatureInArea(const std::shared_ptr<Frame>& frame,
                                   const cv::Point2f& projected_pixel,
                                   int predicted_level, 
                                   const cv::Mat& map_descriptor,
                                   const std::unordered_set<int>& used_feature_indices,
                                   float search_radius) const
{
    if (frame == nullptr || map_descriptor.empty())
        return -1;

    const std::vector<std::shared_ptr<Feature>>& features = frame->getFeatures();
    const cv::Mat& descriptors = frame->getDescriptors();

    const int min_level = std::max(0, predicted_level - 1);
    const int max_level = std::min(levels_num_ - 1, predicted_level + 1);

    const std::vector<int> candidate_indices = frame->getFeatureIndicesInArea(projected_pixel, 
                                                                              search_radius, 
                                                                              min_level, 
                                                                              max_level);

    if (candidate_indices.empty())
        return -1;

    int best_idx = -1;
    int best_distance = std::numeric_limits<int>::max();
    int second_best_distance = std::numeric_limits<int>::max();

    int best_level = -1;
    int second_best_level = -1;

    for (const int i : candidate_indices)
    {
        if (used_feature_indices.count(i) > 0)
            continue;

        const std::shared_ptr<Feature>& feature = features[i];
        if (feature == nullptr)
            continue;

        const int feature_level = feature->getLevel();

        const double pixel_distance = cv::norm(feature->getKeyPoint().pt - projected_pixel);
        if (pixel_distance > search_radius)
            continue;

        const cv::Mat cur_descriptor = descriptors.row(i);
        const int descriptor_distance = 
            static_cast<int>(cv::norm(cur_descriptor, map_descriptor, cv::NORM_HAMMING));

        if (descriptor_distance > matcher_.getMaxHammingDistance())
            continue;

        if (descriptor_distance < best_distance)
        {
            second_best_distance = best_distance;
            second_best_level = best_level;

            best_distance = descriptor_distance;
            best_level = feature_level;
            best_idx = i;
        }
        else if (descriptor_distance < second_best_distance)
        {
            second_best_distance = descriptor_distance;
            second_best_level = feature_level;
        }
    }

    if (best_idx < 0)
        return -1;

    if (!isProjectionMatchReliable(best_distance, 
                                   second_best_distance, 
                                   best_level, 
                                   second_best_level))
    {
        return -1;
    }

    return best_idx;
}

PnPResult Tracker::trackFrameByMap(const std::shared_ptr<Map>& map,
                                   const std::shared_ptr<Frame>& cur_frame) const
{
    if (map == nullptr)
        return {};

    return trackFrameByDescriptorMapPoints(map->getMapPoints(), cur_frame);
}

PnPResult Tracker::trackFrameByBoWKeyframe(const std::shared_ptr<Frame>& keyframe, 
                                           const std::shared_ptr<Frame>& cur_frame) const
{
    PnPResult result;

    if (keyframe == nullptr || cur_frame == nullptr || 
        !keyframe->hasFeatures() || !cur_frame->hasFeatures() ||
        !keyframe->hasBoW() || !cur_frame->hasBoW())
    {
        return result;
    }

    const std::vector<std::pair<int, int>> match_indices = 
        matcher_.matchFramesByBoW(*keyframe, *cur_frame);

    if (match_indices.size() < 4)
        return result;

    const std::vector<std::shared_ptr<Feature>>& key_features = keyframe->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_frame->getFeatures();

    std::unordered_set<std::size_t> used_map_point_ids;
    used_map_point_ids.reserve(match_indices.size());

    result.object_points.reserve(match_indices.size());
    result.img_points.reserve(match_indices.size());
    result.candidate_map_points.reserve(match_indices.size());
    result.candidate_features.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int key_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (key_idx < 0 || key_idx >= static_cast<int>(key_features.size()) ||
            cur_idx < 0 || cur_idx >= static_cast<int>(cur_features.size()))
        {
            continue;
        }

        const std::shared_ptr<Feature>& key_feature = key_features[key_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (key_feature == nullptr || cur_feature == nullptr || !key_feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint> map_point = key_feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (!used_map_point_ids.insert(map_point->getId()).second)
            continue;

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(cur_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(cur_feature);
    }

    if (result.object_points.size() < 4 || pose_optimizer_ == nullptr)
        return {};

    return pose_optimizer_->optimize(result);
}

PnPResult Tracker::trackFrameByMapPoints(const std::vector<std::shared_ptr<MapPoint>>& map_points, 
                                         const std::shared_ptr<Frame> &cur_frame) const
{
    const std::size_t kMinMotionModelMatches = 20;

    std::vector<std::shared_ptr<MapPoint>> visible_map_points;
    PnPResult best_result = 
        trackFrameByProjection(map_points, cur_frame, 1.0f, false, &visible_map_points);

    if (best_result.object_points.size() < kMinMotionModelMatches)
    {
        std::vector<std::shared_ptr<MapPoint>> expanded_visible_map_points;
        PnPResult expanded_result = trackFrameByProjection(map_points, 
                                                           cur_frame, 
                                                           2.0f, 
                                                           false, 
                                                           &expanded_visible_map_points);

        if (expanded_result.success ||
            expanded_result.object_points.size() > best_result.object_points.size())
        {
            best_result = std::move(expanded_result);
            visible_map_points = std::move(expanded_visible_map_points);
        }
    }

    if (!best_result.success || best_result.inlier_num < kMinMotionModelMatches)
    {
        PnPResult descriptor_result = trackFrameByDescriptorMapPoints(map_points, cur_frame);

        const bool descriptor_result_better = 
            descriptor_result.success && 
            (!best_result.success || 
             descriptor_result.inlier_num > best_result.inlier_num ||
             (descriptor_result.inlier_num == best_result.inlier_num && 
              descriptor_result.optimized_reproj_error < best_result.optimized_reproj_error));

        if (descriptor_result_better)
        {
            best_result = std::move(descriptor_result);
        }
    }

    updateProjectionStatistics(visible_map_points, best_result);

    return best_result;
}

PnPResult Tracker::trackFrameByMotionModel(const std::shared_ptr<Frame>& last_frame, 
                                           const std::shared_ptr<Frame> &cur_frame) const
{
    constexpr std::size_t kMinProjectionMatches = 20;

    if (last_frame == nullptr || cur_frame == nullptr)
        return {};

    std::vector<std::shared_ptr<MapPoint>> last_frame_map_points;
    std::unordered_set<std::size_t> map_point_ids;
    last_frame_map_points.reserve(last_frame->getFeatureNum());
    map_point_ids.reserve(last_frame->getFeatureNum() * 2 + 1);

    for (const auto& feature : last_frame->getFeatures())
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (map_point_ids.insert(map_point->getId()).second)
            last_frame_map_points.push_back(map_point);
    }

    if (last_frame_map_points.size() < kMinProjectionMatches)
        return {};

    PnPResult projection_result = 
        trackFrameByProjection(last_frame_map_points, cur_frame, 1.0f, false);

    if (projection_result.object_points.size() >= kMinProjectionMatches)
        return projection_result;

    PnPResult expanded_projection_result = 
        trackFrameByProjection(last_frame_map_points, cur_frame, 2.0f, false);

    if (expanded_projection_result.object_points.size() < kMinProjectionMatches)
        return {};

    return expanded_projection_result;
}

void Tracker::updateProjectionStatistics(
    const std::vector<std::shared_ptr<MapPoint>>& visible_map_points,
    const PnPResult& projection_result) const
{
    for (const auto& map_point : visible_map_points)
    {
        if (map_point != nullptr && !map_point->isBad())
            map_point->increaseVisibleTimes();
    }

    for (const auto& map_point : projection_result.candidate_map_points)
    {
        if (map_point != nullptr && !map_point->isBad())
            map_point->increaseFoundTimes();
    }
}

PnPResult Tracker::trackFrameByDescriptorMapPoints(
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::shared_ptr<Frame>& cur_frame) const
{
    PnPResult result;

    if (cur_frame == nullptr || !cur_frame->hasFeatures() || map_points.empty())
        return result;

    cv::Mat map_descriptors;
    std::vector<std::shared_ptr<MapPoint>> descriptor_map_points;
    descriptor_map_points.reserve(map_points.size());

    for (const auto& map_point : map_points)
    {
        if (map_point == nullptr)
            continue;

        const std::shared_ptr<Feature> ref_feature = selectRefFeature(map_point);

        cv::Mat descriptor;
        if (!getMapPointDescriptor(map_point, descriptor))
            continue;

        map_descriptors.push_back(descriptor);
        descriptor_map_points.push_back(map_point);
    }

    if (map_descriptors.rows < 4)
        return result;

    const std::vector<std::pair<int, int>> match_indices =
        matcher_.matchDescriptors(map_descriptors, cur_frame->getDescriptors());

    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_frame->getFeatures();

    result.object_points.reserve(match_indices.size());
    result.img_points.reserve(match_indices.size());
    result.candidate_map_points.reserve(match_indices.size());
    result.candidate_features.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int map_desc_idx = match_idx.first;
        const int cur_feature_idx = match_idx.second;

        if (map_desc_idx < 0 || map_desc_idx >= static_cast<int>(descriptor_map_points.size()) ||
            cur_feature_idx < 0 || cur_feature_idx >= static_cast<int>(cur_features.size()))
        {
            continue;
        }

        const std::shared_ptr<MapPoint>& map_point = descriptor_map_points[map_desc_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_feature_idx];

        if (map_point == nullptr || cur_feature == nullptr)
            continue;

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(cur_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(cur_feature);
    }

    if (pose_optimizer_ == nullptr)
        return {};

    setInitialPoseGuessFromFrame(cur_frame, result);

    return pose_optimizer_->optimize(result);
}

void Tracker::appendInlierCorrespondences(const PnPResult& source_result, 
                                          PnPResult& destination_result, 
                                          std::unordered_set<std::size_t> &used_map_point_ids, 
                                          std::unordered_set<int> &used_feature_indices) const
{
    for (int i = 0; i < source_result.inlier_indices.rows; i++)
    {
        const int source_idx = source_result.inlier_indices.at<int>(i, 0);

        if (source_idx < 0 || 
            source_idx >= static_cast<int>(source_result.object_points.size()) ||
            source_idx >= static_cast<int>(source_result.img_points.size()) ||
            source_idx >= static_cast<int>(source_result.candidate_map_points.size()) ||
            source_idx >= static_cast<int>(source_result.candidate_features.size()))
        {
            continue;
        }

        const std::shared_ptr<MapPoint>& map_point = 
            source_result.candidate_map_points[source_idx];
        const std::shared_ptr<Feature>& feature = 
            source_result.candidate_features[source_idx];

        if (map_point == nullptr || map_point->isBad() || feature == nullptr)
            continue;

        const int feature_idx = feature->getFeatureIdx();
        if (feature_idx < 0 ||
            used_map_point_ids.count(map_point->getId()) > 0 ||
            used_feature_indices.count(feature_idx) > 0)
        {
            continue;
        }

        used_map_point_ids.insert(map_point->getId());
        used_feature_indices.insert(feature_idx);

        destination_result.object_points.push_back(map_point->getPos());
        destination_result.img_points.push_back(feature->getKeyPoint().pt);
        destination_result.candidate_map_points.push_back(map_point);
        destination_result.candidate_features.push_back(feature);
    }
}

void Tracker::appendProjectionCorrespondences(
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::shared_ptr<Frame>& cur_frame,
    float radius_scale,
    bool update_statistics,
    std::unordered_set<std::size_t>& used_map_point_ids,
    std::unordered_set<int>& used_feature_indices,
    PnPResult& result,
    std::vector<std::shared_ptr<MapPoint>>* visible_map_points) const
{
    if (cur_frame == nullptr || !cur_frame->hasFeatures())
        return;

    if (visible_map_points != nullptr)
    {
        visible_map_points->clear();
        visible_map_points->reserve(map_points.size());
    }

    for (const auto& map_point : map_points)
    {
        if (map_point == nullptr || map_point->isBad() ||
            used_map_point_ids.count(map_point->getId()) > 0)
        {
            continue;
        }

        const std::shared_ptr<Feature> ref_feature = selectRefFeature(map_point);
        if (ref_feature == nullptr)
            continue;

        cv::Mat descriptor;
        if (!getMapPointDescriptor(map_point, descriptor))
            continue;

        cv::Point2f projected_pixel;
        double depth = 0.0;
        double camera_distance = 0.0;
        if (!projectMapPointToFrame(map_point, cur_frame, projected_pixel, depth, camera_distance))
            continue;

        if (visible_map_points != nullptr)
            visible_map_points->push_back(map_point);

        if (update_statistics)
            map_point->increaseVisibleTimes();

        int predicted_level = ref_feature->getLevel();
        if (map_point->hasValidViewStatistics())
        {
            predicted_level = 
                map_point->predictScaleLevel(camera_distance, scale_factor_, levels_num_);
        }

        const float search_radius =
            computeSearchRadius(predicted_level) * std::max(radius_scale, 0.5f);

        const int cur_feature_idx = findBestFeatureInArea(cur_frame, 
                                                          projected_pixel, 
                                                          predicted_level, 
                                                          descriptor, 
                                                          used_feature_indices,
                                                          search_radius);

        if (cur_feature_idx < 0)
            continue;

        const std::shared_ptr<Feature>& cur_feature = cur_frame->getFeatures()[cur_feature_idx];
        if (cur_feature == nullptr)
            continue;

        used_map_point_ids.insert(map_point->getId());
        used_feature_indices.insert(cur_feature_idx);

        if (update_statistics)
            map_point->increaseFoundTimes();

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(cur_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(cur_feature);
    }
}

PnPResult Tracker::refinePoseWithLocalMap(
    const PnPResult& motion_pnp_result, 
    const std::vector<std::shared_ptr<MapPoint>>& local_map_points, 
    const std::shared_ptr<Frame> &cur_frame) const
{
    if (!motion_pnp_result.success || cur_frame == nullptr ||
        !cur_frame->hasFeatures() || pose_optimizer_ == nullptr)
    {
        return {};
    }

    PnPResult combined_result;
    const std::size_t reserve_num = 
        motion_pnp_result.inlier_indices.rows + local_map_points.size();

    combined_result.object_points.reserve(reserve_num);
    combined_result.img_points.reserve(reserve_num);
    combined_result.candidate_map_points.reserve(reserve_num);
    combined_result.candidate_features.reserve(reserve_num);

    std::unordered_set<std::size_t> used_map_point_ids;
    used_map_point_ids.reserve(reserve_num);

    std::unordered_set<int> used_feature_indices;
    used_feature_indices.reserve(reserve_num);

    appendInlierCorrespondences(motion_pnp_result, 
                                combined_result, 
                                used_map_point_ids, 
                                used_feature_indices);     

    appendProjectionCorrespondences(local_map_points, 
                                    cur_frame, 
                                    1.0f, 
                                    false, 
                                    used_map_point_ids, 
                                    used_feature_indices, 
                                    combined_result);

    if (combined_result.object_points.size() < 6 ||
        !setInitialPoseGuessFromFrame(cur_frame, combined_result))
    {
        return {};
    }

    return pose_optimizer_->optimize(combined_result);
}

PnPResult Tracker::trackFrameByProjectionOnly(const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                              const std::shared_ptr<Frame>& cur_frame,
                                              float radius_scale,
                                              bool update_statistics) const
{
    const float valid_radius_scale = radius_scale > 0.0f ? radius_scale : 1.0f;
    return trackFrameByProjection(map_points, cur_frame, valid_radius_scale, update_statistics);
}

PnPResult Tracker::trackFrameByProjection(
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::shared_ptr<Frame>& cur_frame,
    float radius_scale,
    bool update_statistics,
    std::vector<std::shared_ptr<MapPoint>>* visible_map_points) const
{
    PnPResult result;

    if (cur_frame == nullptr || !cur_frame->hasFeatures() || map_points.empty())
        return result;

    result.object_points.reserve(map_points.size());
    result.img_points.reserve(map_points.size());
    result.candidate_map_points.reserve(map_points.size());
    result.candidate_features.reserve(map_points.size());

    std::unordered_set<std::size_t> used_map_point_ids;
    used_map_point_ids.reserve(map_points.size());

    std::unordered_set<int> used_feature_indices;
    used_feature_indices.reserve(cur_frame->getFeatureNum());

    appendProjectionCorrespondences(map_points, 
                                    cur_frame, 
                                    radius_scale, 
                                    update_statistics, 
                                    used_map_point_ids, 
                                    used_feature_indices, 
                                    result, 
                                    visible_map_points);

    if (result.object_points.size() < 6 || pose_optimizer_ == nullptr)
        return {};

    if (!setInitialPoseGuessFromFrame(cur_frame, result))
        return {};

    return pose_optimizer_->optimize(result);
}

TrackingResult Tracker::buildTrackingResult(const InitializationResult& init_result, const PnPResult& pnp_result) const
{
    if (!init_result.success || !pnp_result.success)
        return {};

    return buildTrackingResult(init_result.cur_frame, pnp_result);
}

TrackingResult Tracker::buildTrackingResult(const std::shared_ptr<Frame>& frame, const PnPResult& pnp_result) const
{
    TrackingResult result;

    if (frame == nullptr || !pnp_result.success)
        return result;

    result.frame = frame;
    result.R_cw = pnp_result.R.clone();
    result.t_cw = pnp_result.tvec.clone();

    for (int i = 0; i < pnp_result.inlier_indices.rows; i++)
    {
        const int idx = pnp_result.inlier_indices.at<int>(i, 0);

        if (idx < 0 || 
            idx >= static_cast<int>(pnp_result.candidate_map_points.size()) ||
            idx >= static_cast<int>(pnp_result.candidate_features.size()))
            continue;

        const std::shared_ptr<MapPoint>& map_point = pnp_result.candidate_map_points[idx];
        const std::shared_ptr<Feature>& feature = pnp_result.candidate_features[idx];

        if (map_point == nullptr || feature == nullptr)
            continue;

        result.inlier_map_points.push_back(map_point);
        result.inlier_features.push_back(feature);
    }

    result.success = !result.inlier_map_points.empty();
    return result;
}

} // namespace mini_orb_slam
