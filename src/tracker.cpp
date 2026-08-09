#include <algorithm>
#include <cmath>
#include <cstdint>
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

// ORB-SLAM2 logic reference: GetMin/MaxDistanceInvariance adds a 0.8/1.2
// margin around the reference pyramid-derived range before projection gating.
double minDistanceInvariance(double min_distance)
{
    return 0.8 * min_distance;
}

double maxDistanceInvariance(double max_distance)
{
    return 1.2 * max_distance;
}

// ORB-SLAM2 logic reference: descriptors are 256-bit ORB rows. Keep the
// same Hamming metric while avoiding a per-candidate OpenCV norm dispatch.
int orbDescriptorDistance(const cv::Mat& lhs, const cv::Mat& rhs)
{
    if (lhs.rows != 1 || rhs.rows != 1 || lhs.cols != 32 || rhs.cols != 32 ||
        lhs.type() != CV_8U || rhs.type() != CV_8U ||
        !lhs.isContinuous() || !rhs.isContinuous())
    {
        return static_cast<int>(cv::norm(lhs, rhs, cv::NORM_HAMMING));
    }

    const auto* lhs_words = lhs.ptr<std::uint32_t>();
    const auto* rhs_words = rhs.ptr<std::uint32_t>();
    int distance = 0;
    for (int word = 0; word < 8; ++word)
        distance += __builtin_popcount(lhs_words[word] ^ rhs_words[word]);
    return distance;
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

        if (map_point->isBad())
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

    // ORB-SLAM2 logic reference: keep a descriptor row view during matching;
    // the owning Frame is retained by the Feature and remains alive here.
    descriptor = frame->getDescriptors().row(feature_idx);
    return true;
}

bool Tracker::getMapPointDescriptor(const std::shared_ptr<MapPoint>& map_point, cv::Mat& descriptor) const
{
    descriptor.release();

    if (map_point == nullptr)
        return false;

    descriptor = map_point->getRepresentativeDescriptor();
    if (!descriptor.empty())
        return true;

    const std::shared_ptr<Feature> ref_feature = selectRefFeature(map_point);
    return getFeatureDescriptor(ref_feature, descriptor);
}

bool Tracker::setInitialPoseGuessFromFrame(const std::shared_ptr<Frame>& frame,
                                           PnPResult& result) const
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

    rvec.convertTo(result.rvec, CV_64F);
    t_cw.convertTo(result.tvec, CV_64F);

    return true;
}

bool Tracker::isProjectionMatchReliable(int best_distance,
                                        int second_best_distance,
                                        int best_level,
                                        int second_best_level,
                                        bool apply_ratio_test) const
{
    if (best_distance > matcher_.getMaxHammingDistance())
        return false;

    // ORB-SLAM2 logic reference: local-map projection matching applies the
    // configured NN ratio only when the best and second-best features are in
    // the same pyramid level. Motion-model and relocalization projection
    // paths intentionally pass apply_ratio_test=false.
    if (apply_ratio_test &&
        second_best_distance != std::numeric_limits<int>::max() &&
        best_level == second_best_level &&
        static_cast<float>(best_distance) >=
            matcher_.getRatioThreshold() * static_cast<float>(second_best_distance))
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

Tracker::ProjectionPose Tracker::snapshotProjectionPose(
    const std::shared_ptr<Frame>& frame) const
{
    ProjectionPose pose;
    if (frame == nullptr)
        return pose;

    frame->copyPose(pose.R_cw, pose.t_cw);
    if (pose.R_cw.rows != 3 || pose.R_cw.cols != 3 ||
        pose.t_cw.rows != 3 || pose.t_cw.cols != 1)
    {
        pose.R_cw.release();
        pose.t_cw.release();
        return pose;
    }

    const double tx = pose.t_cw.at<double>(0, 0);
    const double ty = pose.t_cw.at<double>(1, 0);
    const double tz = pose.t_cw.at<double>(2, 0);
    const double r00 = pose.R_cw.at<double>(0, 0);
    const double r01 = pose.R_cw.at<double>(0, 1);
    const double r02 = pose.R_cw.at<double>(0, 2);
    const double r10 = pose.R_cw.at<double>(1, 0);
    const double r11 = pose.R_cw.at<double>(1, 1);
    const double r12 = pose.R_cw.at<double>(1, 2);
    const double r20 = pose.R_cw.at<double>(2, 0);
    const double r21 = pose.R_cw.at<double>(2, 1);
    const double r22 = pose.R_cw.at<double>(2, 2);
    pose.camera_center = cv::Point3d(
        -(r00 * tx + r10 * ty + r20 * tz),
        -(r01 * tx + r11 * ty + r21 * tz),
        -(r02 * tx + r12 * ty + r22 * tz));
    pose.image_width = frame->getImg().cols;
    pose.image_height = frame->getImg().rows;
    pose.valid = pose.image_width > 0 && pose.image_height > 0;
    return pose;
}

bool Tracker::projectMapPointToFrame(const std::shared_ptr<MapPoint>& map_point,
                                     const ProjectionPose& pose,
                                     cv::Point2f& projected_pixel,
                                     double& depth,
                                     double& camera_distance) const
{
    projected_pixel = cv::Point2f(0.0f, 0.0f);
    depth = 0.0;
    camera_distance = 0.0;

    if (camera_ == nullptr || map_point == nullptr || !pose.valid)
        return false;

    const cv::Point3d& pw = map_point->getPos();
    const double x = pose.R_cw.at<double>(0, 0) * pw.x +
                     pose.R_cw.at<double>(0, 1) * pw.y +
                     pose.R_cw.at<double>(0, 2) * pw.z +
                     pose.t_cw.at<double>(0, 0);
    const double y = pose.R_cw.at<double>(1, 0) * pw.x +
                     pose.R_cw.at<double>(1, 1) * pw.y +
                     pose.R_cw.at<double>(1, 2) * pw.z +
                     pose.t_cw.at<double>(1, 0);
    const double z = pose.R_cw.at<double>(2, 0) * pw.x +
                     pose.R_cw.at<double>(2, 1) * pw.y +
                     pose.R_cw.at<double>(2, 2) * pw.z +
                     pose.t_cw.at<double>(2, 0);

    depth = z;
    if (depth <= 0.0)
        return false;

    const cv::Point3d view = pw - pose.camera_center;
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

    Eigen::Vector3d pc(x, y, z);

    const Eigen::Vector2d pixel = camera_->Camera2Pixel(pc);
    projected_pixel.x = static_cast<float>(pixel(0));
    projected_pixel.y = static_cast<float>(pixel(1));

    const int border = 10;

    if (projected_pixel.x < border || projected_pixel.x >= pose.image_width - border ||
        projected_pixel.y < border || projected_pixel.y >= pose.image_height - border)
    {
        return false;
    }

    return true;
}

Tracker::ProjectionFeatureSearchResult Tracker::findBestFeatureInArea(
    const std::shared_ptr<Frame>& frame,
    const cv::Point2f& projected_pixel,
    int predicted_level,
    const cv::Mat& map_descriptor,
    const std::unordered_set<int>& used_feature_indices,
    float search_radius,
    bool apply_ratio_test) const
{
    ProjectionFeatureSearchResult result;

    if (frame == nullptr || map_descriptor.empty())
        return result;

    const std::vector<std::shared_ptr<Feature>>& features = frame->getFeatures();
    const cv::Mat& descriptors = frame->getDescriptors();

    const int min_level = std::max(0, predicted_level - 1);
    // ORB-SLAM2 logic reference: SearchByProjection considers the predicted
    // octave and the immediately finer octave, never a coarser one. A
    // coarser candidate is more likely to be a spatially nearby alias after
    // motion prediction and can perturb the local-map pose refinement.
    const int max_level = std::min(levels_num_ - 1, predicted_level);

    // The radius gate is equivalent in squared form and avoids constructing a
    // temporary point plus a square root for every projected map-point.
    const float search_radius_squared = search_radius * search_radius;

    const std::vector<int> candidate_indices = frame->getFeatureIndicesInArea(projected_pixel, 
                                                                              search_radius, 
                                                                              min_level, 
                                                                              max_level);

    result.has_spatial_candidates = !candidate_indices.empty();
    if (!result.has_spatial_candidates)
        return result;

    int best_idx = -1;
    int best_distance = std::numeric_limits<int>::max();
    int second_best_distance = std::numeric_limits<int>::max();

    int best_level = -1;
    int second_best_level = -1;

    for (const int i : candidate_indices)
    {
        if (used_feature_indices.count(i) > 0)
            continue;

        if (i < 0 || i >= static_cast<int>(features.size()) ||
            i >= descriptors.rows)
        {
            continue;
        }

        const std::shared_ptr<Feature>& feature = features[i];
        if (feature == nullptr)
            continue;

        const int feature_level = feature->getLevel();

        const cv::Point2f pixel_delta = feature->getKeyPoint().pt - projected_pixel;
        const float pixel_distance_squared = pixel_delta.x * pixel_delta.x +
                                              pixel_delta.y * pixel_delta.y;
        if (pixel_distance_squared > search_radius_squared)
            continue;

        const cv::Mat cur_descriptor = descriptors.row(i);
        const int descriptor_distance = orbDescriptorDistance(cur_descriptor, map_descriptor);

        result.has_available_descriptor = true;

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
        return result;

    if (!isProjectionMatchReliable(best_distance, 
                                   second_best_distance, 
                                   best_level, 
                                   second_best_level,
                                   apply_ratio_test))
    {
        result.hamming_rejected = best_distance > matcher_.getMaxHammingDistance();
        result.ratio_rejected = !result.hamming_rejected && apply_ratio_test &&
            second_best_distance != std::numeric_limits<int>::max() &&
            best_level == second_best_level &&
            static_cast<float>(best_distance) >= matcher_.getRatioThreshold() *
                static_cast<float>(second_best_distance);
        return result;
    }

    result.feature_idx = best_idx;
    return result;
}

PnPResult Tracker::trackFrameByMap(const std::shared_ptr<Map>& map,
                                   const std::shared_ptr<Frame>& cur_frame) const
{
    if (map == nullptr)
        return {};

    const std::vector<std::shared_ptr<MapPoint>> map_points = map->copyMapPoints();
    return trackFrameByDescriptorMapPoints(map_points, cur_frame);
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
        trackFrameByProjection(map_points, cur_frame, 1.0f, false, true, true,
                                &visible_map_points);

    constexpr std::size_t kExpandProjectionMatches = 40;
    if (best_result.object_points.size() < kExpandProjectionMatches)
    {
        std::vector<std::shared_ptr<MapPoint>> expanded_visible_map_points;
        PnPResult expanded_result = trackFrameByProjection(map_points, 
                                                           cur_frame, 
                                                           2.0f, 
                                                           false, 
                                                           true,
                                                           true,
                                                           &expanded_visible_map_points);

        if (expanded_result.success ||
            expanded_result.object_points.size() > best_result.object_points.size())
        {
            best_result = std::move(expanded_result);
            visible_map_points = std::move(expanded_visible_map_points);
        }
    }

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R12 frame="
                    << (cur_frame != nullptr ? cur_frame->getId() : 0)
                    << " stage=motion_projection input=" << map_points.size()
                    << " visible=" << visible_map_points.size()
                    << " candidates=" << best_result.object_points.size()
                    << " success=" << (best_result.success ? 1 : 0)
                    << " inliers=" << best_result.inlier_num
                    << " reproj=" << best_result.optimized_reproj_error);

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

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R12 frame="
                    << (cur_frame != nullptr ? cur_frame->getId() : 0)
                    << " stage=motion_final success=" << (best_result.success ? 1 : 0)
                    << " inliers=" << best_result.inlier_num
                    << " candidates=" << best_result.object_points.size()
                    << " reproj=" << best_result.optimized_reproj_error);
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
        trackFrameByProjection(last_frame_map_points, cur_frame, 1.0f, false, false, true);

    if (projection_result.object_points.size() >= kMinProjectionMatches)
        return projection_result;

    PnPResult expanded_projection_result = 
        trackFrameByProjection(last_frame_map_points, cur_frame, 2.0f, false, false, true);

    if (expanded_projection_result.object_points.size() < kMinProjectionMatches)
        return {};

    return expanded_projection_result;
}

PnPResult Tracker::trackFrameByReferenceFrame(
    const std::shared_ptr<Frame>& reference_frame,
    const std::shared_ptr<Frame>& cur_frame) const
{
    PnPResult result;

    if (reference_frame == nullptr || cur_frame == nullptr ||
        !reference_frame->hasFeatures() || !cur_frame->hasFeatures() ||
        pose_optimizer_ == nullptr)
    {
        return result;
    }

    const std::vector<std::pair<int, int>> matches =
        matcher_.matchFrames(*reference_frame, *cur_frame);
    const std::vector<std::shared_ptr<Feature>>& reference_features =
        reference_frame->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& current_features =
        cur_frame->getFeatures();

    std::unordered_set<std::size_t> used_map_point_ids;
    used_map_point_ids.reserve(matches.size() * 2 + 1);

    result.object_points.reserve(matches.size());
    result.img_points.reserve(matches.size());
    result.candidate_map_points.reserve(matches.size());
    result.candidate_features.reserve(matches.size());

    for (const auto& match : matches)
    {
        const int reference_idx = match.first;
        const int current_idx = match.second;
        if (reference_idx < 0 || reference_idx >= static_cast<int>(reference_features.size()) ||
            current_idx < 0 || current_idx >= static_cast<int>(current_features.size()))
        {
            continue;
        }

        const std::shared_ptr<Feature>& reference_feature = reference_features[reference_idx];
        const std::shared_ptr<Feature>& current_feature = current_features[current_idx];
        if (reference_feature == nullptr || current_feature == nullptr ||
            !reference_feature->hasMapPoint())
        {
            continue;
        }

        const std::shared_ptr<MapPoint> map_point = reference_feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad() ||
            !used_map_point_ids.insert(map_point->getId()).second)
        {
            continue;
        }

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(current_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(current_feature);
    }

    if (result.object_points.size() < 6 ||
        !setInitialPoseGuessFromFrame(cur_frame, result))
    {
        return {};
    }

    return pose_optimizer_->optimize(result);
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

        if (map_point->isBad())
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
    bool apply_ratio_test,
    bool apply_view_gate,
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

    std::size_t skipped_map_points = 0;
    std::size_t descriptor_missing = 0;
    std::size_t projection_rejected = 0;
    std::size_t distance_low_rejected = 0;
    std::size_t distance_high_rejected = 0;
    std::size_t view_angle_rejected = 0;
    std::size_t border_rejected = 0;
    std::size_t feature_missing = 0;
    std::size_t no_spatial_candidate = 0;
    std::size_t no_available_descriptor = 0;
    std::size_t hamming_rejected = 0;
    std::size_t ratio_rejected = 0;
    std::size_t matched_num = 0;

    const ProjectionPose projection_pose = snapshotProjectionPose(cur_frame);
    if (!projection_pose.valid)
        return;

    for (const auto& map_point : map_points)
    {
        if (map_point == nullptr || map_point->isBad() ||
            used_map_point_ids.count(map_point->getId()) > 0)
        {
            skipped_map_points++;
            continue;
        }

        const std::shared_ptr<Feature> ref_feature = selectRefFeature(map_point);
        if (ref_feature == nullptr)
        {
            descriptor_missing++;
            continue;
        }

        cv::Mat descriptor;
        if (!getMapPointDescriptor(map_point, descriptor))
        {
            descriptor_missing++;
            continue;
        }

        cv::Point2f projected_pixel;
        double depth = 0.0;
        double camera_distance = 0.0;

        cv::Point3d normal;
        double min_distance = 0.0;
        double max_distance = 0.0;
        if (apply_view_gate && map_point->getViewStatistics(normal, min_distance, max_distance))
        {
            const cv::Point3d point_world = map_point->getPos();
            const cv::Point3d camera_center = projection_pose.camera_center;
            const cv::Point3d view = point_world - camera_center;
            const double predicted_distance = pointNorm(view);

            if (predicted_distance < minDistanceInvariance(min_distance))
            {
                distance_low_rejected++;
                continue;
            }
            else if (predicted_distance > maxDistanceInvariance(max_distance))
            {
                distance_high_rejected++;
                continue;
            }
            else
            {
                const double normal_norm = pointNorm(normal);
                const double view_cos = normal_norm > 1e-6
                    ? dotPoint(view, normal) /
                      (predicted_distance * normal_norm)
                    : -1.0;
                if (view_cos < 0.5)
                {
                    view_angle_rejected++;
                    continue;
                }
            }
        }

        if (!projectMapPointToFrame(map_point, projection_pose,
                                    projected_pixel, depth, camera_distance))
        {
            projection_rejected++;

            if (projected_pixel.x < 10.0f ||
                projected_pixel.x >= cur_frame->getImg().cols - 10.0f ||
                projected_pixel.y < 10.0f ||
                projected_pixel.y >= cur_frame->getImg().rows - 10.0f)
            {
                border_rejected++;
            }
            continue;
        }

        if (visible_map_points != nullptr)
            visible_map_points->push_back(map_point);

        if (update_statistics)
            map_point->increaseVisibleTimes();

        int predicted_level = ref_feature->getLevel();
        if (apply_view_gate && map_point->getViewStatistics(normal, min_distance, max_distance))
        {
            predicted_level = 
                map_point->predictScaleLevel(camera_distance, scale_factor_, levels_num_);
        }

        const float search_radius =
            computeSearchRadius(predicted_level) * std::max(radius_scale, 0.5f);

        const ProjectionFeatureSearchResult search_result = findBestFeatureInArea(
            cur_frame, projected_pixel, predicted_level, descriptor,
            used_feature_indices, search_radius, apply_ratio_test);
        const int cur_feature_idx = search_result.feature_idx;

        if (cur_feature_idx < 0)
        {
            feature_missing++;
            if (!search_result.has_spatial_candidates)
                no_spatial_candidate++;
            else if (!search_result.has_available_descriptor)
                no_available_descriptor++;
            else if (search_result.hamming_rejected)
                hamming_rejected++;
            else if (search_result.ratio_rejected)
                ratio_rejected++;
            continue;
        }

        const std::shared_ptr<Feature>& cur_feature = cur_frame->getFeatures()[cur_feature_idx];
        if (cur_feature == nullptr)
        {
            feature_missing++;
            continue;
        }

        used_map_point_ids.insert(map_point->getId());
        used_feature_indices.insert(cur_feature_idx);

        if (update_statistics)
            map_point->increaseFoundTimes();

        result.object_points.push_back(map_point->getPos());
        result.img_points.push_back(cur_feature->getKeyPoint().pt);
        result.candidate_map_points.push_back(map_point);
        result.candidate_features.push_back(cur_feature);
        matched_num++;
    }

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R19 frame=" << cur_frame->getId()
                    << " timestamp_ns=" << static_cast<long long>(
                           std::llround(cur_frame->getTimestamp() * 1e9))
                    << " stage=projection radius=" << radius_scale
                    << " ratio_test=" << (apply_ratio_test ? 1 : 0)
                    << " input=" << map_points.size()
                    << " skipped=" << skipped_map_points
                    << " descriptor_missing=" << descriptor_missing
                    << " projection_rejected=" << projection_rejected
                    << " distance_low=" << distance_low_rejected
                    << " distance_high=" << distance_high_rejected
                    << " view_angle=" << view_angle_rejected
                    << " border=" << border_rejected
                    << " feature_missing=" << feature_missing
                    << " no_spatial_candidate=" << no_spatial_candidate
                    << " no_available_descriptor=" << no_available_descriptor
                    << " hamming_rejected=" << hamming_rejected
                    << " ratio_rejected=" << ratio_rejected
                    << " matched=" << matched_num);

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
                                    true,
                                    true,
                                    used_map_point_ids, 
                                    used_feature_indices, 
                                    combined_result);

    const std::size_t first_pass_correspondences = combined_result.object_points.size();

    // A slightly stale motion prediction can leave otherwise valid map points
    // just outside the nominal search window. Enlarge the geometric search
    // only when the first pass is sparse; PnP and reprojection checks still
    // decide which correspondences are accepted.
    constexpr std::size_t kSparseLocalMapCorrespondences = 50;
    if (combined_result.object_points.size() < kSparseLocalMapCorrespondences)
    {
        appendProjectionCorrespondences(local_map_points,
                                        cur_frame,
                                        2.0f,
                                        false,
                                        true,
                                        true,
                                        used_map_point_ids,
                                        used_feature_indices,
                                        combined_result);
    }

    if (combined_result.object_points.size() < 6 ||
        !setInitialPoseGuessFromFrame(cur_frame, combined_result))
    {
        return {};
    }

    return pose_optimizer_->optimizeWithPosePrior(combined_result);
}

PnPResult Tracker::trackFrameByProjectionOnly(const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                              const std::shared_ptr<Frame>& cur_frame,
                                              float radius_scale,
                                              bool update_statistics) const
{
    const float valid_radius_scale = radius_scale > 0.0f ? radius_scale : 1.0f;
    return trackFrameByProjection(map_points, cur_frame, valid_radius_scale,
                                  update_statistics, false, true);
}

PnPResult Tracker::trackFrameByProjection(
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::shared_ptr<Frame>& cur_frame,
    float radius_scale,
    bool update_statistics,
    bool apply_ratio_test,
    bool apply_view_gate,
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
                                    apply_ratio_test,
                                    apply_view_gate,
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
