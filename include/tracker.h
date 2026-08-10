#ifndef MINI_ORB_SLAM_INCLUDE_TRACKER_H_
#define MINI_ORB_SLAM_INCLUDE_TRACKER_H_

#include <memory>
#include <unordered_set> 
#include "camera.h"
#include "common.h"
#include "feature.h"
#include "frame.h"
#include "map.h"
#include "matcher.h"
#include "pose_optimizer.h"

namespace mini_orb_slam
{

class Tracker
{
public:
    Tracker(const std::shared_ptr<Camera>& camera, 
            const Matcher& matcher, 
            const std::shared_ptr<PoseOptimizer>& pose_optimizer,
            double scale_factor,
            int levels_num,
            float base_projection_search_radius = 15.0f);

    PnPResult estimatePoseByPnP(const InitializationResult& init_result) const;

    PnPResult trackFrameByMap(const std::shared_ptr<Map>& map, 
                              const std::shared_ptr<Frame>& cur_frame,
                              bool map_mutex_held = false) const;

    PnPResult trackFrameByProjectionOnly(const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                         const std::shared_ptr<Frame>& cur_frame,
                                         float radius_scale = 1.0f,
                                         bool update_statistics = true) const;

    PnPResult trackFrameByMotionModel(const std::shared_ptr<Frame>& last_frame,
                                      const std::shared_ptr<Frame>& cur_frame) const;

    PnPResult trackFrameByReferenceFrame(const std::shared_ptr<Frame>& reference_frame,
                                         const std::shared_ptr<Frame>& cur_frame) const;

    PnPResult refinePoseWithLocalMap(
        const PnPResult& motion_pnp_result,
        const std::vector<std::shared_ptr<MapPoint>>& local_map_points,
        const std::shared_ptr<Frame>& cur_frame) const;

    PnPResult trackFrameByMapPoints(const std::vector<std::shared_ptr<MapPoint>>& map_points, 
                                    const std::shared_ptr<Frame>& cur_frame) const;

    PnPResult trackFrameByBoWKeyframe(const std::shared_ptr<Frame>& keyframe, 
                                      const std::shared_ptr<Frame>& cur_keyframe) const;

    TrackingResult buildTrackingResult(const InitializationResult& init_result, 
                                       const PnPResult& pnp_result) const;

    TrackingResult buildTrackingResult(const std::shared_ptr<Frame>& frame,
                                       const PnPResult& pnp_result) const;

private:
    // P2-EUROC-DEBUG-R19: records why a projected point did not yield a
    // feature match. It is diagnostics-only and never participates in matching.
    struct ProjectionFeatureSearchResult
    {
        int feature_idx{-1};
        bool has_spatial_candidates{false};
        bool has_available_descriptor{false};
        bool hamming_rejected{false};
        bool ratio_rejected{false};
    };

    struct ProjectionPose
    {
        cv::Mat R_cw;
        cv::Mat t_cw;
        cv::Point3d camera_center{0.0, 0.0, 0.0};
        int image_width{0};
        int image_height{0};
        bool valid{false};
    };

    std::shared_ptr<Feature> selectRefFeature(const std::shared_ptr<MapPoint>& map_point) const;
    bool getFeatureDescriptor(const std::shared_ptr<Feature>& feature, cv::Mat& descriptor) const;
    bool getMapPointDescriptor(const std::shared_ptr<MapPoint>& map_point, cv::Mat& descriptor) const;

    bool setInitialPoseGuessFromFrame(const std::shared_ptr<Frame>& frame, PnPResult& result) const;

    bool isProjectionMatchReliable(int best_distance,
                                   int second_best_distance,
                                   int best_level,
                                   int second_best_level,
                                   bool apply_ratio_test) const;

    float computeSearchRadius(int predicted_level) const;

    ProjectionPose snapshotProjectionPose(const std::shared_ptr<Frame>& frame) const;

    bool projectMapPointToFrame(const std::shared_ptr<MapPoint>& map_point, 
                                const ProjectionPose& pose,
                                cv::Point2f& projected_pixel,
                                double& depth,
                                double& camera_distance) const;

    ProjectionFeatureSearchResult findBestFeatureInArea(
        const std::shared_ptr<Frame>& frame,
        const cv::Point2f& projected_pixel,
        int predicted_level,
        const cv::Mat& map_descriptor,
        const std::unordered_set<int>& used_feature_indices,
        float search_radius,
        bool apply_ratio_test) const;

    PnPResult trackFrameByDescriptorMapPoints(const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                              const std::shared_ptr<Frame>& cur_frame) const;

    void updateProjectionStatistics(
        const std::vector<std::shared_ptr<MapPoint>>& visible_map_points,
        const PnPResult& projection_result) const;

    void appendInlierCorrespondences(const PnPResult& source_result,
                                     PnPResult& destination_result,
                                     std::unordered_set<std::size_t>& used_map_point_ids,
                                     std::unordered_set<int>& used_feature_indices) const;

    void appendProjectionCorrespondences(
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        const std::shared_ptr<Frame>& cur_frame,
        float radius_scale,
        bool update_statistics,
        bool apply_ratio_test,
        bool apply_view_gate,
        std::unordered_set<std::size_t>& used_map_point_ids,
        std::unordered_set<int>& used_feature_indices,
        PnPResult& result,
        std::vector<std::shared_ptr<MapPoint>>* visible_map_points = nullptr) const;

    PnPResult trackFrameByProjection(
        const std::vector<std::shared_ptr<MapPoint>>& map_points,                             
        const std::shared_ptr<Frame>& cur_frame,
        float radius_scale,
        bool update_statistics,
        bool apply_ratio_test,
        bool apply_view_gate,
        std::vector<std::shared_ptr<MapPoint>>* visible_map_points = nullptr) const;

    std::shared_ptr<Camera> camera_;
    const Matcher& matcher_;
    std::shared_ptr<PoseOptimizer> pose_optimizer_;

    double scale_factor_{1.2};
    int levels_num_{8};
    float base_projection_search_radius_{15.0f};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_TRACKER_H_
