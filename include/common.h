#ifndef MINI_ORB_SLAM_INCLUDE_COMMON_TYPES_H_
#define MINI_ORB_SLAM_INCLUDE_COMMON_TYPES_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv4/opencv2/core.hpp>

#include "feature.h"
#include "frame.h"
#include "map.h"
#include "map_point.h"

namespace mini_orb_slam
{

enum class TwoViewModel
{
    NONE = 0,
    HOMOGRAPHY = 1,
    FUNDAMENTAL = 2
};

struct PoseRecoveryResult
{
    bool success{false};

    TwoViewModel model{TwoViewModel::NONE};
    double model_score{0.0};
    int model_inlier_num{0};

    cv::Mat E;
    cv::Mat R;
    cv::Mat t;
    cv::Mat inlier_mask;

    std::vector<cv::Point2f> ref_inlier_points;
    std::vector<cv::Point2f> cur_inlier_points;
    std::vector<cv::Point2f> ref_norm_inlier_points;
    std::vector<cv::Point2f> cur_norm_inlier_points;

    std::vector<int> ref_feature_indices;
    std::vector<int> cur_feature_indices;
};

struct TriangulationResult
{
    bool success{false};

    int raw_point_num{0};
    int positive_depth_num{0};
    int reproj_valid_num{0};
    int good_parallax_num{0};

    double mean_reproj_error_ref{0.0};
    double mean_reproj_error_cur{0.0};
    double mean_parallax_deg{0.0};
    double check_rt_parallax_deg{0.0};
    int candidate_good_num{0};

    double good_point_ratio{0.0};
    double good_parallax_ratio{0.0};

    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2f> ref_points;
    std::vector<cv::Point2f> cur_points;

    std::vector<int> ref_feature_indices;
    std::vector<int> cur_feature_indices;
};

struct InitializationResult
{
    bool success{false};

    std::shared_ptr<Frame> ref_frame;
    std::shared_ptr<Frame> cur_frame;
    std::shared_ptr<Map> map;

    std::vector<std::shared_ptr<MapPoint>> map_points;
};

struct InitialMapPointState
{
    std::shared_ptr<MapPoint> map_point;
    cv::Point3d position;
};

struct InitialMapOptimizationResult
{
    bool success{false};

    int edge_num{0};
    int inlier_edge_num{0};

    double mean_reproj_error_before{0.0};
    double mean_reproj_error_after{0.0};

    cv::Mat optimized_ref_R;
    cv::Mat optimized_ref_t;
    cv::Mat optimized_cur_R;
    cv::Mat optimized_cur_t;

    std::vector<InitialMapPointState> optimized_map_points;
    std::vector<std::shared_ptr<MapPoint>> inlier_map_points;
};

struct PnPResult
{
    bool success{false};

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat R;
    cv::Mat inlier_indices;

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2f> img_points;

    std::vector<std::shared_ptr<MapPoint>> candidate_map_points;
    std::vector<std::shared_ptr<Feature>> candidate_features;

    int inlier_num{0};
    bool optimized{false};

    double ransac_reproj_error{0.0};
    double optimized_reproj_error{0.0};
};

struct TrackingResult
{
    bool success{false};

    std::shared_ptr<Frame> frame;

    cv::Mat R_cw;
    cv::Mat t_cw;

    std::vector<std::shared_ptr<MapPoint>> inlier_map_points;
    std::vector<std::shared_ptr<Feature>> inlier_features;
};

struct LocalBAResult
{
    bool solver_success{false};
    bool accepted{false};

    int edge_num{0};
    int rejected_edge_num{0};

    double seed_reproj_error{0.0};
    double candidate_seed_reproj_error{0.0};
    std::string rejection_reason;
};

struct LocalMappingResult
{
    bool success{false};

    std::size_t new_map_point_num{0};
    std::size_t culled_map_point_num{0};
    std::size_t culled_keyframe_num{0};

    bool local_ba_called{false};
    bool local_ba_solver_success{false};
    bool local_ba_success{false};
    bool local_ba_rejected{false};

    double local_ba_duration_ms{0.0};

    int local_ba_edge_num{0};
    int local_ba_rejected_edge_num{0};

    double local_ba_seed_reproj_error{0.0};
    double local_ba_candidate_seed_reproj_error{0.0};
    std::string local_ba_rejection_reason;
};

struct LocalMappingInput
{
    std::shared_ptr<Map> map;
    std::shared_ptr<Frame> ref_keyframe;
    std::shared_ptr<Frame> cur_keyframe;
    PnPResult tracking_seed;
};

struct LocalMappingOutput
{
    LocalMappingInput input;
    LocalMappingResult result;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_COMMON_TYPES_H_
