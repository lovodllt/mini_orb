#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Geometry>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/UInt64.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "frontend.h"

namespace mini_orb_slam
{

namespace
{

class ImageCallbackTiming
{
public:
    ImageCallbackTiming() : start_(std::chrono::steady_clock::now()) {}

    ~ImageCallbackTiming()
    {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count();
        ROS_INFO_STREAM_THROTTLE(1.0,
            "P2-EUROC-PERF-R23 image_callback_ms=" << elapsed_ms);
    }

private:
    std::chrono::steady_clock::time_point start_;
};

} // namespace

Frontend::Frontend(ros::NodeHandle& nh) : nh_(nh) {}

Frontend::~Frontend()
{
    // LocalMapping can enqueue into LoopCloser. Join it first while the
    // LoopCloser object is still alive, then stop the loop worker.
    if (local_mapper_ != nullptr)
    {
        local_mapper_->requestFinish();
        local_mapper_->join();
        local_mapper_->setLoopCloser(nullptr);
    }

    if (loop_closer_ != nullptr)
    {
        loop_closer_->requestFinish();
        loop_closer_->join();
    }
}

bool Frontend::init()
{
    return initImpl(true);
}

bool Frontend::initOffline()
{
    return initImpl(false);
}

bool Frontend::initImpl(bool enable_ros_transport)
{
    nh_.param("camera_topic", camera_topic_, std::string("/camera/image_raw"));
    nh_.param("camera_info_topic", camera_info_topic_, std::string("/camera/camera_info"));
    nh_.param("processed_image_topic", processed_image_topic_,
              camera_topic_ + "/processed");
    nh_.param("image_queue_size", image_queue_size_, image_queue_size_);
    image_queue_size_ = std::max(1, image_queue_size_);
    nh_.param("trajectory_output_path", trajectory_output_path_, std::string(""));
    nh_.param("trajectory_format", trajectory_format_, std::string("tum"));

    if (trajectory_format_ != "tum" && trajectory_format_ != "kitti")
    {
        ROS_ERROR_STREAM("Unsupported trajectory format: " << trajectory_format_);
        return false;
    }

    if (!openTrajectoryOutputFile(true))
    {
        ROS_ERROR("Failed to open trajectory output file.");
        return false;
    }

    nh_.param("use_camera_info", use_camera_info_, false);
    nh_.param("camera_info_required", camera_info_required_, false);
    nh_.param("min_keyframe_gap", min_keyframe_gap_, min_keyframe_gap_);
    nh_.param("max_keyframe_gap", max_keyframe_gap_, max_keyframe_gap_);
    nh_.param("tmp_lost_max_frames", tmp_lost_max_frames_, tmp_lost_max_frames_);
    nh_.param("max_tracking_reproj_error", max_tracking_reproj_error_, max_tracking_reproj_error_);
    nh_.param("min_tracking_inliers", min_tracking_inliers_, min_tracking_inliers_);
    nh_.param("min_recovery_seed_inliers", min_recovery_seed_inliers_, min_recovery_seed_inliers_);
    nh_.param("min_recovery_seed_inlier_ratio", min_recovery_seed_inlier_ratio_, min_recovery_seed_inlier_ratio_);
    nh_.param("max_recovery_seed_reproj_error", max_recovery_seed_reproj_error_, max_recovery_seed_reproj_error_);
    nh_.param("min_recovery_expand_inliers", min_recovery_expand_inliers_, min_recovery_expand_inliers_);
    nh_.param("min_recovery_expand_inlier_ratio", min_recovery_expand_inlier_ratio_, min_recovery_expand_inlier_ratio_);
    nh_.param("max_recovery_expand_reproj_error", max_recovery_expand_reproj_error_, max_recovery_expand_reproj_error_);
    nh_.param("min_recovery_inliers", min_recovery_inliers_, min_recovery_inliers_);
    nh_.param("min_recovery_inlier_ratio", min_recovery_inlier_ratio_, min_recovery_inlier_ratio_);
    nh_.param("max_recovery_reproj_error", max_recovery_reproj_error_, max_recovery_reproj_error_);
    nh_.param("relocalization_cooldown_frames", relocalization_cooldown_frames_, relocalization_cooldown_frames_);
    nh_.param("min_relocalization_inliers", min_relocalization_inliers_, min_relocalization_inliers_);
    nh_.param("relocalization_guard_frames", relocalization_guard_frames_, relocalization_guard_frames_);
    nh_.param("vocabulary_path", vocabulary_path_, std::string(""));
    nh_.param("max_relocalization_candidates", max_relocalization_candidates_, max_relocalization_candidates_);

    min_relocalization_inliers_ = std::max(min_recovery_inliers_, min_relocalization_inliers_);

    if (!vocabulary_path_.empty() && vocabulary_path_.front() != '/')
    {
        const std::string package_path = ros::package::getPath("mini_orb_slam");
        if (package_path.empty())
        {
            ROS_ERROR("Cannot resolve package-relative BoW vocabulary path: mini_orb_slam package was not found.");
            return false;
        }

        vocabulary_path_ = package_path + "/" + vocabulary_path_;
    }

    ros::NodeHandle slam_nh(nh_, "mini_orb_slam");
    slam_nh.param("min_initial_map_points", min_initial_map_points_, min_initial_map_points_);
    slam_nh.param("min_initial_good_point_ratio", min_initial_good_point_ratio_, min_initial_good_point_ratio_);
    slam_nh.param("max_initial_reproj_error_px", max_initial_reproj_error_px_, max_initial_reproj_error_px_);
    slam_nh.param("min_initial_parallax_deg", min_initial_parallax_deg_, min_initial_parallax_deg_);
    slam_nh.param("max_init_geometry_failures_per_ref", max_init_geometry_failures_per_ref_, max_init_geometry_failures_per_ref_);
    slam_nh.param("min_init_reference_features", min_init_reference_features_, min_init_reference_features_);
    slam_nh.param("min_init_reference_occupied_bins_", min_init_reference_occupied_bins_, min_init_reference_occupied_bins_);

    camera_ = std::make_shared<Camera>();

    const bool need_yaml_fallback = !use_camera_info_ || !camera_info_required_;

    if (need_yaml_fallback && !camera_->loadParams(nh_))
    {
        ROS_ERROR("Failed to load camera parameters from YAML.");
        return false;
    }

    if (use_camera_info_)
    {
        camera_info_sub_ = nh_.subscribe(camera_info_topic_, 1, &Frontend::cameraInfoCallback, this);
    }

    ros::NodeHandle orb_nh(nh_, "orb_extractor");
    if (!orb_extractor_.loadParams(orb_nh))
    {
        ROS_ERROR("Failed to load ORB extractor parameters.");
        return false;
    }

    ros::NodeHandle matcher_nh(nh_, "matcher");
    if (!matcher_.loadParams(matcher_nh))
    {
        ROS_ERROR("Failed to load matcher parameters.");
        return false;   
    }

    initializer_ = std::make_unique<Initializer>(camera_);

    pose_optimizer_ = std::make_shared<PoseOptimizer>(camera_,
                                                      orb_extractor_.getScaleFactor(),
                                                      orb_extractor_.getLevelsNum());
    tracker_ = std::make_unique<Tracker>(camera_, 
                                         matcher_, 
                                         pose_optimizer_,
                                         orb_extractor_.getScaleFactor(),
                                         orb_extractor_.getLevelsNum(),
                                         15.0f);

    std::shared_ptr<Initializer> local_mapper_initializer = 
        std::make_shared<Initializer>(camera_);
    local_mapper_ = std::make_unique<LocalMapper>(local_mapper_initializer,
                                                  matcher_,
                                                  pose_optimizer_,
                                                  orb_extractor_.getScaleFactor(),
                                                  orb_extractor_.getLevelsNum());

    if (!vocabulary_path_.empty())
    {
        bow_vocabulary_ = std::make_shared<BoWVocabulary>();
        if (!bow_vocabulary_->loadFromTextFile(vocabulary_path_))
        {
            ROS_ERROR_STREAM("Failed to load BoW vocabulary from: " << vocabulary_path_);
            return false;
        }
        else
        {
            keyframe_database_ = std::make_shared<KeyframeDatabase>(bow_vocabulary_);

            if (keyframe_database_ != nullptr)
                loop_closer_ = std::make_unique<LoopCloser>(keyframe_database_,
                                                            matcher_,
                                                            pose_optimizer_,
                                                            local_mapper_.get(),
                                                            orb_extractor_.getScaleFactor(),
                                                            orb_extractor_.getLevelsNum());

            ROS_INFO_STREAM("BoW vocabulary loaded from: " << vocabulary_path_);
        }
    }
    else 
    {
        ROS_WARN("BoW vocabulary path is empty. BoW-based relocalization will be disabled.");
    }

    if (local_mapper_ != nullptr)
    {
        local_mapper_->setKeyframeDatabase(bow_vocabulary_, keyframe_database_);
        local_mapper_->setLoopCloser(loop_closer_.get());
        local_mapper_->start();
    }

    if (loop_closer_ != nullptr)
        loop_closer_->start();

    if (enable_ros_transport)
    {
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/pose", 1);
        // Replay acknowledgement is a transport stream, not a latched status:
        // retain every count while the publisher is flow-controlled.
        processed_image_pub_ = nh_.advertise<std_msgs::UInt64>(processed_image_topic_, 600);
        image_sub_ = nh_.subscribe(camera_topic_, image_queue_size_, &Frontend::imageCallback, this);
    }

    ROS_INFO_STREAM("Frontend initialized. input mode: "
                    << (enable_ros_transport ? "ROS image transport" : "direct offline image")
                    << ", image topic: " << camera_topic_);
    return true;
}

void Frontend::run()
{
    ros::spin();
}

bool Frontend::openTrajectoryOutputFile(bool truncate)
{
    trajectory_output_.close();

    if (trajectory_output_path_.empty())
        return true;

    std::ios::openmode mode = std::ios::out;

    if (truncate)
        mode |= std::ios::trunc;
    else
        mode |= std::ios::app;

    trajectory_output_.open(trajectory_output_path_, mode);

    if (!trajectory_output_.is_open())
    {
        ROS_ERROR_STREAM("Failed to open trajectory output file: " << trajectory_output_path_);
        return false;
    }

    ROS_INFO_STREAM("Trajectory output file opened: " << trajectory_output_path_
                    << ", format: " << trajectory_format_);

    return true;        
}

std::shared_ptr<Frame> Frontend::buildFrame(std::size_t id, double timestamp, const cv::Mat& img)
{
    const auto build_start = std::chrono::steady_clock::now();
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(id, timestamp, img, camera_);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    const auto orb_start = std::chrono::steady_clock::now();
    const bool initialization_mode = status_ != FrontendStatus::TRACKING;
    orb_extractor_.extract(img, keypoints, descriptors, initialization_mode);
    const double orb_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - orb_start).count();

    const auto feature_wrap_start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<Feature>> features;
    features.reserve(keypoints.size());

    for (std::size_t i = 0; i < keypoints.size(); ++i)
        features.push_back(std::make_shared<Feature>(frame, keypoints[i], i));

    frame->setFeatures(features, descriptors);

    const double feature_wrap_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - feature_wrap_start).count();
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-EUROC-PERF-R25 frame_build orb_ms=" << orb_ms
        << " feature_wrap_ms=" << feature_wrap_ms
        << " total_ms=" << build_ms
        << " features=" << keypoints.size()
        << " initialization_mode=" << (initialization_mode ? 1 : 0)
        << " bow_deferred=1");

    return frame;
}

InitializationResult Frontend::buildInitialMap(const std::shared_ptr<Frame>& ref_frame,
                                               const std::shared_ptr<Frame>& cur_frame,
                                               const TriangulationResult& triangulation_result)
{
    InitializationResult result;

    if (ref_frame == nullptr || cur_frame == nullptr || !triangulation_result.success)
        return result;

    const std::vector<std::shared_ptr<Feature>>& ref_features = ref_frame->getFeatures();
    const std::vector<std::shared_ptr<Feature>>& cur_features = cur_frame->getFeatures();

    result.ref_frame = ref_frame;
    result.cur_frame = cur_frame;
    result.map = std::make_shared<Map>();
    result.map_points.reserve(triangulation_result.points_3d.size());

    for (std::size_t i = 0; i < triangulation_result.points_3d.size(); i++)
    {
        const int ref_idx = triangulation_result.ref_feature_indices[i];
        const int cur_idx = triangulation_result.cur_feature_indices[i];

        if (ref_idx < 0 || ref_idx >= static_cast<int>(ref_features.size()) ||
            cur_idx < 0 || cur_idx >= static_cast<int>(cur_features.size()))
        {
            ROS_WARN("Invalid feature indices for triangulation result.");
            continue;
        }

        const std::shared_ptr<Feature>& ref_feature = ref_features[ref_idx];
        const std::shared_ptr<Feature>& cur_feature = cur_features[cur_idx];

        if (ref_feature == nullptr || cur_feature == nullptr)
            continue;

        std::shared_ptr<MapPoint> map_point = 
            std::make_shared<MapPoint>(result.map->allocateMapPointId(), triangulation_result.points_3d[i]);

        map_point->setFirstKeyframeId(ref_frame->getId());

        map_point->setRefFeature(ref_feature);
        map_point->setCurFeature(cur_feature);

        map_point->addObservation(ref_feature);
        map_point->addObservation(cur_feature);

        ref_feature->setMapPoint(map_point);
        cur_feature->setMapPoint(map_point);

        result.map->addMapPoint(map_point);
        result.map_points.push_back(map_point);
    }

    result.success = !result.map_points.empty();
    return result;
}

bool Frontend::isInitialTriangulationAccepted(
    const TriangulationResult& triangulation_result) const
{
    if (!triangulation_result.success)
        return false;

    if (triangulation_result.points_3d.size() < min_initial_map_points_)
        return false;

    if (triangulation_result.good_point_ratio < min_initial_good_point_ratio_)
        return false;

    if (!std::isfinite(triangulation_result.mean_reproj_error_ref) ||
        !std::isfinite(triangulation_result.mean_reproj_error_cur) ||
        triangulation_result.mean_reproj_error_ref > max_initial_reproj_error_px_ ||
        triangulation_result.mean_reproj_error_cur > max_initial_reproj_error_px_)
    {
        return false;
    }

    if (!std::isfinite(triangulation_result.check_rt_parallax_deg) ||
        triangulation_result.check_rt_parallax_deg < min_initial_parallax_deg_)
    {
        return false;
    }

    return true;
}

bool Frontend::isInitialMapOptimizationAccepted(
    const InitialMapOptimizationResult& optimization_result) const
{
    if (!optimization_result.success)
        return false;

    const int min_inlier_edges = 2 * min_initial_map_points_;

    return optimization_result.inlier_edge_num >= min_inlier_edges &&
           optimization_result.inlier_map_points.size() >= min_initial_map_points_ &&
           std::isfinite(optimization_result.mean_reproj_error_after) &&
           optimization_result.mean_reproj_error_after >= 0.0 &&
           optimization_result.mean_reproj_error_after <= max_initial_reproj_error_px_;
}

bool Frontend::retainInitialMapInliers(InitializationResult& init_result,
                                       const InitialMapOptimizationResult& optimization_result) const
{
    if (init_result.map == nullptr)
        return false;

    std::unordered_set<std::size_t> inlier_ids;
    inlier_ids.reserve(optimization_result.inlier_map_points.size() * 2 + 1);

    for (const auto& map_point : optimization_result.inlier_map_points)
    {
        if (map_point != nullptr && !map_point->isBad())
            inlier_ids.insert(map_point->getId());
    }

    std::vector<std::shared_ptr<MapPoint>> retained_map_points;
    retained_map_points.reserve(inlier_ids.size());

    for (const auto& map_point : init_result.map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (inlier_ids.count(map_point->getId()) == 0)
        {
            map_point->setBad(true);
            continue;
        }

        retained_map_points.push_back(map_point);
    }

    init_result.map->removeBadMapPoints();
    init_result.map_points.swap(retained_map_points);

    return (init_result.map_points.size() >= min_initial_map_points_);
}

bool Frontend::normalizeInitialMapScale(InitializationResult& init_result,
                                        double& median_depth) const
{
    median_depth = 0.0;

    if (!init_result.success ||
        init_result.ref_frame == nullptr ||
        init_result.cur_frame == nullptr)
    {
        return false;
    }

    std::vector<double> depths;
    depths.reserve(init_result.map_points.size());

    for (const auto& map_point : init_result.map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const double depth = map_point->getPos().z;
        if (std::isfinite(depth) && depth > 1e-6)
            depths.push_back(depth);
    }

    if (depths.empty())
        return false;

    const std::size_t median_index = depths.size() / 2;
    std::nth_element(depths.begin(), depths.begin() + median_index, depths.end());

    median_depth = depths[median_index];

    if (!std::isfinite(median_depth) || median_depth <= 1e-6)
        return false;

    cv::Mat scale_R;
    cv::Mat scale_t;
    init_result.cur_frame->copyPose(scale_R, scale_t);
    scale_t /= median_depth;

    if (!cv::checkRange(scale_t))
        return false;

    init_result.cur_frame->setPose(scale_R, scale_t);

    for (const auto& map_point : init_result.map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const cv::Point3d& point = map_point->getPos();
        map_point->setPos(
            cv::Point3d(point.x / median_depth, 
                        point.y / median_depth, 
                        point.z / median_depth));
    }

    return true;
}

void Frontend::rollbackInitialMapAttempt(const std::shared_ptr<Frame>& cur_frame,
                                         const cv::Mat& cur_R_before, 
                                         const cv::Mat& cur_t_before)
{
    for (const auto& map_point : init_result_.map_points)
    {
        if (map_point != nullptr)
            map_point->setBad(true);
    }

    if (init_result_.map != nullptr)
        init_result_.map->removeBadMapPoints();

    init_result_ = {};
    tracking_result_ = {};

    if (cur_frame != nullptr &&
        !cur_R_before.empty() &&
        !cur_t_before.empty())
    {
        cur_frame->setPose(cur_R_before, cur_t_before);
    }
}

bool Frontend::isQualifiedInitializationReference(const std::shared_ptr<Frame>& frame) const
{
    if (frame == nullptr || !frame->hasFeatures())
        return false;

    if (frame->getFeatureNum() < min_init_reference_features_)
        return false;

    const cv::Mat& img = frame->getImg();
    if (img.empty() || img.cols <= 0 || img.rows <= 0)
        return false;

    constexpr int kGridCols = 4;
    constexpr int kGridRows = 4;

    std::array<bool, kGridCols * kGridRows> occupied{};
    bool has_top = false;
    bool has_bottom = false;
    bool has_left = false;
    bool has_right = false;

    for (const auto& keypoint : frame->getKeypoints())
    {
        const float x = keypoint.pt.x;
        const float y = keypoint.pt.y;

        if (x < 0.0f || y < 0.0f || x >= img.cols || y >= img.rows)
            continue;

        const int col = std::min(kGridCols - 1, static_cast<int>(kGridCols * x / img.cols));
        const int row = std::min(kGridRows - 1, static_cast<int>(kGridRows * y / img.rows));

        occupied[row * kGridCols + col] = true;

        if (col < kGridCols / 2)
            has_left = true;
        else
            has_right = true;

        if (row < kGridRows / 2)
            has_top = true;
        else
            has_bottom = true;
    }

    const int occupied_bins = static_cast<int>(std::count(occupied.begin(), occupied.end(), true));

    return occupied_bins >= min_init_reference_occupied_bins_ &&
           has_left && has_right && has_top && has_bottom;
}

bool Frontend::initializeFromRefAndCur(const std::shared_ptr<Frame>& ref_frame, 
                                       const std::shared_ptr<Frame>& cur_frame,
                                       bool& should_reanchor)
{
    if (ref_frame == nullptr || cur_frame == nullptr)
        return false;

    should_reanchor = false;

    const auto registerGeometryFailure =
        [this, &should_reanchor]() -> int
        {
            const int failure_num = ++init_geometry_failures_num_;

            if (failure_num >= max_init_geometry_failures_per_ref_)
            {
                should_reanchor = true;
                init_geometry_failures_num_ = 0;
            }

            return failure_num;
        };

    cv::Mat original_ref_R, original_ref_t;
    cv::Mat original_cur_R, original_cur_t;

    ref_frame->copyPose(original_ref_R, original_ref_t);
    cur_frame->copyPose(original_cur_R, original_cur_t);

    if (original_ref_R.empty() || original_ref_t.empty() ||
        original_cur_R.empty() || original_cur_t.empty())
    {
        return false;
    }

    const auto relativeRotationAngleDeg =
        [this](const cv::Mat& current_R, const cv::Mat& reference_R) -> double
        {
            if (current_R.rows != 3 || current_R.cols != 3 ||
                reference_R.rows != 3 || reference_R.cols != 3)
            {
                return -1.0;
            }

            cv::Mat current_R64;
            cv::Mat reference_R64;
            current_R.convertTo(current_R64, CV_64F);
            reference_R.convertTo(reference_R64, CV_64F);

            const cv::Mat relative_R = current_R64 * reference_R64.t();
            if (!cv::checkRange(relative_R))
                return -1.0;

            return computeRotationAngleDeg(relative_R);
        };

    if (initialization_match_reference_id_ != ref_frame->getId())
    {
        initialization_match_reference_id_ = ref_frame->getId();
        initialization_previous_matched_.clear();
    }

    const std::vector<std::pair<int, int>> match_indices =
        matcher_.matchFramesForInitialization(*ref_frame,
                                              *cur_frame,
                                              initialization_previous_matched_);

    ROS_INFO_STREAM("Initialization candidate matches: " << match_indices.size());

    if (match_indices.size() < 8)
    {
        ROS_WARN("Not enough matches for initialization: %zu", match_indices.size());
        should_reanchor = true;
        init_geometry_failures_num_ = 0;
        return false;
    }

    constexpr std::size_t kMinInitializationMatches = 100;
    if (match_indices.size() < kMinInitializationMatches)
    {
        should_reanchor = true;
        init_geometry_failures_num_ = 0;
        return false;
    }

    PoseRecoveryResult pose_result = 
        initializer_->recoverPoseFromFrames(ref_frame, cur_frame, match_indices);
    if (!pose_result.success)
    {
        const int failure_num = registerGeometryFailure();

        ROS_WARN_STREAM("Pose recovery failed during initialization. "
                       << "geometry_failure_streak=" << failure_num
                       << ", reanchor_next=" << (should_reanchor ? "true" : "false"));
                
        return false;
    }

    const char* model_name = "none";
    if (pose_result.model == TwoViewModel::HOMOGRAPHY)
        model_name = "homography";
    else if (pose_result.model == TwoViewModel::FUNDAMENTAL)
        model_name = "fundamental"; 

    ROS_INFO_STREAM(
        "P2-DEBUG-INIT-03 stage=two_view"
        << " ref_frame=" << ref_frame->getId()
        << " cur_frame=" << cur_frame->getId()
        << " model=" << model_name
        << " relative_rotation_deg=" << computeRotationAngleDeg(pose_result.R)
        << " model_inliers=" << pose_result.model_inlier_num
        << " inliers=" << (pose_result.inlier_mask.empty()
                               ? 0
                               : cv::countNonZero(pose_result.inlier_mask)));

    ROS_INFO_STREAM("Pose recovery model: " << model_name 
                    << ", score: " << pose_result.model_score
                    << ", inliers: " << pose_result.model_inlier_num);

    TriangulationResult triangulation_result = 
        initializer_->triangulateFromPose(pose_result);
    if (!isInitialTriangulationAccepted(triangulation_result))
    {
        const int failure_num = registerGeometryFailure();

        ROS_WARN_STREAM("Triangulation failed during initialization. "
                        << "raw=" << triangulation_result.raw_point_num
                        << ", positive_depth=" << triangulation_result.positive_depth_num
                        << ", reproj_valid=" << triangulation_result.reproj_valid_num
                        << ", good_parallax=" << triangulation_result.good_parallax_num
                        << ", kept=" << triangulation_result.points_3d.size()
                        << ", good_point_ratio=" << triangulation_result.good_point_ratio
                        << ", mean_parallax_deg=" << triangulation_result.mean_parallax_deg
                        << ", reproj_ref=" << triangulation_result.mean_reproj_error_ref
                        << ", reproj_cur=" << triangulation_result.mean_reproj_error_cur
                        << ", geometry_failure_streak=" << failure_num
                        << ", reanchor_next=" << (should_reanchor ? "true" : "false"));

        return false;
    }

    cv::Mat R_ref = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_ref = cv::Mat::zeros(3, 1, CV_64F);

    ref_frame->setPose(R_ref, t_ref);
    cur_frame->setPose(pose_result.R, pose_result.t);

    cv::Mat initial_ref_R;
    cv::Mat initial_ref_t;
    cv::Mat initial_cur_R;
    cv::Mat initial_cur_t;
    ref_frame->copyPose(initial_ref_R, initial_ref_t);
    cur_frame->copyPose(initial_cur_R, initial_cur_t);

    ROS_INFO_STREAM(
        "P2-DEBUG-INIT-03 stage=before_initial_ba"
        << " ref_frame=" << ref_frame->getId()
        << " cur_frame=" << cur_frame->getId()
        << " model=" << model_name
        << " relative_rotation_deg="
        << relativeRotationAngleDeg(initial_cur_R, initial_ref_R));

    init_result_ = buildInitialMap(ref_frame, cur_frame, triangulation_result);
    if (!init_result_.success)
    {
        ROS_WARN("Failed to build initial map.");

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    const InitialMapOptimizationResult initial_ba_result = 
        pose_optimizer_->optimizeInitialMap(init_result_.map, ref_frame, cur_frame);

    ROS_INFO_STREAM(
        "P2-DEBUG-INIT-03 stage=after_initial_ba"
        << " ref_frame=" << ref_frame->getId()
        << " cur_frame=" << cur_frame->getId()
        << " model=" << model_name
        << " relative_rotation_deg="
        << relativeRotationAngleDeg(initial_ba_result.optimized_cur_R, 
                                    initial_ba_result.optimized_ref_R)
        << " accepted=" << initial_ba_result.success
        << " reproj_before=" << initial_ba_result.mean_reproj_error_before
        << " reproj_after=" << initial_ba_result.mean_reproj_error_after);

    if (!isInitialMapOptimizationAccepted(initial_ba_result))
    {
        ROS_WARN_STREAM("Initial BA rejected. edges=" << initial_ba_result.edge_num
                        << ", inlier_edges=" << initial_ba_result.inlier_edge_num
                        << ", inlier_points="
                        << initial_ba_result.inlier_map_points.size()
                        << ", reproj_before="
                        << initial_ba_result.mean_reproj_error_before
                        << ", reproj_after="
                        << initial_ba_result.mean_reproj_error_after);

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    ref_frame->setPose(initial_ba_result.optimized_ref_R, initial_ba_result.optimized_ref_t);
    cur_frame->setPose(initial_ba_result.optimized_cur_R, initial_ba_result.optimized_cur_t);

    for (const auto& point_state : initial_ba_result.optimized_map_points)
    {
        if (point_state.map_point == nullptr || point_state.map_point->isBad())
            continue;

        point_state.map_point->setPos(point_state.position);
    }

    if (!retainInitialMapInliers(init_result_, initial_ba_result))
    {
        ROS_WARN("Failed to retain inlier map points after initial BA.");

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    double median_depth = 0.0;
    if (!normalizeInitialMapScale(init_result_, median_depth))
    {
        ROS_WARN("Failed to normalize initial map scale.");

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    ROS_INFO_STREAM("Initial BA accepted. edges=" << initial_ba_result.edge_num
                    << ", inlier_edges=" << initial_ba_result.inlier_edge_num
                    << ", retained_points=" << init_result_.map_points.size()
                    << ", reproj_before="
                    << initial_ba_result.mean_reproj_error_before
                    << ", reproj_after="
                    << initial_ba_result.mean_reproj_error_after
                    << ", median_depth=" << median_depth);

    const PnPResult pnp_result = tracker_->estimatePoseByPnP(init_result_);
    if (!pnp_result.success)
    {
        ROS_WARN("PnP pose estimation failed after initial BA.");

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    ref_frame->copyPose(initial_ref_R, initial_ref_t);
    cur_frame->copyPose(initial_cur_R, initial_cur_t);

    ROS_INFO_STREAM(
        "P2-DEBUG-INIT-03 stage=post_initial_ba_pnp"
        << " ref_frame=" << ref_frame->getId()
        << " cur_frame=" << cur_frame->getId()
        << " model=" << model_name
        << " relative_rotation_deg="
        << relativeRotationAngleDeg(pnp_result.R, initial_ref_R)
        << " delta_from_two_view_deg="
        << relativeRotationAngleDeg(pnp_result.R, pose_result.R)
        << " delta_from_ba_deg="
        << relativeRotationAngleDeg(pnp_result.R, initial_cur_R)
        << " inliers=" << pnp_result.inlier_num
        << " optimized_reproj_error=" << pnp_result.optimized_reproj_error);

    tracking_result_ = tracker_->buildTrackingResult(init_result_, pnp_result);
    if (!tracking_result_.success)
    {
        ROS_WARN("Failed to build tracking result after initial BA.");

        should_reanchor = true;
        init_geometry_failures_num_ = 0;

        rollbackInitialMapAttempt(cur_frame,
                                  original_cur_R,
                                  original_cur_t);

        return false;
    }

    cur_frame->setPose(pnp_result.R, pnp_result.tvec);

    ref_frame->setKeyframe(true);
    cur_frame->setKeyframe(true);

    init_result_.map->addKeyframe(ref_frame);
    init_result_.map->addKeyframe(cur_frame);

    registerKeyframeInDatabase(ref_frame);
    registerKeyframeInDatabase(cur_frame);

    for (const auto& map_point : init_result_.map_points)
    {
        if (map_point == nullptr)
            continue;

        map_point->updateViewStatistics(orb_extractor_.getScaleFactor(),
                                        orb_extractor_.getLevelsNum());
        map_point->updateRepresentativeDescriptor();
    }

    resetMotionModel();
    initialization_previous_matched_.clear();
    initialization_match_reference_id_ = std::numeric_limits<std::size_t>::max();
    last_tracked_frame_ = cur_frame;
    tracking_reference_keyframe_ = cur_frame;
    init_geometry_failures_num_ = 0;

    relocalization_cooldown_remaining_ = 0;
    relocalization_guard_remaining_ = 0;
    status_ = FrontendStatus::TRACKING;

    publishCurrentPose(cur_frame);

    ROS_INFO_STREAM("Initialization success. map points: " << init_result_.map_points.size()
                    << ", tracking inliers: " << tracking_result_.inlier_map_points.size()
                    << ", pnp inliers: " << pnp_result.inlier_num
                    << ", pnp_optimized: " << (pnp_result.optimized ? "true" : "false")
                    << ", pnp_reproj_error: " << pnp_result.optimized_reproj_error
                    << ", mean parallax(deg): " << triangulation_result.mean_parallax_deg);

    return true;
}

void Frontend::updateTrackObservations(const TrackingResult& tracking_result)
{
    if (!tracking_result.success)
        return;

    const std::size_t pair_num = std::min(tracking_result.inlier_map_points.size(), 
                                          tracking_result.inlier_features.size());

    for (std::size_t i = 0; i < pair_num; i++)
    {
        const std::shared_ptr<MapPoint>& map_point = tracking_result.inlier_map_points[i];
        const std::shared_ptr<Feature>& feature = tracking_result.inlier_features[i];

        if (map_point == nullptr || feature == nullptr)
            continue;

        // ORB-SLAM2 logic reference: a MapPoint's persistent observations are
        // KeyFrame-owned.  A normal tracking frame keeps only its local
        // Feature -> MapPoint association; LocalMapper commits the reverse
        // observation when that frame becomes a keyframe.
        feature->setMapPoint(map_point);
    }
}

void Frontend::updateTrackingReferenceKeyframe(
    const std::shared_ptr<Map>& map,
    const TrackingResult& tracking_result)
{
    if (map == nullptr || !tracking_result.success)
    {
        tracking_reference_keyframe_.reset();
        return;
    }

    // collectLocalKeyframes ranks keyframes by observations from current inlier points.
    const std::vector<std::shared_ptr<Frame>> local_keyframes =
        collectLocalKeyframes(map, tracking_result);

    if (!local_keyframes.empty() && local_keyframes.front() != nullptr)
        tracking_reference_keyframe_ = local_keyframes.front();
}

double Frontend::computeRotationAngleDeg(const cv::Mat& R) const
{
    if (R.empty())
        return 0.0;

    cv::Mat R_copy;
    R.convertTo(R_copy, CV_64F);

    const double trace_val = R_copy.at<double>(0, 0) + R_copy.at<double>(1, 1) + R_copy.at<double>(2, 2);

    double cos_theta = (trace_val - 1.0) * 0.5;
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));

    return std::acos(cos_theta) * 180.0 / CV_PI;
}

bool Frontend::shouldInsertKeyframe(const std::shared_ptr<Map>& map, 
                                    const std::shared_ptr<Frame>& cur_frame, 
                                    const TrackingResult& tracking_result)
{
    if (map == nullptr || cur_frame == nullptr || !tracking_result.success)
        return false;

    if (relocalization_cooldown_remaining_ > 0)
        return false;

    const std::shared_ptr<Frame> last_keyframe = map->getLastKeyframe();
    if (last_keyframe == nullptr)
        return true;

    std::shared_ptr<Frame> reference_keyframe = tracking_reference_keyframe_.lock();
    if (reference_keyframe == nullptr || !reference_keyframe->isKeyframe())
        reference_keyframe = last_keyframe;

    if (reference_keyframe == nullptr || !reference_keyframe->isKeyframe())
        return false;

    if (local_mapper_ != nullptr &&
       (local_mapper_->isStopped() || local_mapper_->stopRequested()))
    {
        return false;
    }

    const std::size_t keyframe_num = map->getKeyframeNum();
    constexpr std::size_t kMinTrackingInliers = 15;
    constexpr double kMonoReferenceRatio = 0.90;

    const std::size_t current_inlier_num = tracking_result.inlier_map_points.size();
    if (current_inlier_num <= kMinTrackingInliers)
        return false;

    const int min_observation_num = keyframe_num <= 2 ? 2 : 3;

    std::size_t reference_reliable_point_num = 0;
    for (const auto& feature : reference_keyframe->getFeatures())
    {
        if (feature == nullptr || !feature->hasMapPoint())
            continue;

        const std::shared_ptr<MapPoint>& map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        if (map_point->getKeyframeObservationCount() < min_observation_num)
            continue;

        reference_reliable_point_num++;
    }

    if (reference_reliable_point_num == 0)
        reference_reliable_point_num = current_inlier_num;

    const std::size_t frame_gap = cur_frame->getId() - last_keyframe->getId();

    const bool force_insert = frame_gap >= max_keyframe_gap_;
    const bool allow_insert = frame_gap >= min_keyframe_gap_;
    const bool weak_tracking = current_inlier_num < 
                               (kMonoReferenceRatio * reference_reliable_point_num);

    keyframe_decision_num_++;

    const bool insert = force_insert || (allow_insert && weak_tracking);
    if (insert)
    {
        if (force_insert)
            keyframe_force_insert_num_++;
        else
            keyframe_weak_insert_num_++;
    }
    ROS_INFO_STREAM(
        "P2-KITTI-DIAG-R131 keyframe_admission frame=" << cur_frame->getId()
        << " reference=" << reference_keyframe->getId()
        << " map_kf=" << keyframe_num
        << " gap=" << frame_gap
        << " inliers=" << current_inlier_num
        << " reference_reliable=" << reference_reliable_point_num
        << " allow=" << (allow_insert ? 1 : 0)
        << " weak=" << (weak_tracking ? 1 : 0)
        << " force=" << (force_insert ? 1 : 0)
        << " insert=" << (insert ? 1 : 0));
    ROS_INFO_STREAM_THROTTLE(1.0,
        "P2-EUROC-PERF-R29 keyframe_decision total=" << keyframe_decision_num_
        << " busy_rejected=" << keyframe_busy_rejected_num_
        << " commit_barrier=" << keyframe_commit_barrier_num_
        << " force_insert=" << keyframe_force_insert_num_
        << " weak_insert=" << keyframe_weak_insert_num_
        << " frame_gap=" << frame_gap
        << " current_inliers=" << current_inlier_num
        << " reference_reliable=" << reference_reliable_point_num);
    return insert;
}

void Frontend::appendUniqueMapPoint(const std::shared_ptr<MapPoint>& map_point, 
                                    std::vector<std::shared_ptr<MapPoint>>& local_map_points,
                                    std::unordered_set<std::size_t>& local_map_point_ids) const
{
    if (map_point == nullptr || map_point->isBad())
        return;

    if (local_map_point_ids.insert(map_point->getId()).second)
        local_map_points.push_back(map_point);
}

std::vector<std::shared_ptr<Frame>> Frontend::collectLocalKeyframes(
    const std::shared_ptr<Map>& map,
    const TrackingResult& tracking_result) const
{
    std::vector<std::shared_ptr<Frame>> local_keyframes;

    if (map == nullptr || !tracking_result.success)
        return local_keyframes;

    std::unordered_map<std::size_t, int> keyframe_votes;
    std::unordered_map<std::size_t, std::shared_ptr<Frame>> voted_keyframe;

    for (const auto& map_point : tracking_result.inlier_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::vector<std::shared_ptr<Feature>> observations = 
            map_point->getKeyframeObservations();

        for (const auto& feature : observations)
        {
            if (feature == nullptr)
                continue;

            const std::shared_ptr<Frame> keyframe = feature->getFrame();
            if (keyframe == nullptr || !keyframe->isKeyframe())
                continue;

            keyframe_votes[keyframe->getId()]++;
            voted_keyframe[keyframe->getId()] = keyframe;
        }
    }

    struct KeyframeVote
    {
        std::shared_ptr<Frame> keyframe;
        int votes{0};
    };

    std::vector<KeyframeVote> ranked_keyframes;
    ranked_keyframes.reserve(keyframe_votes.size());

    for (const auto& item : voted_keyframe)    
    {
        const auto vote_it = keyframe_votes.find(item.first);
        if (vote_it == keyframe_votes.end() || item.second == nullptr)
            continue;

        ranked_keyframes.push_back({item.second, vote_it->second});
    }

    std::sort(ranked_keyframes.begin(), ranked_keyframes.end(),
              [](const KeyframeVote& a, const KeyframeVote& b) 
              {
                if (a.votes != b.votes)
                    return a.votes > b.votes;
                 
                return a.keyframe->getId() > b.keyframe->getId();
              });

    // ORB-SLAM2 logic reference: every keyframe observing a current-frame
    // inlier MapPoint becomes a local seed. Each seed then contributes at
    // most one covisible neighbor. This keeps the local map tied to actual
    // observations rather than to recency alone. The 80-frame limit is only
    // a computational safety bound; it is not a scene-specific threshold.
    constexpr std::size_t kMaxLocalKeyframes = 80;

    std::unordered_set<std::size_t> selected_ids;
    selected_ids.reserve(kMaxLocalKeyframes * 2);

    std::vector<std::shared_ptr<Frame>> seed_keyframes;
    seed_keyframes.reserve(kMaxLocalKeyframes);

    const std::size_t ranked_keyframe_num = ranked_keyframes.size();

    for (std::size_t i = 0;
         i < ranked_keyframes.size() && local_keyframes.size() < kMaxLocalKeyframes;
         i++)
    {
        const std::shared_ptr<Frame>& keyframe = ranked_keyframes[i].keyframe;
        if (keyframe == nullptr)
            continue;

        if (selected_ids.insert(keyframe->getId()).second)
        {
            seed_keyframes.push_back(keyframe);
            local_keyframes.push_back(keyframe);
        }
    }

    for (const auto& seed_keyframe : seed_keyframes)
    {
        if (seed_keyframe == nullptr)
            continue;

        const std::vector<std::shared_ptr<Frame>> neighbors = 
            seed_keyframe->getBestCovisibilityKeyframes(10);

        for (const auto& neighbor : neighbors)
        {
            if (neighbor == nullptr)
                continue;

            if (selected_ids.insert(neighbor->getId()).second)
            {
                local_keyframes.push_back(neighbor);
                break;
            }

            if (local_keyframes.size() >= kMaxLocalKeyframes)
                break;
        }

        if (local_keyframes.size() >= kMaxLocalKeyframes)
            break;
    }

    const std::size_t selected_after_neighbors = local_keyframes.size();

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R12 frame="
                    << (tracking_result.frame != nullptr ? tracking_result.frame->getId() : 0)
                    << " stage=local_keyframes ranked=" << ranked_keyframe_num
                    << " seed=" << seed_keyframes.size()
                    << " after_neighbors=" << selected_after_neighbors
                    << " final=" << local_keyframes.size()
                    << " fallback_recent=0"
                    << " map_keyframes=" << map->getKeyframes().size());

    return local_keyframes;
}

std::vector<std::shared_ptr<Frame>> Frontend::collectRelocalizationCandidates(
    const std::shared_ptr<Map>& map, 
    const std::shared_ptr<Frame>& cur_frame) const
{
    std::vector<std::shared_ptr<Frame>> candidates;

    if (map == nullptr || cur_frame == nullptr || !cur_frame->hasFeatures())
        return candidates;

    const std::vector<std::shared_ptr<Frame>> map_keyframes = map->copyKeyframes();

    // ORB-SLAM2 logic reference: normal motion-model tracking does not build
    // BoW. Relocalization is one of the explicit paths that needs it.
    if (keyframe_database_ != nullptr && bow_vocabulary_ != nullptr &&
        !cur_frame->hasBoW())
    {
        cur_frame->computeBoW(bow_vocabulary_);
    }

    const auto ensureInitialAnchor = [&map_keyframes, &candidates, this]()
    {
        if (map_keyframes.empty() || map_keyframes.front() == nullptr ||
            !map_keyframes.front()->isKeyframe() || max_relocalization_candidates_ <= 0)
        {
            return;
        }

        const std::shared_ptr<Frame>& anchor = map_keyframes.front();
        const auto existing = std::find(candidates.begin(), candidates.end(), anchor);
        if (existing != candidates.end())
            return;

        if (candidates.size() >= static_cast<std::size_t>(max_relocalization_candidates_))
            candidates.pop_back();

        candidates.insert(candidates.begin(), anchor);
    };

    if (keyframe_database_ != nullptr && cur_frame->hasBoW())
    {
        struct CandidateGroup
        {
            std::shared_ptr<Frame> representative;
            double bow_score{0.0};
            double group_score{0.0};
            int support_num{0};
        };

        constexpr int kMaxSeedCandidates = 20;
        constexpr int kMaxNeighborNum = 10;
        constexpr double kMinBowScore = 0.01;
        constexpr double kNeighborScoreWeight = 0.75;
        constexpr double kKeepBestGroupRatio = 0.75;

        const std::vector<KeyframeQueryResult> bow_results = 
            keyframe_database_->query(cur_frame, 
                                      std::max(max_relocalization_candidates_, kMaxSeedCandidates),
                                      kMinBowScore);

        if (!bow_results.empty())
        {
            std::unordered_map<std::size_t, double> seed_scores;
            seed_scores.reserve(bow_results.size() * 2 + 1);

            for (const auto& result : bow_results)
            {
                if (result.keyframe == nullptr || !result.keyframe->isKeyframe())
                    continue;

                seed_scores[result.keyframe->getId()] = result.score;
            }

            std::unordered_map<std::size_t, CandidateGroup> grouped_candidates;
            grouped_candidates.reserve(bow_results.size() * 2 + 1);

            double best_group_score = 0.0;

            for (const auto& result : bow_results)
            {
                if (result.keyframe == nullptr || !result.keyframe->isKeyframe())
                    continue;

                std::shared_ptr<Frame> best_group_keyframe = result.keyframe;
                double best_group_bow_score = result.score;
                double group_score = result.score;
                int support_num = 0;

                const std::vector<std::shared_ptr<Frame>> neighbors = 
                    result.keyframe->getBestCovisibilityKeyframes(kMaxNeighborNum, 1);

                for (const auto& neighbor : neighbors)
                {
                    if (neighbor == nullptr || !neighbor->isKeyframe())
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
                if (group_it == grouped_candidates.end() || group_score > group_it->second.group_score)
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

            std::vector<CandidateGroup> ranked_groups;
            ranked_groups.reserve(grouped_candidates.size());

            for (const auto& item : grouped_candidates)
                ranked_groups.push_back(item.second);

            std::sort(ranked_groups.begin(), ranked_groups.end(),
                      [](const CandidateGroup& a, const CandidateGroup& b) 
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

                candidates.push_back(group.representative);

                if (candidates.size() >= max_relocalization_candidates_)
                    break;
            }

            if (candidates.empty())
            {
                for (const auto& group : ranked_groups)
                {
                    if (group.representative == nullptr)
                        continue;

                    candidates.push_back(group.representative);

                    if (candidates.size() >= max_relocalization_candidates_)
                        break;
                }
            }

            if (!candidates.empty())
            {
                // BoW can return a recent keyframe with very few map
                // observations while omitting the dense initial anchor.
                ensureInitialAnchor();
                return candidates;
            }
        }
    }

    struct KeyframeScore
    {
        std::shared_ptr<Frame> keyframe;
        int map_match_num{0};
        double group_score{0.0};
        int support_num{0};
    };

    std::vector<KeyframeScore> scored_keyframes;
    scored_keyframes.reserve(map_keyframes.size() * 2 + 1);

    std::unordered_map<std::size_t, int> base_scores;
    base_scores.reserve(map_keyframes.size() * 2 + 1);

    for (const auto& keyframe : map_keyframes)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe() || !keyframe->hasFeatures())
            continue;

        const std::vector<std::pair<int, int>> matches = 
            matcher_.matchFrames(*keyframe, *cur_frame);

        if (matches.size() < 15)
            continue;

        const std::vector<std::shared_ptr<Feature>>& keyframe_features = keyframe->getFeatures();

        std::unordered_set<std::size_t> matched_map_point_ids;
        matched_map_point_ids.reserve(matches.size());

        int unique_map_match_num = 0;

        for (const auto& match_idx : matches)
        {
            const int keyframe_idx = match_idx.first;
            if (keyframe_idx < 0 || keyframe_idx >= 
                static_cast<int>(keyframe_features.size()))
            {
                continue;
            }

            const std::shared_ptr<Feature>& feature = keyframe_features[keyframe_idx];
            if (feature == nullptr || !feature->hasMapPoint())
                continue;

            const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
            if (map_point == nullptr || map_point->isBad())
                continue;

            if (matched_map_point_ids.insert(map_point->getId()).second)
                unique_map_match_num++;
        }

        if (unique_map_match_num < 10)
            continue;

        scored_keyframes.push_back({keyframe,
                                    unique_map_match_num,
                                    static_cast<double>(unique_map_match_num),
                                    0});

        base_scores[keyframe->getId()] = unique_map_match_num;
    }

    if (scored_keyframes.empty())
        return candidates;

    constexpr double kNeightborSupportWeight = 0.5;

    for (auto& item : scored_keyframes)
    {
        const std::vector<std::shared_ptr<Frame>> neighbors = 
            item.keyframe->getBestCovisibilityKeyframes(4, 4);

        double group_score = static_cast<double>(item.map_match_num);
        int support_num = 0;

        for (const auto& neighbor : neighbors)
        {
            if (neighbor == nullptr)
                continue;

            const auto it = base_scores.find(neighbor->getId());
            if (it == base_scores.end())
                continue;

            group_score += kNeightborSupportWeight * it->second;
            support_num++;
        }

        item.group_score = group_score;
        item.support_num = support_num;
    }

    std::sort(scored_keyframes.begin(), scored_keyframes.end(),
              [](const KeyframeScore& a, const KeyframeScore& b) 
              {
                if (a.group_score != b.group_score)
                    return a.group_score > b.group_score;

                if (a.map_match_num != b.map_match_num)
                    return a.map_match_num > b.map_match_num;

                if (a.support_num != b.support_num)
                    return a.support_num > b.support_num;

                return a.keyframe->getId() > b.keyframe->getId();
              });

    constexpr std::size_t kMaxRelocCandidates = 5;
    constexpr double kMinGroupScore = 12.0;

    for (std::size_t i = 0; 
        i < scored_keyframes.size() && candidates.size() < kMaxRelocCandidates; 
        i++)
    {
        if (scored_keyframes[i].group_score < kMinGroupScore)
            continue;

        candidates.push_back(scored_keyframes[i].keyframe);
    }

    if (candidates.empty())
    {
        for (std::size_t i = 0; 
             i < scored_keyframes.size() && candidates.size() < kMaxRelocCandidates; 
             i++)
        {
            candidates.push_back(scored_keyframes[i].keyframe);
        }
    }

    ensureInitialAnchor();
    return candidates;
}

std::vector<std::shared_ptr<Frame>> Frontend::collectRelocalizationContextKeyframes(
    const std::shared_ptr<Frame>& seed_keyframe) const
{
    std::vector<std::shared_ptr<Frame>> context_keyframes;

    if (seed_keyframe == nullptr || !seed_keyframe->isKeyframe())
        return context_keyframes;

    std::unordered_set<std::size_t> selected_ids;
    selected_ids.reserve(8);

    context_keyframes.push_back(seed_keyframe);
    selected_ids.insert(seed_keyframe->getId());

    const std::vector<std::shared_ptr<Frame>> neighbors = 
        seed_keyframe->getBestCovisibilityKeyframes(4, 4); 

    for (const auto& neighbor : neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe())
            continue;

        if (selected_ids.insert(neighbor->getId()).second)
            context_keyframes.push_back(neighbor);
    }

    return context_keyframes;
}

std::vector<std::shared_ptr<MapPoint>> Frontend::collectMapPointsFromKeyframes(
    const std::vector<std::shared_ptr<Frame>>& keyframes) const
{
    std::vector<std::shared_ptr<MapPoint>> map_points;
    std::unordered_set<std::size_t> map_point_ids;
    map_point_ids.reserve(256);

    for (const auto& keyframe : keyframes)
    {
        if (keyframe == nullptr)
            continue;

        for (const auto& feature : keyframe->getFeatures())
        {
            if (feature == nullptr)
                continue;

            appendUniqueMapPoint(feature->getMapPoint(), map_points, map_point_ids);
        }
    }

    return map_points;
}

std::vector<std::shared_ptr<MapPoint>> Frontend::collectLocalMapPoints(
    const std::shared_ptr<Map>& map, 
    const TrackingResult& tracking_result) const
{
    std::vector<std::shared_ptr<MapPoint>> local_map_points;
    std::unordered_set<std::size_t> local_map_point_ids;

    if (map == nullptr || !tracking_result.success)
        return local_map_points;

    local_map_point_ids.reserve(256);

    for (const auto& map_point : tracking_result.inlier_map_points)
        appendUniqueMapPoint(map_point, local_map_points, local_map_point_ids);

    const std::size_t seed_map_point_num = local_map_points.size();

    const std::vector<std::shared_ptr<Frame>> local_keyframes = 
        collectLocalKeyframes(map, tracking_result);

    for (const auto& keyframe : local_keyframes)
    {
        if (keyframe == nullptr)
            continue;

        for (const auto& feature : keyframe->getFeatures())
        {
            if (feature == nullptr)
                continue;

            appendUniqueMapPoint(feature->getMapPoint(), local_map_points, local_map_point_ids);
        }
    }

    std::size_t observation_count_one = 0;
    std::size_t observation_count_two_or_more = 0;
    for (const auto& map_point : local_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::size_t observation_num = map_point->getKeyframeObservationCount();
        if (observation_num <= 1)
            observation_count_one++;
        else
            observation_count_two_or_more++;
    }

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R12 frame="
                    << (tracking_result.frame != nullptr ? tracking_result.frame->getId() : 0)
                    << " stage=local_map_points seed=" << seed_map_point_num
                    << " final=" << local_map_points.size()
                    << " added_from_keyframes="
                    << (local_map_points.size() >= seed_map_point_num
                        ? local_map_points.size() - seed_map_point_num : 0)
                    << " obs_one=" << observation_count_one
                    << " obs_two_or_more=" << observation_count_two_or_more);

    return local_map_points;
}

bool Frontend::predictCurrentPoseByMotionModel(const std::shared_ptr<Frame>& cur_frame)
{
    if (cur_frame == nullptr || last_tracked_frame_ == nullptr)
        return false;

    cv::Mat last_R_cw;
    cv::Mat last_t_cw;
    last_tracked_frame_->copyPose(last_R_cw, last_t_cw);

    if (last_R_cw.empty() || last_t_cw.empty())
        return false;

    if (has_motion_model_ && !motion_R_.empty() && !motion_t_.empty())
    {
        const cv::Mat R_pred = motion_R_ * last_R_cw;
        const cv::Mat t_pred = motion_R_ * last_t_cw + motion_t_;

        cur_frame->setPose(R_pred, t_pred);
        return true;
    }

    cur_frame->setPose(last_R_cw, last_t_cw);
    return true;
}

void Frontend::updateMotionModel(const std::shared_ptr<Frame>& prev_tracked_frame,
                                 const std::shared_ptr<Frame>& cur_frame)
{
    if (prev_tracked_frame == nullptr || cur_frame == nullptr)
    {
        resetMotionModel();
        return;
    }

    cv::Mat prev_R_cw;
    cv::Mat prev_t_cw;
    cv::Mat cur_R_cw;
    cv::Mat cur_t_cw;
    prev_tracked_frame->copyPose(prev_R_cw, prev_t_cw);
    cur_frame->copyPose(cur_R_cw, cur_t_cw);

    if (prev_R_cw.empty() || prev_t_cw.empty() ||
        cur_R_cw.empty() || cur_t_cw.empty())
    {
        resetMotionModel();
        return;
    }

    motion_R_ = cur_R_cw * prev_R_cw.t();
    motion_t_ = cur_t_cw - motion_R_ * prev_t_cw;
    has_motion_model_ = true;
}

void Frontend::refreshMapAfterPoseGraphOptimization(const std::shared_ptr<Map>& map)
{
    if (map == nullptr)
        return;

    for (const auto& map_point : map->getMapPoints())
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        map_point->updateViewStatistics(orb_extractor_.getScaleFactor(), 
                                        orb_extractor_.getLevelsNum());
    }

    for (const auto& keyframe : map->getKeyframes())
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
            keyframe->updateConnections();
    }

    for (const auto& keyframe : map->getKeyframes())
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
            map->recordCovisibilityConstraints(keyframe);
    }
}

void Frontend::resetMotionModel()
{
    motion_R_ = cv::Mat();
    motion_t_ = cv::Mat();
    has_motion_model_ = false;
}

bool Frontend::refineTrackingSeed(const PnPResult& seed_pnp_result,
                                  int min_seed_inliers,
                                  int min_local_map_inliers,
                                  const cv::Mat& predicted_R,
                                  const cv::Mat& predicted_t,
                                  const std::shared_ptr<Frame>& cur_frame,
                                  PnPResult& pnp_result,
                                  TrackingResult& tracking_result,
                                  std::vector<std::shared_ptr<MapPoint>>& local_map_points)
{
    pnp_result = {};
    tracking_result = {};
    local_map_points.clear();

    if (cur_frame == nullptr || tracker_ == nullptr || init_result_.map == nullptr)
        return false;

    auto restorePredictedPose = [&]()
    {
        if (!predicted_R.empty() && !predicted_t.empty())
            cur_frame->setPose(predicted_R, predicted_t);
    };

    if (!seed_pnp_result.success || seed_pnp_result.inlier_num < min_seed_inliers)
    {
        ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                        << " stage=seed_reject success=" << seed_pnp_result.success
                        << " candidates=" << seed_pnp_result.object_points.size()
                        << " inliers=" << seed_pnp_result.inlier_num
                        << " min_seed=" << min_seed_inliers);
        restorePredictedPose();
        return false;
    }

    const TrackingResult seed_tracking_result = 
        tracker_->buildTrackingResult(cur_frame, seed_pnp_result);

    if (!seed_tracking_result.success ||
        seed_tracking_result.inlier_map_points.size() < min_seed_inliers)
    {
        ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                        << " stage=seed_tracking_reject candidates="
                        << seed_pnp_result.object_points.size()
                        << " pnp_inliers=" << seed_pnp_result.inlier_num
                        << " tracking_inliers="
                        << seed_tracking_result.inlier_map_points.size());
        restorePredictedPose();
        return false;
    }

    cur_frame->setPose(seed_pnp_result.R, seed_pnp_result.tvec);

    std::vector<std::shared_ptr<MapPoint>> candidate_local_map_points = 
        collectLocalMapPoints(init_result_.map, seed_tracking_result);

    if (candidate_local_map_points.size() < 20)
    {
        ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                        << " stage=local_map_reject seed_tracking_inliers="
                        << seed_tracking_result.inlier_map_points.size()
                        << " local_map_points=" << candidate_local_map_points.size());
        restorePredictedPose();
        return false;
    }

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=local_map_ready seed_candidates="
                    << seed_pnp_result.object_points.size()
                    << " seed_inliers=" << seed_pnp_result.inlier_num
                    << " seed_tracking_inliers="
                    << seed_tracking_result.inlier_map_points.size()
                    << " local_map_points=" << candidate_local_map_points.size());

    const PnPResult refined_pnp_result = 
        tracker_->refinePoseWithLocalMap(seed_pnp_result, candidate_local_map_points, cur_frame);

    if (!refined_pnp_result.success)
    {
        ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                        << " stage=refined_pnp_reject local_map_points="
                        << candidate_local_map_points.size());
        restorePredictedPose();
        return false;
    }

    const TrackingResult refined_tracking_result = 
        tracker_->buildTrackingResult(cur_frame, refined_pnp_result);

    pnp_result = refined_pnp_result;
    tracking_result = refined_tracking_result;
    local_map_points = candidate_local_map_points;

    const bool accepted = isTrackingAccepted(refined_pnp_result, 
                                             refined_tracking_result, 
                                             min_local_map_inliers);   

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=refined_result candidates="
                    << refined_pnp_result.object_points.size()
                    << " pnp_inliers=" << refined_pnp_result.inlier_num
                    << " tracking_inliers="
                    << refined_tracking_result.inlier_map_points.size()
                    << " reproj=" << refined_pnp_result.optimized_reproj_error
                    << " accepted=" << (accepted ? 1 : 0)
                    << " required=" << min_local_map_inliers);
                                             
    if (!accepted)
        restorePredictedPose();

    return accepted;
}

bool Frontend::trackCurrentFrame(const std::shared_ptr<Frame>& cur_frame,
                                 PnPResult& pnp_result,
                                 TrackingResult& tracking_result,
                                 std::vector<std::shared_ptr<MapPoint>>& local_map_points)
{
    constexpr int kMinMotionModelInliers = 10;

    pnp_result = {};
    tracking_result = {};
    local_map_points.clear();

    if (cur_frame == nullptr || init_result_.map == nullptr || 
        last_tracked_frame_ == nullptr || tracker_ == nullptr)
        return false;

    const auto tracking_start = std::chrono::steady_clock::now();
    auto emitTiming = [&](const char* path,
                          double prediction_ms,
                          double seed_ms,
                          double refine_ms)
    {
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tracking_start).count();
        ROS_INFO_STREAM_THROTTLE(1.0,
            "P2-EUROC-PERF-R33 tracking_path=" << path
            << " prediction_ms=" << prediction_ms
            << " seed_ms=" << seed_ms
            << " refine_ms=" << refine_ms
            << " total_ms=" << total_ms);
    };

    const auto prediction_start = std::chrono::steady_clock::now();
    if (!predictCurrentPoseByMotionModel(cur_frame))
    {
        emitTiming("prediction_failed", 0.0, 0.0, 0.0);
        return false;
    }
    const double prediction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prediction_start).count();

    cv::Mat predicted_R;
    cv::Mat predicted_t;
    cur_frame->copyPose(predicted_R, predicted_t);

    // Normal tracking and relocalization have separate acceptance contracts.
    // A post-recovery guard prevents immediate keyframe churn, but must not
    // raise the normal tracking inlier requirement to the relocalization one.
    const int required_local_inliers = min_tracking_inliers_;

    const auto motion_seed_start = std::chrono::steady_clock::now();
    const PnPResult motion_pnp_result = 
        tracker_->trackFrameByMotionModel(last_tracked_frame_, cur_frame);
    const double motion_seed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - motion_seed_start).count();

    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=motion_seed candidates="
                    << motion_pnp_result.object_points.size()
                    << " inliers=" << motion_pnp_result.inlier_num
                    << " success=" << (motion_pnp_result.success ? 1 : 0));

    if (motion_pnp_result.success &&
        motion_pnp_result.inlier_num >= kMinMotionModelInliers)
    {
        const auto motion_refine_start = std::chrono::steady_clock::now();
        if (refineTrackingSeed(motion_pnp_result,
                                kMinMotionModelInliers,
                                required_local_inliers,
                                predicted_R,
                                predicted_t,
                                cur_frame,
                                pnp_result,
                                tracking_result,
                                local_map_points))
        {
            const double motion_refine_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - motion_refine_start).count();
            emitTiming("motion", prediction_ms, motion_seed_ms, motion_refine_ms);
            return true;
        }
    }

    const auto reference_frame_seed_start = std::chrono::steady_clock::now();
    const PnPResult reference_frame_pnp_result =
        tracker_->trackFrameByReferenceFrame(last_tracked_frame_, cur_frame);
    const double reference_frame_seed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - reference_frame_seed_start).count();
    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=reference_frame_seed candidates="
                    << reference_frame_pnp_result.object_points.size()
                    << " inliers=" << reference_frame_pnp_result.inlier_num
                    << " success=" << (reference_frame_pnp_result.success ? 1 : 0));
    const auto reference_frame_refine_start = std::chrono::steady_clock::now();
    if (refineTrackingSeed(reference_frame_pnp_result,
                            kMinMotionModelInliers,
                            required_local_inliers,
                            predicted_R,
                            predicted_t,
                            cur_frame,
                            pnp_result,
                            tracking_result,
                            local_map_points))
    {
        const double reference_frame_refine_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - reference_frame_refine_start).count();
        emitTiming("reference_frame", prediction_ms, reference_frame_seed_ms,
                   reference_frame_refine_ms);
        return true;
    }

    std::shared_ptr<Frame> reference_keyframe = tracking_reference_keyframe_.lock();

    if (reference_keyframe == nullptr || !reference_keyframe->isKeyframe())
        reference_keyframe = init_result_.map->copyLastKeyframe();

    if (reference_keyframe == nullptr || !reference_keyframe->isKeyframe())
    {
        emitTiming("reference_keyframe_missing", prediction_ms,
                   reference_frame_seed_ms, 0.0);
        return false;
    }

    if (bow_vocabulary_ != nullptr && !cur_frame->hasBoW())
        cur_frame->computeBoW(bow_vocabulary_);

    const auto bow_seed_start = std::chrono::steady_clock::now();
    const PnPResult reference_pnp_result = 
        tracker_->trackFrameByBoWKeyframe(reference_keyframe, cur_frame);
    const double bow_seed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - bow_seed_start).count();
    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=bow_seed ref_keyframe=" << reference_keyframe->getId()
                    << " candidates=" << reference_pnp_result.object_points.size()
                    << " inliers=" << reference_pnp_result.inlier_num
                    << " success=" << (reference_pnp_result.success ? 1 : 0));
    const auto bow_refine_start = std::chrono::steady_clock::now();
    if (refineTrackingSeed(reference_pnp_result,
                            kMinMotionModelInliers,
                            required_local_inliers,
                            predicted_R,
                            predicted_t,
                            cur_frame,
                            pnp_result,
                            tracking_result,
                            local_map_points))
    {
        const double bow_refine_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bow_refine_start).count();
        emitTiming("bow", prediction_ms, bow_seed_ms, bow_refine_ms);
        return true;
    }

    // The reference-keyframe path can fail when descriptors are temporarily
    // weak. A descriptor PnP over the active map is the final tracking-stage
    // fallback; relocalization remains the separate recovery path.
    const auto map_seed_start = std::chrono::steady_clock::now();
    const PnPResult map_pnp_result =
        tracker_->trackFrameByMap(init_result_.map, cur_frame, true);
    const double map_seed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - map_seed_start).count();
    ROS_DEBUG_STREAM("P2-EUROC-DEBUG-R18 frame=" << cur_frame->getId()
                    << " stage=map_seed candidates=" << map_pnp_result.object_points.size()
                    << " inliers=" << map_pnp_result.inlier_num
                    << " success=" << (map_pnp_result.success ? 1 : 0));
    const auto map_refine_start = std::chrono::steady_clock::now();
    const bool map_refined = refineTrackingSeed(map_pnp_result,
                                                 kMinMotionModelInliers,
                                                 required_local_inliers,
                                                 predicted_R,
                                                 predicted_t,
                                                 cur_frame,
                                                 pnp_result,
                                                 tracking_result,
                                                 local_map_points);
    const double map_refine_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - map_refine_start).count();

    emitTiming("map", prediction_ms, map_seed_ms, map_refine_ms);

    return map_refined;
}

bool Frontend::tryRelocalization(const std::shared_ptr<Frame>& cur_frame,
                                 PnPResult& pnp_result,
                                 TrackingResult& tracking_result) const
{
    pnp_result = {};
    tracking_result = {};

    if (cur_frame == nullptr || tracker_ == nullptr ||
        !init_result_.success || init_result_.map == nullptr)
    {
        return false;
    }

    bool found_valid_candidate = false;
    double best_score = 0.0;

    cv::Mat backup_R;
    cv::Mat backup_t;
    cur_frame->copyPose(backup_R, backup_t);

    const std::vector<std::shared_ptr<Frame>> candidate_keyframes = 
        collectRelocalizationCandidates(init_result_.map, cur_frame);

    for (const auto& keyframe : candidate_keyframes)
    {
        if (keyframe == nullptr)
            continue;

        PnPResult seed_pnp_result = 
            tracker_->trackFrameByBoWKeyframe(keyframe, cur_frame);

        if (!isRelocalizationSeedAccepted(seed_pnp_result))
            continue;

        cur_frame->setPose(seed_pnp_result.R, seed_pnp_result.tvec);

        const std::vector<std::shared_ptr<Frame>> context_keyframes = 
            collectRelocalizationContextKeyframes(keyframe);

        const std::vector<std::shared_ptr<MapPoint>> candidate_map_points =
            collectMapPointsFromKeyframes(context_keyframes);

        if (candidate_map_points.size() < 20)
            continue;

        PnPResult candidate_pnp_result = 
            tracker_->trackFrameByProjectionOnly(candidate_map_points, cur_frame, 1.0f, false);
        if (candidate_pnp_result.success)
            cur_frame->setPose(candidate_pnp_result.R, candidate_pnp_result.tvec);

        if (!isRelocalizationStageAccepted(candidate_pnp_result,
                                           min_recovery_expand_inliers_,
                                           min_recovery_expand_inlier_ratio_,
                                           max_recovery_expand_reproj_error_))
        {
            const PnPResult expanded_pnp_result = 
                tracker_->trackFrameByProjectionOnly(candidate_map_points, cur_frame, 2.0f, false);

            if (expanded_pnp_result.success && 
                isRelocalizationStageAccepted(expanded_pnp_result,
                                           min_recovery_expand_inliers_,
                                           min_recovery_expand_inlier_ratio_,
                                           max_recovery_expand_reproj_error_) &&
                (!candidate_pnp_result.success || 
                 expanded_pnp_result.inlier_num > candidate_pnp_result.inlier_num))
            {
                candidate_pnp_result = expanded_pnp_result;
                cur_frame->setPose(candidate_pnp_result.R, candidate_pnp_result.tvec);
            }
        }

        if (!candidate_pnp_result.success)
            continue;

        TrackingResult candidate_tracking_result =
            tracker_->buildTrackingResult(cur_frame, candidate_pnp_result);
        if (!candidate_tracking_result.success)
            continue;

        if (!isRelocalizationAccepted(candidate_pnp_result, candidate_tracking_result))
            continue;

        const double candidate_score = 
            computeRelocalizationScore(candidate_pnp_result, candidate_tracking_result);

        if (!found_valid_candidate || candidate_score > best_score)
        {
            best_score = candidate_score;
            pnp_result = candidate_pnp_result;
            tracking_result = candidate_tracking_result;
            found_valid_candidate = true;
        }
    }

    if (found_valid_candidate)
        return true;

    if (!backup_R.empty() && !backup_t.empty())
        cur_frame->setPose(backup_R, backup_t);
        
    PnPResult fallback_pnp_result =
        tracker_->trackFrameByMap(init_result_.map, cur_frame, true);
    if (!fallback_pnp_result.success)
        return false;

    TrackingResult fallback_tracking_result = 
        tracker_->buildTrackingResult(cur_frame, fallback_pnp_result);
    if (!fallback_tracking_result.success)
        return false;

    if (!isRelocalizationAccepted(fallback_pnp_result, fallback_tracking_result))
        return false;

    pnp_result = fallback_pnp_result;
    tracking_result = fallback_tracking_result;
    return true;
}

void Frontend::consumeLocalMappingResult(const LocalMappingOutput& output)
{
    const std::shared_ptr<Map> map = output.input.map;
    const std::shared_ptr<Frame> cur_keyframe = output.input.cur_keyframe;
    if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return;

    if (init_result_.map == nullptr || map != init_result_.map)
        return;

        // Keep the existing tracking reference when it still belongs to the
        // active map. A newly inserted keyframe may contain only the weak
        // inlier set that triggered insertion and is not yet a robust
        // relocalization anchor.
    const std::shared_ptr<Frame> existing_reference =
        tracking_reference_keyframe_.lock();
    if (existing_reference == nullptr || !existing_reference->isKeyframe())
        tracking_reference_keyframe_ = cur_keyframe;

    ROS_INFO_STREAM("Inserted new keyframe: " << cur_keyframe->getId()
                        << ", keyframe num: " << output.result.active_keyframe_num
                        << ", new map points: " << output.result.new_map_point_num
                        << ", culled map points: " << output.result.culled_map_point_num
                        << ", culled keyframes: " << output.result.culled_keyframe_num
                        << ", fusion called: "
                        << (output.result.fusion_called ? "true" : "false")
                        << ", keyframe cull called: "
                        << (output.result.keyframe_cull_called ? "true" : "false")
                        << ", convergence yielded: "
                        << (output.result.convergence_skipped_for_pending_keyframe
                            ? "true" : "false")
                        << ", local ba called: "
                        << (output.result.local_ba_called ? "true" : "false")
                        << ", local ba solver success: "
                        << (output.result.local_ba_solver_success ? "true" : "false")
                        << ", local ba success: "
                        << (output.result.local_ba_success ? "true" : "false")
                        << ", local ba rejected: "
                        << (output.result.local_ba_rejected ? "true" : "false")
                        << ", local ba duration ms: "
                        << output.result.local_ba_duration_ms
                        << ", local ba edges: " << output.result.local_ba_edge_num
                        << ", local ba rejected edge num: "
                        << output.result.local_ba_rejected_edge_num
                        << ", local ba seed reproj error: "
                        << output.result.local_ba_seed_reproj_error
                        << ", local ba candidate seed reproj error: "
                        << output.result.local_ba_candidate_seed_reproj_error
                        << ", local ba rejection reason: "
                        << (output.result.local_ba_rejection_reason.empty()
                            ? "none" : output.result.local_ba_rejection_reason)
                        << ", total map points: " << output.result.active_map_point_num
                        << ", keyframe database registered: "
                        << (output.result.keyframe_database_registered ? "true" : "false")
                        << ", loop keyframe queued: "
                        << (output.result.loop_keyframe_queued ? "true" : "false"));

        // Current project adaptation: normal frames lack ORB-SLAM2's stored
        // reference-keyframe-relative pose, so a committed Local BA can make
        // the cached normal-frame velocity stale.
    if (output.result.local_ba_success)
        resetMotionModel();
}

void Frontend::drainLocalMappingResults()
{
    if (local_mapper_ == nullptr)
        return;

    LocalMappingOutput output;
    while (local_mapper_->tryPopFinishedResult(output))
        consumeLocalMappingResult(output);
}

bool Frontend::submitKeyframeWithCommitBarrier(
    const std::shared_ptr<Map>& map,
    const std::shared_ptr<Frame>& cur_frame,
    const PnPResult& tracking_seed,
    const TrackingResult& tracking_result)
{
    if (local_mapper_ == nullptr || map == nullptr || cur_frame == nullptr)
        return false;

    const auto barrier_start = std::chrono::steady_clock::now();
    LocalMappingOutput completed_output;
    while (!local_mapper_->acceptKeyframe())
    {
        if (local_mapper_->waitPopFinishedResult(completed_output))
        {
            consumeLocalMappingResult(completed_output);
            continue;
        }

        if (local_mapper_->finishRequested())
            return false;
    }

    std::shared_ptr<Frame> ref_keyframe;
    bool keyframe_queued = false;
    std::size_t keyframe_num_before_mapping = 0;
    while (!keyframe_queued)
    {
        while (!local_mapper_->acceptKeyframe())
        {
            if (local_mapper_->waitPopFinishedResult(completed_output))
            {
                consumeLocalMappingResult(completed_output);
                continue;
            }
            if (local_mapper_->finishRequested())
                return false;
        }

        std::lock_guard<std::mutex> map_lock(map->getMutex());
        if (init_result_.map != map)
            return false;

        ref_keyframe = map->getLastKeyframe();
        if (ref_keyframe == nullptr || !ref_keyframe->isKeyframe())
            return false;

        keyframe_num_before_mapping = map->getKeyframeNum();
        cur_frame->setKeyframe(true);
        keyframe_queued = local_mapper_->insertKeyframe(
            LocalMappingInput{map, ref_keyframe, cur_frame, tracking_seed});
        if (!keyframe_queued)
            cur_frame->setKeyframe(false);
    }

    keyframe_commit_barrier_num_++;
    ROS_INFO_STREAM("Queued new keyframe: " << cur_frame->getId()
                    << ", ref keyframe: " << ref_keyframe->getId()
                    << ", tracking inliers: " << tracking_result.inlier_map_points.size()
                    << ", map keyframe num(before local mapping): "
                    << keyframe_num_before_mapping);

    if (!local_mapper_->waitPopFinishedResult(completed_output))
        return false;
    consumeLocalMappingResult(completed_output);

    const double barrier_wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - barrier_start).count();
    ROS_INFO_STREAM("P2-KITTI-SCHED-R136 commit_barrier frame=" << cur_frame->getId()
                    << " wait_ms=" << barrier_wait_ms);
    return true;
}

void Frontend::drainLoopClosingResults()
{
    if (loop_closer_ == nullptr)
        return;

    LoopClosingOutput output;
    while (loop_closer_->tryPopFinishedResult(output))
    {
        const std::shared_ptr<Map> map = output.input.map;
        const std::shared_ptr<Frame> cur_keyframe = output.input.cur_keyframe;

        if (map == nullptr || cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
            continue;

        if (init_result_.map == nullptr || map != init_result_.map)
            continue;

        if (output.graph_optimized)
        {
            std::lock_guard<std::mutex> map_lock(map->getMutex());
            refreshMapAfterPoseGraphOptimization(map);
            resetMotionModel();
        }

        if (output.verification_result.success &&
            output.verification_result.candidate_keyframe != nullptr)
        {
            ROS_INFO_STREAM("Loop verified for keyframe " << cur_keyframe->getId()
                            << ", candidate: "
                            << output.verification_result.candidate_keyframe->getId()
                            << ", raw matches: " << output.verification_result.raw_match_num
                            << ", pnp inliers: " << output.verification_result.pnp_inlier_num
                            << ", sim3 inliers: "
                            << output.verification_result.sim3_result.inlier_num
                            << ", sim3 ratio: "
                            << output.verification_result.sim3_result.inlier_ratio
                            << ", sim3 mean error: "
                            << output.verification_result.sim3_result.mean_error
                            << ", sim3 scale: "
                            << output.verification_result.sim3_result.scale
                            << ", fused loop map points: "
                            << output.correction_result.fused_map_point_num
                            << ", updated loop keyframes: "
                            << output.correction_result.updated_keyframe_num
                            << ", graph optimized: "
                            << (output.graph_optimized ? "true" : "false"));
        }
    }
}

void Frontend::acceptTrackingResult(const std::shared_ptr<Frame>& cur_frame,
                                    const PnPResult& pnp_result,
                                    const TrackingResult& tracking_result,
                                    bool recovered_from_tmp_lost)
{
    const std::shared_ptr<Frame> prev_tracked_frame = last_tracked_frame_;

    cur_frame->setPose(pnp_result.R, pnp_result.tvec);

    if (recovered_from_tmp_lost)
    {
        resetMotionModel();

        relocalization_cooldown_remaining_ = 
            relocalization_cooldown_frames_ > 0 ? relocalization_cooldown_frames_ : 0;

        relocalization_guard_remaining_ = 
            relocalization_guard_frames_ > 0 ? relocalization_guard_frames_ : 0;
    }

    last_tracked_frame_ = cur_frame;
    tracking_result_ = tracking_result;
    consecutive_lost_num_ = 0;
    status_ = FrontendStatus::TRACKING;

    const std::shared_ptr<Map> map = init_result_.map;
    bool keyframe_requested = false;
    if (map != nullptr)
    {
        const auto map_lock_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> map_lock(map->getMutex());
        const double map_lock_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - map_lock_start).count();
        ROS_INFO_STREAM_THROTTLE(1.0,
            "P2-EUROC-PERF-R21 frontend_tracking_map_lock_wait_ms="
            << map_lock_wait_ms);

        // Current-project concurrency repair: Local BA may publish a keyframe
        // pose on its worker thread. Capture both tracking poses while map
        // publication is excluded so the velocity model cannot mix BA epochs.
        if (!recovered_from_tmp_lost)
            updateMotionModel(prev_tracked_frame, cur_frame);

        updateTrackObservations(tracking_result_);
        updateTrackingReferenceKeyframe(map, tracking_result_);

        if (shouldInsertKeyframe(map, cur_frame, tracking_result_))
        {
            keyframe_requested = true;
        }
        else if (relocalization_cooldown_remaining_ > 0) 
        {
            ROS_INFO_STREAM_THROTTLE(1.0,
                                    "Recovery cooldown active. Skip keyframe insertion. remaining=" 
                                    << relocalization_cooldown_remaining_);                       
        }
    }

    if (keyframe_requested &&
        !submitKeyframeWithCommitBarrier(map, cur_frame, pnp_result, tracking_result_))
    {
        ROS_WARN_STREAM("Keyframe commit barrier did not submit frame " << cur_frame->getId());
    }
    else if (!recovered_from_tmp_lost)
    {
        updateMotionModel(prev_tracked_frame, cur_frame);
    }

    if (!recovered_from_tmp_lost && relocalization_cooldown_remaining_ > 0)
        relocalization_cooldown_remaining_--;

    if (!recovered_from_tmp_lost && relocalization_guard_remaining_ > 0)
        relocalization_guard_remaining_--;

    publishCurrentPose(cur_frame);

    last_frame_ = cur_frame;

    ROS_INFO_STREAM_THROTTLE(1.0,
                            "Tracking success. candidates: " <<
                            pnp_result.object_points.size()
                            << ", inliers: " <<
                            tracking_result.inlier_map_points.size()
                            << ", pnp_optimized: " << (pnp_result.optimized ? "true" :
                            "false")
                            << ", pnp_reproj_error: " <<
                            pnp_result.optimized_reproj_error
                            << ", recovered_from_tmp_lost: "
                            << (recovered_from_tmp_lost ? "true" : "false"));

}

bool Frontend::isTrackingAccepted(const PnPResult& pnp_result,
                                  const TrackingResult& tracking_result,
                                  int min_inlier_num) const
{
    if (!pnp_result.success || !tracking_result.success)
        return false;

    if (pnp_result.inlier_num < min_inlier_num)
        return false;

    if (static_cast<int>(tracking_result.inlier_map_points.size()) < min_inlier_num)
        return false;

    const double reproj_error = 
        pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;

    return std::isfinite(reproj_error) &&
           reproj_error > 0.0 && 
           reproj_error <= max_tracking_reproj_error_;
}

bool Frontend::isRelocalizationSeedAccepted(const PnPResult& pnp_result) const
{
    return isRelocalizationStageAccepted(pnp_result, 
                                         min_recovery_seed_inliers_, 
                                         min_recovery_seed_inlier_ratio_, 
                                         max_recovery_seed_reproj_error_);
}

bool Frontend::isRelocalizationStageAccepted(const PnPResult& pnp_result,
                                             int min_inlier_num,
                                             double min_inlier_ratio,
                                             double max_reproj_error) const
{
    if (!pnp_result.success)
        return false;

    if (pnp_result.inlier_num < min_inlier_num)
        return false;

    const std::size_t candidate_num = pnp_result.object_points.size();
    if (candidate_num == 0)
        return false;

    const double inlier_ratio =
        static_cast<double>(pnp_result.inlier_num) / candidate_num;

    if (inlier_ratio < min_inlier_ratio)
        return false;

    const double reproj_error =
        pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;

    if (reproj_error <= 0.0 || reproj_error > max_reproj_error)
        return false;

    return true;
}

bool Frontend::isRelocalizationAccepted(const PnPResult& pnp_result,
                                        const TrackingResult& tracking_result) const
{
    const int required_relocalization_inliers = 
        std::max(min_recovery_inliers_, min_relocalization_inliers_);

    if (!isTrackingAccepted(pnp_result, tracking_result, required_relocalization_inliers))
        return false;

    const std::size_t candidate_num = pnp_result.object_points.size();
    if (candidate_num == 0)
        return false;

    const double pnp_inlier_ratio = 
        static_cast<double>(pnp_result.inlier_num) / candidate_num;
    
    const double tracking_inlier_ratio = 
        static_cast<double>(tracking_result.inlier_map_points.size()) / candidate_num;

    const double reproj_error = 
        pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;

    if (pnp_inlier_ratio < min_recovery_inlier_ratio_)
        return false;

    if (tracking_inlier_ratio < min_recovery_inlier_ratio_)
        return false;

    if (reproj_error <= 0.0 || reproj_error > max_recovery_reproj_error_)
        return false;

    return true;
}

double Frontend::computeRelocalizationScore(const PnPResult& pnp_result,
                                            const TrackingResult& tracking_result) const
{
    const std::size_t candidate_num = pnp_result.object_points.size();
    if (candidate_num == 0)
        return 0.0;

    const double pnp_inlier_ratio = 
        static_cast<double>(pnp_result.inlier_num) / candidate_num;

    const double tracking_inlier_ratio = 
        static_cast<double>(tracking_result.inlier_map_points.size()) / candidate_num;

    const double reproj_error =
        (pnp_result.optimized && pnp_result.optimized_reproj_error > 0.0)
            ? pnp_result.optimized_reproj_error
            : pnp_result.ransac_reproj_error;

    if (reproj_error <= 0.0)
        return 0.0;

    const double support_score = 
        static_cast<double>(pnp_result.inlier_num) + tracking_result.inlier_map_points.size();

    const double ratio_score = 0.5 * (pnp_inlier_ratio + tracking_inlier_ratio);

    return support_score * (1.0 + ratio_score) / reproj_error;
}

void Frontend::resetToInitializing(const std::shared_ptr<Frame>& seed_frame)
{
    init_result_ = {};
    tracking_result_ = {};
    tracking_reference_keyframe_.reset();
    initialization_previous_matched_.clear();
    initialization_match_reference_id_ = std::numeric_limits<std::size_t>::max();
    consecutive_lost_num_ = 0;
    init_geometry_failures_num_ = 0;
    relocalization_cooldown_remaining_ = 0;
    relocalization_guard_remaining_ = 0;
    status_ = FrontendStatus::INITING;

    last_frame_ = seed_frame;
    last_tracked_frame_.reset();
    resetMotionModel();

    if (keyframe_database_ != nullptr)
        keyframe_database_->clear();

    if (!openTrajectoryOutputFile(true))
        ROS_ERROR("Failed to open trajectory output file for writing.");
}

void Frontend::registerKeyframeInDatabase(const std::shared_ptr<Frame>& keyframe)
{
    if (keyframe == nullptr || !keyframe->isKeyframe() || keyframe_database_ == nullptr)
        return;

    if (!keyframe->hasBoW() && bow_vocabulary_ != nullptr)
        keyframe->computeBoW(bow_vocabulary_);

    keyframe_database_->addKeyframe(keyframe);
}

void Frontend::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
    if (!use_camera_info_ || calibration_locked_ || camera_ == nullptr)
        return;

    if (!camera_->setCameraInfo(*msg))
    {
        ROS_ERROR_THROTTLE(1.0, "Rejected CameraInfo calibration.");
        return;
    }

    camera_info_received_ = true;

    ROS_INFO_STREAM("Receive CameraInfo calibration: "
                    << msg->width << "x" << msg->height
                    << ", distortion model: " 
                    << msg->distortion_model);
}

bool Frontend::lockCalibrationForImage(const cv::Size& image_size)
{
    if (camera_ == nullptr)
        return false;

    if (use_camera_info_ &&
        camera_info_required_ &&
        !camera_info_received_)
    {
        ROS_WARN_THROTTLE(1.0, "Waiting for CameraInfo calibration.");
        return false;
    }

    if (!camera_->isValid())
    {
        ROS_WARN_THROTTLE(1.0, "Camera calibration is not valid.");
        return false;
    }

    if (image_size.width != camera_->getImageWidth() ||
        image_size.height != camera_->getImageHeight())
    {
        ROS_WARN_THROTTLE(1.0, "Image size does not match camera calibration.");
        return false;
    }

    if (!calibration_locked_)
    {
        calibration_locked_ = true;
        ROS_INFO_STREAM("Camera calibration locked. source: "
                        << (use_camera_info_ ? "CameraInfo" : "YAML"));
    }

    return true;
}

void Frontend::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    const ImageCallbackTiming callback_timing;
    struct ProcessedImageAckGuard
    {
        Frontend* frontend;
        ~ProcessedImageAckGuard()
        {
            frontend->publishProcessedImageAck();
        }
    } processed_image_ack{this};

    cv_bridge::CvImageConstPtr cv_ptr;
    cv::Mat gray;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        gray = cv_ptr->image.clone();
    }
    catch (const cv_bridge::Exception&)
    {
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    processImage(gray, msg->header.stamp.toSec());
}

void Frontend::processImage(const cv::Mat& image, double timestamp)
{
    if (image.empty() || image.channels() != 1)
    {
        ROS_ERROR("Direct frontend input must be a non-empty mono image.");
        return;
    }

    cv::Mat gray = image;

    if (!lockCalibrationForImage(gray.size()))
        return;

    cv::Mat undistorted_gray;
    if (!camera_->undistrortImage(gray, undistorted_gray))
    {
        ROS_WARN_THROTTLE(1.0, "Failed to undistort current image.");
        return;
    }

    gray = undistorted_gray;

    std::shared_ptr<Frame> cur_frame = buildFrame(next_frame_id_, timestamp, gray);
    next_frame_id_++;

    if (cur_frame == nullptr || !cur_frame->hasFeatures())
    {
        ROS_WARN("Failed to build current frame.");
        return;
    }

    drainLocalMappingResults();
    drainLoopClosingResults();

    if (status_ == FrontendStatus::INITING)
    {
        if (last_frame_ == nullptr)
        {
            last_frame_ = cur_frame;
            ROS_INFO("Receive first frame, waiting for next frame to initialize.");
            return;
        }

        bool should_reanchor = false;
        const bool init_success = initializeFromRefAndCur(last_frame_, cur_frame, should_reanchor);

        if (!init_success)
        {
            if (should_reanchor)
            {
                if (isQualifiedInitializationReference(cur_frame))
                {
                    last_frame_ = cur_frame;
                }
                else 
                {
                    last_frame_.reset();

                    ROS_WARN_STREAM("Initialization reanchor requested, but current frame is not a qualified "
                                    << "initialization reference. wait for a new seed frame. "
                                    << "feature_num=" << cur_frame->getFeatureNum());
                }
            }

            ROS_WARN("Initialization failed, waiting for next frame.");
            return;
        }

        last_frame_ = cur_frame;
        consecutive_lost_num_ = 0;
        return;
    }

    if (status_ == FrontendStatus::TRACKING || status_ == FrontendStatus::TMP_LOST)
    {
        if (!init_result_.success || init_result_.map == nullptr)
        {
            ROS_WARN("Map is not ready in tracking state.");
            status_ = FrontendStatus::LOST;
            last_frame_ = cur_frame;
            return;
        }

        PnPResult pnp_result;
        TrackingResult new_tracking_result;
        std::vector<std::shared_ptr<MapPoint>> local_map_points;

        PnPResult relocal_pnp_result;
        TrackingResult relocal_tracking_result;
        
        const int required_inliers = 
            (status_ == FrontendStatus::TRACKING)
                ? min_tracking_inliers_
                : std::max(min_recovery_inliers_, min_relocalization_inliers_);

        bool track_success = false;
        bool tracking_accepted = false;
        bool relocal_success = false;

        const auto tracking_section_start = std::chrono::steady_clock::now();
        double tracking_map_lock_wait_ms = 0.0;
        {
            const auto tracking_map_lock_start = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> map_lock(init_result_.map->getMutex());
            tracking_map_lock_wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tracking_map_lock_start).count();
            track_success = 
                trackCurrentFrame(cur_frame, pnp_result, new_tracking_result, local_map_points);

            tracking_accepted = 
                track_success && isTrackingAccepted(pnp_result, new_tracking_result, required_inliers);
        }

        // Relocalization obtains its own map snapshot through
        // collectRelocalizationCandidates()/Map::copyKeyframes().  Do not
        // call it while the tracking transaction owns Map::mutex_: the
        // snapshot helper would otherwise self-deadlock on the same mutex.
        if (!tracking_accepted)
        {
            relocal_success =
                tryRelocalization(cur_frame, relocal_pnp_result, relocal_tracking_result);
        }

        const double tracking_section_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tracking_section_start).count();
        ROS_INFO_STREAM_THROTTLE(1.0,
            "P2-EUROC-PERF-R34 tracking_section_ms=" << tracking_section_ms
            << " map_lock_wait_ms=" << tracking_map_lock_wait_ms
            << " tracking_work_ms=" << (tracking_section_ms - tracking_map_lock_wait_ms));

        if (!tracking_accepted)
        {
            if (relocal_success)
            {
                acceptTrackingResult(cur_frame, relocal_pnp_result, relocal_tracking_result, true);
                return;
            }

            consecutive_lost_num_++;
            last_frame_ = cur_frame;

            if (consecutive_lost_num_ >= tmp_lost_max_frames_)
            {
                status_ = FrontendStatus::LOST;
                ROS_WARN_STREAM("Tracking and relocalization failed. Enter LOST."
                                << " lost_count=" << consecutive_lost_num_
                                << ", track_pnp_inliers=" << pnp_result.inlier_num
                                << ", track_inliers=" <<
                                new_tracking_result.inlier_map_points.size());
            }
            else 
            {
                status_ = FrontendStatus::TMP_LOST;
                ROS_WARN_STREAM("Tracking failed, relocalization failed. Enter TMP_LOST."
                                << " lost_count=" << consecutive_lost_num_
                                << "/" << tmp_lost_max_frames_
                                << ", track_pnp_inliers=" << pnp_result.inlier_num
                                << ", track_inliers=" <<
                                new_tracking_result.inlier_map_points.size());
            }

            return;
        }

        const bool recovered_from_tmp_lost = (status_ == FrontendStatus::TMP_LOST);
        acceptTrackingResult(cur_frame, 
                             pnp_result, 
                             new_tracking_result, 
                             recovered_from_tmp_lost);

        return; 
    }

    if (status_ == FrontendStatus::LOST)
    {
        if (init_result_.map == nullptr)
        {
            ROS_WARN("Frontend is LOST without a valid map. Reset current session and return to INITING");
            resetToInitializing(cur_frame);
            return;
        }

        PnPResult relocal_pnp_result;
        TrackingResult relocal_tracking_result;
        bool relocal_success = false;

        // tryRelocalization() takes a consistent map snapshot internally;
        // keep it outside the caller's map transaction to avoid recursive
        // acquisition of Map::mutex_.
        relocal_success =
            tryRelocalization(cur_frame, relocal_pnp_result, relocal_tracking_result);

        if (relocal_success)
        {
            ROS_INFO("Relocalization succeeded from LOST state.");
            acceptTrackingResult(cur_frame, 
                                 relocal_pnp_result, 
                                 relocal_tracking_result, 
                                 true);

            return;
        }

        ROS_WARN("Frontend is LOST. Reset current session and return to INITING.");
        resetToInitializing(cur_frame);
        return;
    }

    last_frame_ = cur_frame;
}

void Frontend::publishProcessedImageAck()
{
    if (!processed_image_pub_)
        return;

    std_msgs::UInt64 message;
    message.data = ++processed_image_count_;
    processed_image_pub_.publish(message);
}

void Frontend::publishCurrentPose(const std::shared_ptr<Frame>& frame)
{
    if (frame == nullptr)
        return;

    const cv::Mat R_wc = frame->getRwc();
    if (R_wc.rows != 3 || R_wc.cols != 3)
        return;

    Eigen::Matrix3d rotation;

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            rotation(r, c) = R_wc.at<double>(r, c);
        }
    }

    Eigen::Quaterniond q(rotation);
    q.normalize();

    const cv::Point3d position = frame->getCameraCenter();

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(frame->getTimestamp());
    pose_msg.header.frame_id = "map";

    pose_msg.pose.position.x = position.x;
    pose_msg.pose.position.y = position.y;
    pose_msg.pose.position.z = position.z;

    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();

    pose_pub_.publish(pose_msg);

    if (!trajectory_output_.is_open())
        return;

    trajectory_output_ << std::fixed << std::setprecision(9);

    if (trajectory_format_ == "tum")
    {
        trajectory_output_ << frame->getTimestamp() << " "
                           << position.x << " " << position.y << " " << position.z << " "
                           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    else
    {
        trajectory_output_ 
            << rotation(0, 0) << " " << rotation(0, 1) << " " << rotation(0, 2) << " " 
            << position.x << " "
            << rotation(1, 0) << " " << rotation(1, 1) << " " << rotation(1, 2) << " "
            << position.y << " "
            << rotation(2, 0) << " " << rotation(2, 1) << " " << rotation(2, 2) << " "
            << position.z << "\n";
    }

    // A replay acknowledgement means this accepted pose is durable.  Without
    // this flush, a forced shutdown can silently discard the buffered tail.
    trajectory_output_.flush();
    if (!trajectory_output_)
        ROS_ERROR("Failed to persist accepted trajectory pose.");

}

} // namespace mini_orb_slam
