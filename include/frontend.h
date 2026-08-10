#ifndef MINI_ORB_SLAM_INCLUDE_FRONTEND_H_
#define MINI_ORB_SLAM_INCLUDE_FRONTEND_H_

#include <memory>
#include <string>
#include <fstream>
#include <limits>
#include <cstdint>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv4/opencv2/core.hpp>
#include <unordered_set>
#include <sensor_msgs/CameraInfo.h>

#include "camera.h"
#include "common.h"
#include "frame.h"
#include "initializer.h"
#include "matcher.h"
#include "orb_extractor.h"
#include "tracker.h"
#include "local_mapper.h"
#include "pose_optimizer.h"
#include "bow_vocabulary.h"
#include "keyframe_database.h"
#include "loop_closer.h"

namespace mini_orb_slam
{

enum class FrontendStatus
{
    INITING = 0,
    TRACKING = 1,
    TMP_LOST = 2,
    LOST = 3
};

class Frontend
{
public:
    explicit Frontend(ros::NodeHandle& nh);
    ~Frontend();

    bool init();
    // Uses the same frontend processing path as imageCallback(), but does not
    // create ROS image transport endpoints. Intended for offline benchmarks.
    bool initOffline();
    void run();
    void processImage(const cv::Mat& image, double timestamp);

private:
    bool initImpl(bool enable_ros_transport);
    bool openTrajectoryOutputFile(bool truncate);

    bool initializeFromRefAndCur(const std::shared_ptr<Frame>& ref_frame, 
                                 const std::shared_ptr<Frame>& cur_frame,
                                 bool& should_reanchor);

    std::shared_ptr<Frame> buildFrame(std::size_t id, double timestamp, const cv::Mat& img);

    InitializationResult buildInitialMap(const std::shared_ptr<Frame>& ref_frame,
                                         const std::shared_ptr<Frame>& cur_frame,
                                         const TriangulationResult& triangulation_result);

    bool isInitialTriangulationAccepted(const TriangulationResult& triangulation_result) const;

    bool isInitialMapOptimizationAccepted(
        const InitialMapOptimizationResult& optimization_result) const;

    bool retainInitialMapInliers(InitializationResult& init_result,
                                 const InitialMapOptimizationResult& optimization_result) const;

    bool normalizeInitialMapScale(InitializationResult& init_result, 
                                  double& median_depth) const;

    void rollbackInitialMapAttempt(const std::shared_ptr<Frame>& cur_frame,
                                   const cv::Mat& cur_R_before, 
                                   const cv::Mat& cur_t_before);

    void updateTrackObservations(const TrackingResult& tracking_result);
    void updateTrackingReferenceKeyframe(const std::shared_ptr<Map>& map,
                                         const TrackingResult& tracking_result);

    double computeRotationAngleDeg(const cv::Mat& R) const;

    bool isQualifiedInitializationReference(const std::shared_ptr<Frame>& frame) const;

    bool shouldInsertKeyframe(const std::shared_ptr<Map>& map, 
                              const std::shared_ptr<Frame>& cur_frame, 
                              const TrackingResult& tracking_result);

    void appendUniqueMapPoint(const std::shared_ptr<MapPoint>& map_point, 
                              std::vector<std::shared_ptr<MapPoint>>& local_map_points,
                              std::unordered_set<std::size_t>& map_point_ids) const;

    std::vector<std::shared_ptr<Frame>> collectLocalKeyframes(
        const std::shared_ptr<Map>& map, 
        const TrackingResult& tracking_result) const;

    std::vector<std::shared_ptr<Frame>> collectRelocalizationCandidates(
        const std::shared_ptr<Map>& map, 
        const std::shared_ptr<Frame>& cur_frame) const;

    std::vector<std::shared_ptr<Frame>> collectRelocalizationContextKeyframes(
        const std::shared_ptr<Frame>& seed_keyframe) const;
    
    std::vector<std::shared_ptr<MapPoint>> collectMapPointsFromKeyframes(
        const std::vector<std::shared_ptr<Frame>>& keyframes) const;

    std::vector<std::shared_ptr<MapPoint>> collectLocalMapPoints(
        const std::shared_ptr<Map>& map, 
        const TrackingResult& tracking_result) const;

    bool predictCurrentPoseByMotionModel(const std::shared_ptr<Frame>& cur_frame);
    void updateMotionModel(const std::shared_ptr<Frame>& prev_tracked_frame,
                           const std::shared_ptr<Frame>& cur_frame);

    void refreshMapAfterPoseGraphOptimization(const std::shared_ptr<Map>& map);

    void consumeLocalMappingResult(const LocalMappingOutput& output);
    void drainLocalMappingResults();
    void drainLoopClosingResults();
    bool submitKeyframeWithCommitBarrier(const std::shared_ptr<Map>& map,
                                         const std::shared_ptr<Frame>& cur_frame,
                                         const PnPResult& tracking_seed,
                                         const TrackingResult& tracking_result);
    
    void resetMotionModel();

    bool trackCurrentFrame(const std::shared_ptr<Frame>& cur_frame,
                           PnPResult& pnp_result,
                           TrackingResult& tracking_result,
                           std::vector<std::shared_ptr<MapPoint>>& local_map_points);

    bool refineTrackingSeed(const PnPResult& seed_pnp_result,
                            int min_seed_inliers,
                            int min_local_map_inliers,
                            const cv::Mat& predicted_R,
                            const cv::Mat& predicted_t,
                            const std::shared_ptr<Frame>& cur_frame,
                            PnPResult& pnp_result,
                            TrackingResult& tracking_result,
                            std::vector<std::shared_ptr<MapPoint>>& local_map_points);

    bool tryRelocalization(const std::shared_ptr<Frame>& cur_frame,
                           PnPResult& pnp_result,
                           TrackingResult& tracking_result) const;

    void acceptTrackingResult(const std::shared_ptr<Frame>& cur_frame,
                              const PnPResult& pnp_result,
                              const TrackingResult& tracking_result,
                              bool recovered_from_tmp_lost);

    bool isTrackingAccepted(const PnPResult& pnp_result,
                            const TrackingResult& tracking_result,
                            int min_inlier_num) const;

    bool isRelocalizationSeedAccepted(const PnPResult& pnp_result) const;

    bool isRelocalizationStageAccepted(const PnPResult& pnp_result,
                                       int min_inlier_num,
                                       double min_inlier_ratio,
                                       double max_reproj_error) const;
 
    bool isRelocalizationAccepted(const PnPResult& pnp_result,
                                  const TrackingResult& tracking_result) const;

    double computeRelocalizationScore(const PnPResult& pnp_result,
                                      const TrackingResult& tracking_result) const;
                            
    void resetToInitializing(const std::shared_ptr<Frame>& seed_frame);

    void registerKeyframeInDatabase(const std::shared_ptr<Frame>& keyframe);

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);

    bool lockCalibrationForImage(const cv::Size& image_size);

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void publishProcessedImageAck();

    void publishCurrentPose(const std::shared_ptr<Frame>& frame);
    
    ros::NodeHandle& nh_;
    ros::Subscriber image_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher processed_image_pub_;
    std::ofstream trajectory_output_;

    std::string camera_topic_;
    std::string camera_info_topic_;
    std::string processed_image_topic_;
    int image_queue_size_{10};
    std::uint64_t processed_image_count_{0};

    std::string trajectory_output_path_;
    std::string trajectory_format_;

    bool use_camera_info_{false};
    bool camera_info_required_{false};
    bool camera_info_received_{false};
    bool calibration_locked_{false};

    std::size_t next_frame_id_{0};

    FrontendStatus status_{FrontendStatus::INITING};

    std::shared_ptr<Camera> camera_;
    ORBExtractor orb_extractor_;
    Matcher matcher_;
    std::unique_ptr<Initializer> initializer_;
    std::unique_ptr<Tracker> tracker_;
    std::shared_ptr<PoseOptimizer> pose_optimizer_;
    std::unique_ptr<LocalMapper> local_mapper_;
    std::unique_ptr<LoopCloser> loop_closer_;

    std::shared_ptr<Frame> last_frame_;
    std::shared_ptr<Frame> last_tracked_frame_;
    std::weak_ptr<Frame> tracking_reference_keyframe_;

    // ORB-SLAM2 logic reference: the monocular initializer retains matched
    // image positions while its reference frame remains unchanged.
    std::size_t initialization_match_reference_id_{std::numeric_limits<std::size_t>::max()};
    std::vector<cv::Point2f> initialization_previous_matched_;

    cv::Mat motion_R_;
    cv::Mat motion_t_;
    bool has_motion_model_{false};

    InitializationResult init_result_;
    TrackingResult tracking_result_;

    int consecutive_lost_num_{0};

    int min_initial_map_points_{100};
    double min_initial_good_point_ratio_{0.5};
    double max_initial_reproj_error_px_{2.0};
    double min_initial_parallax_deg_{1.0};

    int max_init_geometry_failures_per_ref_{3};
    int init_geometry_failures_num_{0};

    int min_init_reference_features_{250};
    int min_init_reference_occupied_bins_{6};

    int min_keyframe_gap_{2};
    int max_keyframe_gap_{10};
    std::size_t keyframe_decision_num_{0};
    std::size_t keyframe_busy_rejected_num_{0};
    std::size_t keyframe_commit_barrier_num_{0};
    std::size_t keyframe_force_insert_num_{0};
    std::size_t keyframe_weak_insert_num_{0};
    int tmp_lost_max_frames_{5};
    double max_tracking_reproj_error_{8.0};
    // ORB-SLAM2's TrackLocalMap contract requires at least 30 map-point
    // inliers for normal tracking. Keep this separate from the higher
    // relocalization acceptance contract.
    int min_tracking_inliers_{30};
    int min_recovery_inliers_{15};

    int min_recovery_seed_inliers_{10};
    double min_recovery_seed_inlier_ratio_{0.20};
    double max_recovery_seed_reproj_error_{12.0};

    int min_recovery_expand_inliers_{15};
    double min_recovery_expand_inlier_ratio_{0.25};
    double max_recovery_expand_reproj_error_{10.0};

    double min_recovery_inlier_ratio_{0.25};
    double max_recovery_reproj_error_{8.0};

    int relocalization_cooldown_frames_{3};
    int relocalization_cooldown_remaining_{0};

    int min_relocalization_inliers_{50};
    int relocalization_guard_frames_{10};
    int relocalization_guard_remaining_{0};

    std::shared_ptr<BoWVocabulary> bow_vocabulary_;
    std::shared_ptr<KeyframeDatabase> keyframe_database_;

    std::string vocabulary_path_;
    int max_relocalization_candidates_{5};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_FRONTEND_H_
