#ifndef MINI_ORB_SLAM_INCLUDE_LOOP_CLOSER_H_
#define MINI_ORB_SLAM_INCLUDE_LOOP_CLOSER_H_

#include <memory>
#include <unordered_set>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "common.h"
#include "keyframe_database.h"
#include "bow_vocabulary.h"
#include "frame.h"
#include "map.h"
#include "matcher.h"
#include "pose_optimizer.h"

namespace mini_orb_slam
{

class LocalMapper;

struct LoopCandidateGroup
{
    std::shared_ptr<Frame> representative;
    double bow_score{0.0};
    double group_score{0.0};
    int support_num{0};
    int consistency{0};
};

struct LoopSim3Correspondence
{
    std::shared_ptr<MapPoint> candidate_map_point;
    std::shared_ptr<MapPoint> current_map_point;

    cv::Point3d candidate_point_world;
    cv::Point3d current_point_world;

    cv::Point3d candidate_point_camera;
    cv::Point3d current_point_camera;

    int candidate_feature_idx{-1};
    int current_feature_idx{-1};
};

struct LoopSim3Result
{
    bool success{false};

    double scale{1.0};
    cv::Mat R;
    cv::Mat t;

    int inlier_num{0};
    double inlier_ratio{0.0};
    double mean_error{0.0};

    std::vector<int> inlier_indices;
};

struct LoopVerificationResult
{
    bool success{false};

    std::shared_ptr<Frame> candidate_keyframe;
    int raw_match_num{0};
    int pnp_inlier_num{0};
    double inlier_ratio{0.0};
    double reproj_error{0.0};

    PnPResult pnp_result;
    std::vector<LoopSim3Correspondence> sim3_correspondences;
    LoopSim3Result sim3_result;
};

struct LoopCorrectionResult
{
    bool success{false};

    std::size_t fused_map_point_num{0};
    std::size_t updated_keyframe_num{0};

    bool registered_loop_edge{false};
};

struct LoopClosingInput
{
    std::shared_ptr<Map> map;
    std::shared_ptr<Frame> cur_keyframe;
};

struct LoopClosingOutput
{
    LoopClosingInput input;
    LoopVerificationResult verification_result;
    LoopCorrectionResult correction_result;
    bool graph_optimized{false};
};

class LoopCloser
{
public:
    LoopCloser(const std::shared_ptr<KeyframeDatabase>& keyframe_database,
               const Matcher& matcher,
               const std::shared_ptr<PoseOptimizer>& pose_optimizer,
               LocalMapper* local_mapper,
               double scale_factor,
               int levels_num);

    ~LoopCloser();

    void start();
    void requestFinish();
    void join();

    bool insertKeyframe(const LoopClosingInput& input);
    bool tryPopFinishedResult(LoopClosingOutput& output);

    bool acceptKeyframe() const;
    bool isStopped() const;
    bool stopRequested() const;

    std::vector<std::shared_ptr<Frame>> detectLoopCandidates(
        const std::shared_ptr<Map>& map,
        const std::shared_ptr<Frame>& cur_keyframe) const;

    LoopVerificationResult detectAndVerifyLoop(
        const std::shared_ptr<Map>& map,
        const std::shared_ptr<Frame>& cur_keyframe) const;

    LoopCorrectionResult applyVerifiedLoop(
        const std::shared_ptr<Map>& map,
        const std::shared_ptr<Frame>& cur_keyframe,
        const LoopVerificationResult& loop_result) const;

private:
    struct ConsistentLoopGroup
    {
        std::unordered_set<std::size_t> keyframe_ids;
        int consistency{0};
    };

    void run();

    std::unordered_set<std::size_t> collectConnectedKeyframeIds(
        const std::shared_ptr<Frame>& keyframe) const;

    bool buildLoopPnPInput(const std::shared_ptr<Frame>& candidate_keyframe,
                           const std::shared_ptr<Frame>& cur_keyframe,
                           PnPResult& pnp_result,
                           int& raw_match_num) const;

    bool setPoseGuessFromFrame(const std::shared_ptr<Frame>& frame,
                               PnPResult& pnp_result) const;

    bool isLoopCandidateAccepted(const PnPResult& pnp_result) const;

    double computeLoopCandidateScore(const PnPResult& pnp_result) const;

    bool worldPointToCameraPoint(const std::shared_ptr<Frame>& frame,
                                 const cv::Point3d& point_world,
                                 cv::Point3d& point_camera) const;

    std::vector<LoopSim3Correspondence> collectLoopSim3Correspondence(
        const std::shared_ptr<Frame>& candidate_keyframe,
        const std::shared_ptr<Frame>& cur_keyframe) const;

    bool estimateSim3ByUmeyama(const std::vector<cv::Point3d>& src_points,
                               const std::vector<cv::Point3d>& dst_points,
                               double& scale,
                               cv::Mat& R,
                               cv::Mat& t) const;

    double computeSim3PointError(const LoopSim3Correspondence& correspondence,
                                 double scale,
                                 const cv::Mat& R,
                                 const cv::Mat& t) const;

    LoopSim3Result estimateLoopSim3(const std::vector<LoopSim3Correspondence>& correspondences) const;

    bool isLoopSim3Accepted(const LoopSim3Result& sim3_result, std::size_t correspondence_num) const;

    double computeLoopSim3Score(const LoopSim3Result& sim3_result, 
                                std::size_t correspondence_num) const;

    bool buildLoopPoseGraphConstraint(const LoopVerificationResult& loop_result,
                                      cv::Mat& R_21,
                                      cv::Mat& t_21,
                                      double& scale) const;

    std::shared_ptr<MapPoint> chooseDominantMapPoint(const std::shared_ptr<MapPoint>& lhs,
                                                     const std::shared_ptr<MapPoint>& rhs) const;

    void mergeMapPoints(const std::shared_ptr<MapPoint>& keep_point,
                        const std::shared_ptr<MapPoint>& remove_point) const;

    bool computeRelativePoseBetweenFrames(const std::shared_ptr<Frame>& from_keyframe,
                                          const std::shared_ptr<Frame>& to_keyframe,
                                          cv::Mat& R_21,
                                          cv::Mat& t_21) const;

    double scale_factor_{1.2};
    int levels_num_{8};

    std::shared_ptr<KeyframeDatabase> keyframe_database_;
    const Matcher& matcher_;
    std::shared_ptr<PoseOptimizer> pose_optimizer_;
    LocalMapper* local_mapper_{nullptr};

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<LoopClosingInput> pending_keyframes_;
    std::deque<LoopClosingOutput> finished_results_;
    std::thread worker_thread_;

    mutable bool accept_keyframes_{true};
    mutable bool stop_requested_{false};
    mutable bool stopped_{false};
    mutable bool finish_requested_{false};
    mutable bool finished_{false};
    mutable bool worker_started_{false};

    mutable std::vector<ConsistentLoopGroup> consistent_loop_groups_;
};

} // namespace mini_orb_slam

#endif // MINI_ORB_SLAM_INCLUDE_LOOP_CLOSER_H_
