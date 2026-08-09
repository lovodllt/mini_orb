#ifndef MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_
#define MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_

#include <memory>
#include <list>
#include <mutex>
#include <thread>
#include <deque>
#include <atomic>
#include <condition_variable>
#include <unordered_set>

#include "common.h"
#include "frame.h"
#include "initializer.h"
#include "map.h"
#include "matcher.h"
#include "pose_optimizer.h"

namespace mini_orb_slam
{

class LocalMapper
{
public:
    LocalMapper(const std::shared_ptr<Initializer>& initializer,
                const Matcher& matcher,
                const std::shared_ptr<PoseOptimizer>& pose_optimizer,
                double scale_factor,
                int levels_num);
    ~LocalMapper();

    void start();
    void requestFinish();
    void join();
    void requestStop();
    void release();

    bool insertKeyframe(const LocalMappingInput& inoput);
    bool hasPendingKeyframe() const;
    bool tryPopFinishedResult(LocalMappingOutput& output);

    bool acceptKeyframe() const;
    bool isStopped() const;
    bool stopRequested() const;

    LocalMappingResult processNewKeyframe(const std::shared_ptr<Map>& map,
                                          const std::shared_ptr<Frame>& ref_keyframe,
                                          const std::shared_ptr<Frame>& cur_keyframe,
                                          const PnPResult& tracking_seed) const;

private:
    void run();

    void processCurrentKeyframeMapPoints(const std::shared_ptr<Frame>& keyframe) const;
    void registerRecentMapPoint(const std::shared_ptr<MapPoint>& map_point) const;
    void refreshMapPointState(const std::shared_ptr<MapPoint>& map_point) const;
    void refreshMapPointStates(const std::vector<std::shared_ptr<MapPoint>>& map_points) const;

    void updateCovisibilityGraph(const std::shared_ptr<Map>& map, 
                                 const std::shared_ptr<Frame>& keyframe) const;

    double computeMedianSceneDepth(const std::shared_ptr<Frame>& keyframe) const;

    std::size_t countReliableMapPointFeatures(const std::shared_ptr<Frame>& keyframe) const;
    
    bool isTriangulationPartnerValid(const std::shared_ptr<Frame>& keyframe,
                                     const std::shared_ptr<Frame>& cur_keyframe) const;

    std::vector<std::shared_ptr<Frame>> collectTriangulationKeyframes(
        const std::shared_ptr<Frame>& ref_frame,
        const std::shared_ptr<Frame>& cur_frame) const;

    std::size_t growMapByKeyFrames(const std::shared_ptr<Map>& map,
                                   const std::shared_ptr<Frame>& ref_keyframe,
                                   const std::shared_ptr<Frame>& cur_keyframe,
                                   std::size_t current_local_mapping_generation) const;

    std::size_t growMapByKeyFramePair(const std::shared_ptr<Map>& map,
                                      const std::shared_ptr<Frame>& ref_keyframe,
                                      const std::shared_ptr<Frame>& cur_keyframe,
                                      std::size_t current_local_mapping_generation) const;

    std::vector<std::shared_ptr<Frame>> collectFusionKeyframes(
        const std::shared_ptr<Frame>& cur_keyframe) const;

    struct FusionProjectionContext
    {
        std::shared_ptr<Camera> camera;
        cv::Mat R_cw;
        cv::Mat t_cw;
        cv::Point3d camera_center;
        cv::Size image_size;
    };

    std::size_t searchInNeighbors(const std::shared_ptr<Map>& map,
                                  const std::shared_ptr<Frame>& cur_keyframe) const;

    bool projectionMapPointToFrame(const std::shared_ptr<MapPoint>& map_point,
                                   const FusionProjectionContext& context,
                                   cv::Point2f& projected_pixel,
                                   double& camera_distance,
                                   int& pred_level) const;

    int findFuseMatchInKeyframe(const std::shared_ptr<MapPoint>& map_point,
                                 const std::shared_ptr<Frame>& keyframe,
                                 const cv::Mat& map_descriptor,
                                 const cv::Point2f& projected_pixel,
                                int pred_level,
                                const std::unordered_set<int>& used_feature_indices) const;

    std::shared_ptr<MapPoint> chooseDominantMapPoint(
        const std::shared_ptr<MapPoint>& lhs,
        const std::shared_ptr<MapPoint>& rhs) const;

    void mergeMapPoints(const std::shared_ptr<MapPoint>& keep_point,
                        const std::shared_ptr<MapPoint>& remove_point) const;

    std::size_t fuseMapPointsIntoKeyframe(
        const std::vector<std::shared_ptr<MapPoint>>& source_map_points,
        const std::shared_ptr<Frame>& target_keyframe) const;

    std::size_t cullMapPoints(const std::shared_ptr<Map>& map,
                              std::size_t current_local_mapping_generation) const;

    struct KeyframeRedundancyStats
    {
        std::size_t total_map_features{0};
        std::size_t redundant_map_features{0};
        std::size_t duplicate_keyframe_observations{0};
        double redundant_ratio{0.0};
    };

    KeyframeRedundancyStats evaluateKeyframeRedundancy(
        const std::shared_ptr<Frame>& keyframe,
        const std::shared_ptr<Map>& map) const;

    bool isKeyframeRedundant(const std::shared_ptr<Frame>& keyframe,
                             const std::shared_ptr<Map>& map,
                             KeyframeRedundancyStats* stats = nullptr) const;

    std::vector<std::shared_ptr<Frame>> collectKeyframeCullingCandidates(
        const std::shared_ptr<Frame>& cur_keyframe) const;

    std::size_t cullKeyframes(const std::shared_ptr<Map>& map,
                              const std::shared_ptr<Frame>& cur_keyframe) const;
    
    std::shared_ptr<Initializer> initializer_;
    const Matcher& matcher_;
    std::shared_ptr<PoseOptimizer> pose_optimizer_;

    double scale_factor_{1.2};
    int levels_num_{8};

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<LocalMappingInput> pending_keyframes_;
    std::deque<LocalMappingOutput> finished_results_;
    std::thread worker_thread_;

    mutable bool accept_keyframes_{true};
    mutable bool stop_requested_{false};
    mutable bool stopped_{false};
    mutable bool finish_requested_{false};
    mutable bool finished_{false};
    mutable bool worker_started_{false};

    mutable std::weak_ptr<Map> recent_map_;
    mutable std::list<std::shared_ptr<MapPoint>> recent_added_map_points_;
    mutable std::size_t local_mapping_generation_{0};
    mutable std::atomic_bool processing_new_keyframe_{false};
    // g2o consumes a bool force-stop flag. It is set when a newer keyframe is
    // queued during the opportunistic BA window.
    mutable bool abort_ba_{false};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_
