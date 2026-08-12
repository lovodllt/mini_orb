#ifndef MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_
#define MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_

#include <memory>
#include <list>
#include <mutex>
#include <thread>
#include <deque>
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

class BoWVocabulary;
class KeyframeDatabase;
class LoopCloser;

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

    // These dependencies are configured before start() and are then owned by
    // Frontend for the whole LocalMapper worker lifetime.
    void setKeyframeDatabase(const std::shared_ptr<BoWVocabulary>& vocabulary,
                             const std::shared_ptr<KeyframeDatabase>& database);
    void setLoopCloser(LoopCloser* loop_closer);

    // A small FIFO absorbs transient Local BA bursts without allowing
    // unbounded tracking-to-mapping backlog.
    bool insertKeyframe(const LocalMappingInput& input);
    std::shared_ptr<Frame> getLatestScheduledKeyframe(
        const std::shared_ptr<Map>& map) const;
    bool hasPendingKeyframe() const;
    bool tryPopFinishedResult(LocalMappingOutput& output);
    bool waitPopFinishedResult(LocalMappingOutput& output);

    // Results are consumed independently from admission.  A completed output
    // must never stall tracking's next Mapper admission.
    bool acceptKeyframe() const;
    bool isStopped() const;
    bool stopRequested() const;
    bool finishRequested() const;

    LocalMappingResult processNewKeyframe(const std::shared_ptr<Map>& map,
                                          const std::shared_ptr<Frame>& ref_keyframe,
                                          const std::shared_ptr<Frame>& cur_keyframe,
                                          const PnPResult& tracking_seed);

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

    struct TriangulationMatchCache
    {
        std::shared_ptr<Frame> ref_keyframe;
        std::vector<cv::DMatch> raw_matches;
    };

    std::vector<TriangulationMatchCache> collectTriangulationMatchCache(
        const std::vector<std::shared_ptr<Frame>>& triangulation_keyframes,
        const std::shared_ptr<Frame>& cur_keyframe) const;

    std::size_t growMapByKeyFrames(const std::shared_ptr<Map>& map,
                                   const std::vector<TriangulationMatchCache>& match_cache,
                                   const std::shared_ptr<Frame>& cur_keyframe,
                                   std::size_t current_local_mapping_generation,
                                   LocalMappingResult* result) const;

    std::size_t growMapByKeyFramePair(const std::shared_ptr<Map>& map,
                                      const std::shared_ptr<Frame>& ref_keyframe,
                                      const std::shared_ptr<Frame>& cur_keyframe,
                                      const std::vector<cv::DMatch>& raw_matches,
                                      std::size_t current_local_mapping_generation,
                                      LocalMappingResult* result) const;

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
                                  const std::shared_ptr<Frame>& cur_keyframe,
                                  LocalMappingResult* result) const;

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
                                 const std::unordered_set<int>& used_feature_indices,
                                std::vector<int>& candidate_indices,
                                LocalMappingResult* result) const;

    std::shared_ptr<MapPoint> chooseDominantMapPoint(
        const std::shared_ptr<MapPoint>& lhs,
        const std::shared_ptr<MapPoint>& rhs) const;

    void mergeMapPoints(const std::shared_ptr<MapPoint>& keep_point,
                        const std::shared_ptr<MapPoint>& remove_point) const;

    std::size_t fuseMapPointsIntoKeyframe(
        const std::vector<std::shared_ptr<MapPoint>>& source_map_points,
        const std::shared_ptr<Frame>& target_keyframe,
        LocalMappingResult* result) const;

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
        const std::shared_ptr<Map>& map,
        LocalMappingResult* result) const;

    bool isKeyframeRedundant(const std::shared_ptr<Frame>& keyframe,
                             const std::shared_ptr<Map>& map,
                             KeyframeRedundancyStats* stats = nullptr,
                             LocalMappingResult* result = nullptr) const;

    std::vector<std::shared_ptr<Frame>> collectKeyframeCullingCandidates(
        const std::shared_ptr<Frame>& cur_keyframe) const;

    std::size_t cullKeyframes(const std::shared_ptr<Map>& map,
                              const std::shared_ptr<Frame>& cur_keyframe,
                              LocalMappingResult* result = nullptr) const;

    void handOffCommittedKeyframe(const std::shared_ptr<Map>& map,
                                  const std::shared_ptr<Frame>& cur_keyframe,
                                  LocalMappingResult& result) const;
    
    std::shared_ptr<Initializer> initializer_;
    const Matcher& matcher_;
    std::shared_ptr<PoseOptimizer> pose_optimizer_;
    std::shared_ptr<BoWVocabulary> bow_vocabulary_;
    std::shared_ptr<KeyframeDatabase> keyframe_database_;
    LoopCloser* loop_closer_{nullptr};

    double scale_factor_{1.2};
    int levels_num_{8};

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<LocalMappingInput> pending_keyframes_;
    std::deque<LocalMappingOutput> finished_results_;
    std::thread worker_thread_;

    static constexpr std::size_t kMaxPendingKeyframes = 2;
    std::weak_ptr<Map> latest_scheduled_map_;
    std::weak_ptr<Frame> latest_scheduled_keyframe_;

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
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_LOCAL_MAPPER_H_
