#ifndef MINI_ORB_SLAM_INCLUDE_MAP_H_
#define MINI_ORB_SLAM_INCLUDE_MAP_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <cmath>
#include <cstddef>
#include <vector>
#include <mutex>

#include <opencv4/opencv2/core.hpp>

#include "frame.h"
#include "map_point.h"

namespace mini_orb_slam
{

enum class PoseGraphConstraintKind
{
    SEQUENTIAL = 0,
    COVISIBILITY = 1,
    LOOP = 2
};

struct PoseGraphConstraint
{
    std::weak_ptr<Frame> from_keyframe;
    std::weak_ptr<Frame> to_keyframe;

    int weight{1};

    cv::Mat R_21;
    cv::Mat t_21;
    double scale{1.0};

    PoseGraphConstraintKind kind{PoseGraphConstraintKind::SEQUENTIAL};
};

class Map
{
public:
    Map() = default;

    std::mutex& getMutex() const
    {
        return mutex_;
    }

    void clear()
    {
        map_points_.clear();
        keyframes_.clear();
        pose_graph_constraints_.clear();
        next_map_point_id_.store(0, std::memory_order_relaxed);
        markModified();
    }

    std::size_t allocateMapPointId()
    {
        return next_map_point_id_.fetch_add(1, std::memory_order_relaxed);
    }

    void addMapPoint(const std::shared_ptr<MapPoint>& map_point)
    {
        if (map_point != nullptr)
        {
            map_points_.push_back(map_point);
            markModified();
        }
    }

    void addKeyframe(const std::shared_ptr<Frame>& keyframe);

    void recordCovisibilityConstraints(const std::shared_ptr<Frame>& keyframe);

    void removeBadMapPoints()
    {
        const std::size_t map_point_num_before = map_points_.size();
        map_points_.erase(
            std::remove_if(
                map_points_.begin(),
                map_points_.end(),
                [](const std::shared_ptr<MapPoint>& map_point)
                {
                    return map_point == nullptr || map_point->isBad();
                }),
            map_points_.end());

        if (map_points_.size() != map_point_num_before)
            markModified();
    }

    void removeKeyframe(const std::shared_ptr<Frame>& keyframe)
    {
        if (keyframe == nullptr)
            return;

        // Retire the frame before rebuilding graph state. This prevents stale
        // keyframe observations from surviving in MapPoint/Frame covisibility
        // queries while external shared_ptr users still hold the Frame.
        std::vector<std::shared_ptr<MapPoint>> touched_map_points;
        for (const auto& feature : keyframe->getFeatures())
        {
            if (feature == nullptr)
                continue;

            const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
            if (map_point == nullptr)
                continue;

            feature->setMapPoint(nullptr);
            map_point->removeObservation(feature);
            touched_map_points.push_back(map_point);
        }

        keyframe->setKeyframe(false);

        keyframes_.erase(
            std::remove_if(
                keyframes_.begin(),
                keyframes_.end(),
                [&keyframe](const std::shared_ptr<Frame>& candidate)
                {
                    return candidate == nullptr || candidate == keyframe;
                }),
            keyframes_.end());

        for (const auto& map_point : touched_map_points)
        {
            if (map_point != nullptr && !map_point->isBad())
            {
                map_point->updateViewStatistics(1.2, 8);
                map_point->updateRepresentativeDescriptor();
            }
        }

        for (const auto& surviving_keyframe : keyframes_)
        {
            if (surviving_keyframe != nullptr)
                surviving_keyframe->updateConnections();
        }

        pose_graph_constraints_.erase(
            std::remove_if(
                pose_graph_constraints_.begin(),
                pose_graph_constraints_.end(),
                [&keyframe](const PoseGraphConstraint& constraint)
                {
                    const std::shared_ptr<Frame> from_keyframe = constraint.from_keyframe.lock();
                    const std::shared_ptr<Frame> to_keyframe = constraint.to_keyframe.lock();

                    return from_keyframe == nullptr || to_keyframe == nullptr ||
                           from_keyframe == keyframe || to_keyframe == keyframe;
                }),
            pose_graph_constraints_.end());

        markModified();
    }

    std::size_t getMapPointNum() const { return map_points_.size(); }
    std::size_t getKeyframeNum() const { return keyframes_.size(); }

    // The caller must hold getMutex(). This is a map-transaction generation,
    // used to reject an optimization solved from a stale immutable snapshot.
    std::size_t getVersion() const
    {
        return version_.load(std::memory_order_acquire);
    }

    void markModified()
    {
        version_.fetch_add(1, std::memory_order_release);
    }

    const std::vector<std::shared_ptr<MapPoint>>& getMapPoints() const { return map_points_; }
    const std::vector<std::shared_ptr<Frame>>& getKeyframes() const { return keyframes_; }

    // Snapshot readers may be used outside a map transaction. They retain the
    // pointed-to objects while LocalMapping changes the owning containers.
    std::vector<std::shared_ptr<MapPoint>> copyMapPoints() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_points_;
    }

    std::vector<std::shared_ptr<Frame>> copyKeyframes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return keyframes_;
    }

    std::shared_ptr<Frame> copyLastKeyframe() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return keyframes_.empty() ? nullptr : keyframes_.back();
    }

    std::shared_ptr<Frame> getLastKeyframe() const
    {
        if (keyframes_.empty())
            return nullptr;

        return keyframes_.back();
    }

    bool addPoseGraphConstraint(const std::shared_ptr<Frame>& from_keyframe, 
                                const std::shared_ptr<Frame>& to_keyframe, 
                                int weight,
                                const cv::Mat& R_21,
                                const cv::Mat& t_21,
                                double scale,
                                PoseGraphConstraintKind kind);

    bool hasPoseGraphConstraint(const std::shared_ptr<Frame>& from_keyframe, 
                                const std::shared_ptr<Frame>& to_keyframe, 
                                PoseGraphConstraintKind kind) const;

    const std::vector<PoseGraphConstraint>& getPoseGraphConstraints() const 
    {
        return pose_graph_constraints_;
    }

    // Rebuild geometry-derived graph measurements after Local BA changed
    // keyframe poses. Verified loop measurements are intentionally preserved.
    std::size_t refreshPoseGraphMeasurements();

private:
    std::vector<std::shared_ptr<MapPoint>> map_points_;
    std::vector<std::shared_ptr<Frame>> keyframes_;
    std::vector<PoseGraphConstraint> pose_graph_constraints_;
    std::atomic<std::size_t> next_map_point_id_{0};
    std::atomic<std::size_t> version_{0};

    mutable std::mutex mutex_;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_MAP_H_
