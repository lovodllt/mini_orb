#ifndef MINI_ORB_SLAM_INCLUDE_MAP_POINT_H_
#define MINI_ORB_SLAM_INCLUDE_MAP_POINT_H_

#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <limits>
#include <opencv2/core.hpp>

namespace mini_orb_slam
{

class Feature;
class Frame;

class MapPoint
{
public:
    MapPoint() = default;
    MapPoint(std::size_t id, const cv::Point3d& pos) : id_(id), pos_(pos) {};

    std::size_t getId() const { return id_; }

    cv::Point3d getPos() const;
    void setPos(const cv::Point3d& pos);

    void addObservation(const std::shared_ptr<Feature>& feature);
    bool removeObservation(const std::shared_ptr<Feature>& feature);
    void replaceWith(const std::shared_ptr<MapPoint>& map_point);

    std::vector<std::shared_ptr<Feature>> getObservations() const;
    std::size_t getObservationCount() const { return getObservations().size(); };

    std::vector<std::shared_ptr<Feature>> getKeyframeObservations(
        const std::shared_ptr<Frame>& exclude_frame = nullptr) const;
    std::vector<std::shared_ptr<Frame>> getKeyframeObservationFrames(
        const std::shared_ptr<Frame>& exclude_frame = nullptr) const;
    std::size_t getKeyframeObservationCount(
        const std::shared_ptr<Frame>& exclude_frame = nullptr) const
    {
        return getKeyframeObservations(exclude_frame).size();
    }

    std::shared_ptr<Feature> selectRefFeatureCandidate() const;

    void setRefFeature(const std::shared_ptr<Feature>& feature);
    void setCurFeature(const std::shared_ptr<Feature>& feature);

    std::shared_ptr<Feature> getRefFeature() const;
    std::shared_ptr<Feature> getCurFeature() const;

    void setFirstKeyframeId(std::size_t keyframe_id) { first_keyframe_id_ = keyframe_id; }
    std::size_t getFirstKeyframeId() const { return first_keyframe_id_; }

    void setFirstLocalMappingGeneration(std::size_t generation) 
    { 
        first_local_mapping_generation_ = generation; 
    }

    std::size_t getFirstLocalMappingGeneration() const { return first_local_mapping_generation_; }

    void setBad(bool is_bad);
    bool isBad() const;

    void updateViewStatistics(double scale_factor, int levels_num);
    bool hasValidViewStatistics() const;
    bool getViewStatistics(cv::Point3d& normal_vector,
                           double& min_distance,
                           double& max_distance) const;

    void increaseVisibleTimes();
    void increaseFoundTimes();

    int getVisibleTimes() const;
    int getFoundTimes() const;

    double getFoundRatio() const;

    void updateRepresentativeDescriptor();
    bool hasRepresentativeDescriptor() const;
    cv::Mat getRepresentativeDescriptor() const;
    bool getRepresentativeDescriptorView(cv::Mat& descriptor) const;

    cv::Point3d getNormalVector() const;
    double getMinDistance() const;
    double getMaxDistance() const;

    int predictScaleLevel(double cur_distance, double scale_factor, int levels_num) const;

private:
    std::size_t id_{0};
    cv::Point3d pos_;

    std::vector<std::weak_ptr<Feature>> observations_;

    mutable std::vector<std::weak_ptr<Frame>> keyframe_observation_frame_cache_;
    mutable std::size_t observation_generation_{0};
    mutable std::size_t keyframe_cache_generation_{
        std::numeric_limits<std::size_t>::max()};

    std::weak_ptr<Feature> ref_feature_; 
    std::weak_ptr<Feature> cur_feature_; 

    std::size_t first_keyframe_id_{0};
    std::size_t first_local_mapping_generation_{0};
    bool is_bad_{false};

    cv::Point3d normal_vector_{0.0, 0.0, 1.0};
    double min_distance_{0.0};
    double max_distance_{0.0};
    bool has_view_statistics_{false};

    cv::Mat representative_descriptor_;

    int visible_times_{0};
    int found_times_{0};

    mutable std::mutex position_mutex_;
    mutable std::mutex observation_mutex_;
    mutable std::mutex tracking_state_mutex_;
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_MAP_POINT_H_
