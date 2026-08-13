#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "map_point.h"
#include "feature.h"
#include "frame.h"

namespace mini_orb_slam
{

namespace
{

double pointNorm(const cv::Point3d& p)
{
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

cv::Point3d normalizePoint(const cv::Point3d& p)
{
    const double norm = pointNorm(p);
    if (norm <= 1e-12)
        return cv::Point3d(0.0, 0.0, 0.0);

    return cv::Point3d(p.x / norm, p.y / norm, p.z / norm);
}

std::shared_ptr<Feature> selectBestObservationByLevel(
    const std::vector<std::shared_ptr<Feature>>& observation)
{
    std::shared_ptr<Feature> best_feature;

    for (const auto& feature : observation)
    {
        if (feature == nullptr)
            continue;

        if (best_feature == nullptr)
        {
            best_feature = feature;
            continue;
        }

        if (feature->getLevel() < best_feature->getLevel())
        {
            best_feature = feature;
            continue;
        }

        if (feature->getLevel() == best_feature->getLevel())
        {
            const std::shared_ptr<Frame> frame = feature->getFrame();
            const std::shared_ptr<Frame> best_frame = best_feature->getFrame();

            if (frame != nullptr && 
                best_frame != nullptr && 
                frame->getId() < best_frame->getId())
            {
                best_feature = feature;
            }
        }
    }

    return best_feature;
}

bool getFeatureDescriptor(const std::shared_ptr<Feature>& feature, cv::Mat& descriptor)
{
    descriptor.release();

    if (feature == nullptr)
        return false;

    const std::shared_ptr<Frame> frame = feature->getFrame();
    if (frame == nullptr || frame->getDescriptors().empty())
        return false;

    const int feature_idx = feature->getFeatureIdx();
    if (feature_idx < 0 || feature_idx >= frame->getDescriptors().rows)
        return false;

    descriptor = frame->getDescriptors().row(feature_idx).clone();
    return true;
}

int computeMedian(std::vector<int>& values)
{
    if (values.empty())
        return 0;

    const std::size_t mid = values.size() * 0.5;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

} // namespace

void MapPoint::setPos(const cv::Point3d &pos)
{
    std::lock_guard<std::mutex> lock(position_mutex_);
    pos_ = pos;
}

cv::Point3d MapPoint::getPos() const
{
    std::lock_guard<std::mutex> lock(position_mutex_);
    return pos_;
}

void MapPoint::setRefFeature(const std::shared_ptr<Feature>& feature)
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    ref_feature_ = feature;
}

void MapPoint::setCurFeature(const std::shared_ptr<Feature>& feature)
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    cur_feature_ = feature;
}

std::shared_ptr<Feature> MapPoint::getRefFeature() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return ref_feature_.lock();
}

std::shared_ptr<Feature> MapPoint::getCurFeature() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return cur_feature_.lock();
}

void MapPoint::addObservation(const std::shared_ptr<Feature>& feature)
{
    if (feature == nullptr)
        return;

    std::lock_guard<std::mutex> lock(observation_mutex_);

    for (auto it = observations_.begin(); it != observations_.end();)
    {
        const std::shared_ptr<Feature> obs = it->lock();
        if (obs == nullptr)
        {
            it = observations_.erase(it);
            continue;
        }

        if (obs == feature)
            return;

        const std::shared_ptr<Frame> obs_frame = obs->getFrame();
        const std::shared_ptr<Frame> feature_frame = feature->getFrame();
        if (obs_frame != nullptr && feature_frame != nullptr &&
            obs_frame == feature_frame)
        {
            feature->setMapPoint(nullptr);
            return;
        }

        it++;
    }

    observations_.push_back(feature);
    ++observation_generation_;
}

bool MapPoint::removeObservation(const std::shared_ptr<Feature>& feature)
{
    if (feature == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(observation_mutex_);

    bool removed = false;

    for (auto it = observations_.begin(); it != observations_.end();)
    {
        const std::shared_ptr<Feature> observed_feature = it->lock();

        if (observed_feature == nullptr || observed_feature == feature)
        {
            if (observed_feature == feature)
                removed = true;

            it = observations_.erase(it);
            continue;
        }

        it++;
    }

    if (removed)
        ++observation_generation_;

    return removed;
}

void MapPoint::replaceWith(const std::shared_ptr<MapPoint>& map_point)
{
    if (map_point == nullptr || map_point.get() == this)
        return;

    const std::vector<std::shared_ptr<Feature>> observations = getObservations();

    for (const auto& feature : observations)
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<MapPoint> current_map_point = feature->getMapPoint();
        if (current_map_point != nullptr && 
            current_map_point.get() != this &&
            current_map_point != map_point)
        {
            continue;
        }

        feature->setMapPoint(map_point);
        map_point->addObservation(feature);
    }

    if (map_point->getRefFeature() == nullptr)
        map_point->setRefFeature(getRefFeature());

    if (map_point->getCurFeature() == nullptr)
        map_point->setCurFeature(getCurFeature());

    const int visible_times = getVisibleTimes();
    const int found_times = getFoundTimes();

    for (int i = 0; i < visible_times; i++)
        map_point->increaseVisibleTimes();

    for (int i = 0; i < found_times; i++)
        map_point->increaseFoundTimes();

    setBad(true);
}

void MapPoint::setBad(bool is_bad)
{
    if (!is_bad)
    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        is_bad_ = false;
        return;
    }
    
    std::vector<std::shared_ptr<Feature>> observations;

    {
        std::lock_guard<std::mutex> lock(observation_mutex_);

        if (is_bad_)
            return;

        is_bad_ = true;
        ++observation_generation_;

        observations.reserve(observations_.size());
        for (const auto& obs : observations_)
        {
            const std::shared_ptr<Feature> feature = obs.lock();
            if (feature != nullptr)
                observations.push_back(feature);
        }

        observations_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(tracking_state_mutex_);
        ref_feature_.reset();
        cur_feature_.reset();
        representative_descriptor_.release();
        has_view_statistics_ = false;
        normal_vector_ = cv::Point3d(0.0, 0.0, 1.0);
        min_distance_ = 0.0;
        max_distance_ = 0.0;
    }

    for (const auto& feature : observations)
    {
        if (feature == nullptr)
            continue;

        if (feature->getMapPoint().get() == this)
            feature->setMapPoint(nullptr);
    }
}

bool MapPoint::isBad() const
{
    std::lock_guard<std::mutex> lock(observation_mutex_);
    return is_bad_;
}

std::vector<std::shared_ptr<Feature>> MapPoint::getObservations() const
{
    std::vector<std::shared_ptr<Feature>> observations;
    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        observations.reserve(observations_.size());
        for (const auto& obs : observations_)
        {
            const std::shared_ptr<Feature> feature = obs.lock();
            if (feature != nullptr)
                observations.push_back(feature);
        }
    }

    observations.erase(
        std::remove_if(observations.begin(), observations.end(),
                       [this](const std::shared_ptr<Feature>& feature)
                       {
                           return feature == nullptr ||
                                  feature->getMapPoint().get() != this;
                       }),
        observations.end());
    
    return observations;
}

std::vector<std::shared_ptr<Feature>> MapPoint::getKeyframeObservations(
    const std::shared_ptr<Frame>& exclude_frame) const
{
    std::vector<std::shared_ptr<Feature>> keyframe_observations;

    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        keyframe_observations.reserve(observations_.size());
        for (const auto& obs : observations_)
        {
            const std::shared_ptr<Feature> feature = obs.lock();
            if (feature != nullptr)
                keyframe_observations.push_back(feature);
        }
    }

    std::unordered_set<std::size_t> observed_keyframe_ids;
    observed_keyframe_ids.reserve(keyframe_observations.size());
    keyframe_observations.erase(
        std::remove_if(keyframe_observations.begin(), keyframe_observations.end(),
                       [this, &exclude_frame, &observed_keyframe_ids]
                       (const std::shared_ptr<Feature>& feature)
                       {
                           if (feature == nullptr ||
                               feature->getMapPoint().get() != this)
                               return true;
                           const std::shared_ptr<Frame> frame = feature->getFrame();
                           if (frame == nullptr || !frame->isKeyframe() ||
                               (exclude_frame != nullptr && frame == exclude_frame))
                               return true;
                           return !observed_keyframe_ids.insert(frame->getId()).second;
                       }),
        keyframe_observations.end());

    return keyframe_observations;
}

std::vector<std::shared_ptr<Frame>> MapPoint::getKeyframeObservationFrames(
    const std::shared_ptr<Frame>& exclude_frame) const
{
    std::vector<std::shared_ptr<Feature>> observations;
    std::size_t observation_generation = 0;
    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        observation_generation = observation_generation_;
        if (keyframe_cache_generation_ == observation_generation_)
        {
            std::vector<std::shared_ptr<Frame>> cached_frames;
            cached_frames.reserve(keyframe_observation_frame_cache_.size());
            for (const auto& cached_frame : keyframe_observation_frame_cache_)
            {
                const std::shared_ptr<Frame> frame = cached_frame.lock();
                if (frame != nullptr &&
                    (exclude_frame == nullptr || frame != exclude_frame))
                {
                    cached_frames.push_back(frame);
                }
            }
            return cached_frames;
        }

        observations.reserve(observations_.size());
        for (const auto& obs : observations_)
        {
            const std::shared_ptr<Feature> feature = obs.lock();
            if (feature != nullptr)
                observations.push_back(feature);
        }
    }

    std::vector<std::shared_ptr<Frame>> keyframe_frames;
    keyframe_frames.reserve(observations.size());
    std::unordered_set<std::size_t> observed_keyframe_ids;
    observed_keyframe_ids.reserve(observations.size());

    for (const auto& feature : observations)
    {
        if (feature == nullptr || feature->getMapPoint().get() != this)
            continue;

        const std::shared_ptr<Frame> frame = feature->getFrame();
        if (frame == nullptr || !frame->isKeyframe() ||
            (exclude_frame != nullptr && frame == exclude_frame))
        {
            continue;
        }

        if (observed_keyframe_ids.insert(frame->getId()).second)
            keyframe_frames.push_back(frame);
    }

    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        if (observation_generation_ == observation_generation)
        {
            keyframe_observation_frame_cache_.clear();
            keyframe_observation_frame_cache_.reserve(keyframe_frames.size());
            for (const auto& frame : keyframe_frames)
                keyframe_observation_frame_cache_.push_back(frame);
            keyframe_cache_generation_ = observation_generation;
        }
    }

    return keyframe_frames;
}

std::shared_ptr<Feature> MapPoint::selectRefFeatureCandidate() const
{
    const std::vector<std::shared_ptr<Feature>> keyframe_observations = getKeyframeObservations();
    const std::shared_ptr<Feature> keyframe_feature = 
        selectBestObservationByLevel(keyframe_observations);
    if (keyframe_feature != nullptr)
        return keyframe_feature;

    const std::vector<std::shared_ptr<Feature>> observations = getObservations();
    const std::shared_ptr<Feature> best_feature = 
        selectBestObservationByLevel(observations);
    if (best_feature != nullptr)
        return best_feature;

    return getCurFeature();
}

void MapPoint::updateViewStatistics(double scale_factor, int level_num)
{
    if (scale_factor <= 1.0 || level_num <= 0)
        return;

    std::vector<std::shared_ptr<Feature>> observations = getKeyframeObservations();
    if (observations.empty())
        observations = getObservations();

    if (observations.empty())
        return;

    std::shared_ptr<Feature> ref_feature = selectRefFeatureCandidate();
    if (ref_feature == nullptr)
        return;

    const std::shared_ptr<Frame> ref_frame = ref_feature->getFrame();
    if (ref_frame == nullptr)
        return;

    const cv::Point3d pos = getPos();
    const cv::Point3d ref_view = pos - ref_frame->getCameraCenter();
    const double ref_distance = pointNorm(ref_view);
    if (ref_distance <= 1e-6)
        return;

    const int ref_level = std::max(ref_feature->getLevel(), 0);
    const double max_distance = ref_distance * std::pow(scale_factor, ref_level);
    const double min_distance =
        max_distance / std::pow(scale_factor, std::max(level_num - 1, 0));

    cv::Point3d normal_sum(0.0, 0.0, 0.0);
    int valid_obs_num = 0;

    for (const auto& feature : observations)
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<Frame> frame = feature->getFrame();
        if (frame == nullptr)
            continue;

        const cv::Point3d view = pos - frame->getCameraCenter();
        const double distance = pointNorm(view);
        if (distance <= 1e-6)
            continue;

        normal_sum.x += view.x / distance;
        normal_sum.y += view.y / distance;
        normal_sum.z += view.z / distance;
        valid_obs_num++;
    }

    if (valid_obs_num <= 0)
        return;

    const cv::Point3d normal_vector = normalizePoint(normal_sum);
    const bool has_view_statistics = (pointNorm(normal_vector) > 1e-6) &&
                                     (min_distance > 0.0) &&
                                     (max_distance >= min_distance);

    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    ref_feature_ = ref_feature;
    normal_vector_ = normal_vector;
    min_distance_ = min_distance;
    max_distance_ = max_distance;
    has_view_statistics_ = has_view_statistics;
}

void MapPoint::updateRepresentativeDescriptor()
{
    std::vector<std::shared_ptr<Feature>> observations = getKeyframeObservations();
    if (observations.empty())
        observations = getObservations();

    if (observations.empty())
        return;

    std::vector<cv::Mat> descriptors;
    descriptors.reserve(observations.size());

    for (const auto& feature : observations)
    {
        cv::Mat descriptor;
        if (!getFeatureDescriptor(feature, descriptor))
            continue;

        descriptors.push_back(descriptor);
    }

    if (descriptors.empty())
        return;

    if (descriptors.size() == 1)
    {
        std::lock_guard<std::mutex> lock(tracking_state_mutex_);
        representative_descriptor_ = descriptors.front().clone();
        return;
    }

    int best_idx = -1;
    int best_median = std::numeric_limits<int>::max();
    int best_sum = std::numeric_limits<int>::max();

    for (int i = 0; i < static_cast<int>(descriptors.size()); i++)
    {
        std::vector<int> distances;
        distances.reserve(descriptors.size() - 1);

        int distance_sum = 0;
        for (int j = 0; j < static_cast<int>(descriptors.size()); j++)
        {
            if (i == j)
                continue;

            const int dist = static_cast<int>(
                cv::norm(descriptors[i], descriptors[j], cv::NORM_HAMMING));
            distances.push_back(dist);
            distance_sum += dist;
        }

        const int median_distance = computeMedian(distances);

        if (median_distance < best_median || 
            (median_distance == best_median && distance_sum < best_sum))
        {
            best_idx = i;
            best_median = median_distance;
            best_sum = distance_sum;
        }
    }

    if (best_idx >= 0)
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    representative_descriptor_ = descriptors[best_idx].clone();
}

bool MapPoint::hasValidViewStatistics() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return has_view_statistics_;
}

bool MapPoint::getViewStatistics(cv::Point3d& normal_vector,
                                 double& min_distance,
                                 double& max_distance) const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    if (!has_view_statistics_)
        return false;

    normal_vector = normal_vector_;
    min_distance = min_distance_;
    max_distance = max_distance_;
    return true;
}

void MapPoint::increaseVisibleTimes()
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    visible_times_++;
}

void MapPoint::increaseFoundTimes()
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    found_times_++;
}

int MapPoint::getVisibleTimes() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return visible_times_;
}

int MapPoint::getFoundTimes() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return found_times_;
}

double MapPoint::getFoundRatio() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    if (visible_times_ <= 0)
        return 0.0;

    return static_cast<double>(found_times_) / visible_times_;
}

bool MapPoint::hasRepresentativeDescriptor() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return !representative_descriptor_.empty();
}

cv::Mat MapPoint::getRepresentativeDescriptor() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return representative_descriptor_.clone();
}

bool MapPoint::getRepresentativeDescriptorView(cv::Mat& descriptor) const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    descriptor = representative_descriptor_;
    return !descriptor.empty();
}

cv::Point3d MapPoint::getNormalVector() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return normal_vector_;
}

double MapPoint::getMinDistance() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return min_distance_;
}

double MapPoint::getMaxDistance() const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    return max_distance_;
}

int MapPoint::predictScaleLevel(double current_distance, double scale_factor, int levels_num) const
{
    std::lock_guard<std::mutex> lock(tracking_state_mutex_);
    if (!has_view_statistics_ || current_distance <= 1e-6 ||
        scale_factor <= 1.0 || levels_num <= 0)
    {
        return 0;
    }

    const double ratio = max_distance_ / current_distance;
    const int predicted_level = 
        static_cast<int>(std::ceil(std::log(ratio) / std::log(scale_factor)));

    return std::max(0, std::min(predicted_level, levels_num - 1));
}


} // namespace mini_orb_slam
