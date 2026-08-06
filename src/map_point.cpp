#include <algorithm>
#include <cmath>
#include <limits>

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

        it++;
    }

    observations_.push_back(feature);
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

        observations.reserve(observations_.size());
        for (const auto& obs : observations_)
        {
            const std::shared_ptr<Feature> feature = obs.lock();
            if (feature != nullptr)
                observations.push_back(feature);
        }

        observations_.clear();
        ref_feature_.reset();
        cur_feature_.reset();
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

    std::lock_guard<std::mutex> lock(observation_mutex_);

    observations.reserve(observations_.size());

    for (const auto& obs : observations_)
    {
        const std::shared_ptr<Feature> feature = obs.lock();
        if (feature == nullptr)
            continue;

        if (feature->getMapPoint().get() != this)
            continue;

        observations.push_back(feature);
    }
    
    return observations;
}

std::vector<std::shared_ptr<Feature>> MapPoint::getKeyframeObservations(
    const std::shared_ptr<Frame>& exclude_frame) const
{
    std::vector<std::shared_ptr<Feature>> keyframe_observations;

    std::lock_guard<std::mutex> lock(observation_mutex_);

    keyframe_observations.reserve(observations_.size());

    for (const auto& obs : observations_)
    {
        const std::shared_ptr<Feature> feature = obs.lock();
        if (feature == nullptr)
            continue;

        if (feature->getMapPoint().get() != this)
            continue;

        const std::shared_ptr<Frame> frame = feature->getFrame();
        if (frame == nullptr || !frame->isKeyframe())
            continue;

        if (exclude_frame != nullptr && frame == exclude_frame)
            continue;

        keyframe_observations.push_back(feature);
    }

    return keyframe_observations;
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
    has_view_statistics_ = false;
    normal_vector_ = cv::Point3d(0.0, 0.0, 1.0);
    min_distance_ = 0.0;
    max_distance_ = 0.0;

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

    setRefFeature(ref_feature);

    const std::shared_ptr<Frame> ref_frame = ref_feature->getFrame();
    if (ref_frame == nullptr)
        return;

    const cv::Point3d& ref_view = pos_ - ref_frame->getCameraCenter();
    const double ref_distance = pointNorm(ref_view);
    if (ref_distance <= 1e-6)
        return;

    const int ref_level = std::max(ref_feature->getLevel(), 0);
    max_distance_ = ref_distance * std::pow(scale_factor, ref_level);
    min_distance_ = max_distance_ / std::pow(scale_factor, std::max(level_num - 1, 0));

    cv::Point3d normal_sum(0.0, 0.0, 0.0);
    int valid_obs_num = 0;

    for (const auto& feature : observations)
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<Frame> frame = feature->getFrame();
        if (frame == nullptr)
            continue;

        const cv::Point3d& view = pos_ - frame->getCameraCenter();
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

    normal_vector_ = normalizePoint(normal_sum);
    has_view_statistics_ = (pointNorm(normal_vector_) > 1e-6) &&
                           (min_distance_ > 0.0) &&
                           (max_distance_ >= min_distance_);
}

void MapPoint::updateRepresentativeDescriptor()
{
    representative_descriptor_.release();

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
        representative_descriptor_ = descriptors[best_idx].clone();
}

int MapPoint::predictScaleLevel(double current_distance, double scale_factor, int levels_num) const
{
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
