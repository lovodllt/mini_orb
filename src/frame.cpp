#include <cmath>
#include <algorithm>

#include "frame.h"
#include "map_point.h"

namespace mini_orb_slam
{

Frame::Frame(std::size_t id, double timestamp, const cv::Mat& img, const std::shared_ptr<Camera>& camera) :
    id_{id},
    timestamp_(timestamp), 
    img_(img.clone()), 
    camera_(camera),
    R_cw_(cv::Mat::eye(3, 3, CV_64F)),
    t_cw_(cv::Mat::zeros(3, 1, CV_64F)) {}

void Frame::setFeatures(const std::vector<std::shared_ptr<Feature>>& features, const cv::Mat& descriptors)
{
    features_ = features;
    descriptors_ = descriptors.clone();
    clearBoW();
    updateFeatureCache();
}

void Frame::clearFeatures()
{
    features_.clear();
    keypoints_.clear();
    descriptors_.release();
    pyramid_levels_.clear();

    feature_grid_.clear();
    grid_cols_ = 0;
    grid_rows_ = 0;
    grid_cell_width_inv_ = 0.0f;
    grid_cell_height_inv_ = 0.0f;

    connected_keyframes_.clear();

    clearBoW();
}

void Frame::setPose(const cv::Mat& R_cw, const cv::Mat& t_cw)
{
    if (R_cw.empty() || t_cw.empty())
        return;

    std::lock_guard<std::mutex> lock(pose_mutex_);

    R_cw.convertTo(R_cw_, CV_64F);
    t_cw.convertTo(t_cw_, CV_64F);
}

void Frame::copyPose(cv::Mat& R_cw, cv::Mat& t_cw) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    R_cw = R_cw_.clone();
    t_cw = t_cw_.clone();
}

cv::Mat Frame::getRwc() const
{
    cv::Mat R_cw;
    cv::Mat t_cw;
    copyPose(R_cw, t_cw);

    if (R_cw.empty())
        return cv::Mat();

    return R_cw.t();
}

cv::Point3d Frame::getCameraCenter() const
{
    cv::Mat R_cw;
    cv::Mat t_cw;
    copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return cv::Point3d(0.0, 0.0, 0.0);

    const cv::Mat center = -R_cw.t() * t_cw;
    
    return cv::Point3d(center.at<double>(0), center.at<double>(1), center.at<double>(2));
}

void Frame::updateConnections()
{
    if (!isKeyframe())
    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connected_keyframes_.clear();
        return;
    }

    std::unordered_map<std::size_t, int> keyframe_votes;
    std::unordered_map<std::size_t, std::shared_ptr<Frame>> voted_keyframes;

    for (const auto& feature : features_)
    {
        if (feature == nullptr)
            continue;

        const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::vector<std::shared_ptr<Frame>> observations =
            map_point->getKeyframeObservationFrames();

        for (const auto& keyframe : observations)
        {
            if (keyframe == nullptr || keyframe.get() == this || !keyframe->isKeyframe())
                continue;

            keyframe_votes[keyframe->getId()]++;
            voted_keyframes[keyframe->getId()] = keyframe;
        }
    }

    std::unordered_map<std::size_t, KeyframeConnection> new_connections;

    if (!keyframe_votes.empty())
    {
        int best_weight = 0;
        for (const auto& kv : keyframe_votes)
            best_weight = std::max(best_weight, kv.second);

        constexpr int kMinCovisibilityWeight = 15;

        for (const auto& vk : voted_keyframes)
        {
            const auto vote_it = keyframe_votes.find(vk.first);
            if (vote_it == keyframe_votes.end() || vk.second == nullptr)
                continue;

            const int weight = vote_it->second;
            if (weight < kMinCovisibilityWeight && weight < best_weight)
                continue;

            new_connections[vk.first] = {vk.second, weight};
        }   
    }

    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connected_keyframes_ = std::move(new_connections);
    }
}

std::vector<std::shared_ptr<Frame>> Frame::getConnectedKeyframes(int min_weight) const
{
    return copyConnectedKeyframes(min_weight);
}

std::vector<std::shared_ptr<Frame>> Frame::copyConnectedKeyframes(int min_weight) const
{
    std::vector<std::shared_ptr<Frame>> keyframes;

    std::lock_guard<std::mutex> lock(connection_mutex_);
    keyframes.reserve(connected_keyframes_.size());

    for (const auto& item : connected_keyframes_)
    {
        if (item.second.weight < min_weight)
            continue;

        const std::shared_ptr<Frame> keyframe = item.second.keyframe.lock();
        if (keyframe != nullptr)
            keyframes.push_back(keyframe);
    }

    return keyframes;
}

void Frame::computeBoW(const std::shared_ptr<BoWVocabulary>& vocabulary)
{
    clearBoW();

    if (vocabulary == nullptr || !vocabulary->isLoaded() || descriptors_.empty())
        return;

    has_bow_ = vocabulary->transform(descriptors_, bow_vector_, feature_vector_);
}

void Frame::clearBoW()
{
    bow_vector_.clear();
    feature_vector_.clear();
    has_bow_ = false;
}

std::vector<std::shared_ptr<Frame>> Frame::getBestCovisibilityKeyframes(
    std::size_t max_num, int min_weight) const
{
    return copyBestCovisibilityKeyframes(max_num, min_weight);
}

std::vector<std::shared_ptr<Frame>> Frame::copyBestCovisibilityKeyframes(
    std::size_t max_num, int min_weight) const
{
    struct RankedKeyframe
    {
        std::shared_ptr<Frame> keyframe;
        int weight{0};
    };

    std::vector<RankedKeyframe> ranked;

    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        ranked.reserve(connected_keyframes_.size());

        for (const auto& item : connected_keyframes_)
        {
            if (item.second.weight < min_weight)
                continue;

            const std::shared_ptr<Frame> keyframe = item.second.keyframe.lock();
            if (keyframe == nullptr)
                continue;

            ranked.push_back({keyframe, item.second.weight});
        }
    }
    
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedKeyframe& a, const RankedKeyframe& b) 
              {
                  if (a.weight != b.weight)
                      return a.weight > b.weight;

                  return a.keyframe->getId() > b.keyframe->getId();
              });

    std::vector<std::shared_ptr<Frame>> result;
    result.reserve(std::min(max_num, ranked.size()));

    for (std::size_t i = 0; i < ranked.size() && i < max_num; i++)
        result.push_back(ranked[i].keyframe);

    return result;
}

int Frame::getConnectionWeight(std::size_t keyframe_id) const
{
    return copyConnectionWeight(keyframe_id);
}

int Frame::copyConnectionWeight(std::size_t keyframe_id) const
{
    std::lock_guard<std::mutex> lock(connection_mutex_);
    const auto it = connected_keyframes_.find(keyframe_id);
    if (it == connected_keyframes_.end())
        return 0;

    return it->second.weight;
}

void Frame::updateFeatureCache()
{
    keypoints_.clear();
    pyramid_levels_.clear();

    keypoints_.reserve(features_.size());
    pyramid_levels_.reserve(features_.size());

    for (const auto& feature : features_)
    {
        if (feature == nullptr)
            continue;

        keypoints_.push_back(feature->getKeyPoint());
        pyramid_levels_.push_back(feature->getLevel());
    }

    buildFeatureGrid();
}

bool Frame::pointToGridCell(const cv::Point2f& pt, int& cell_x, int& cell_y) const
{
    if (img_.empty() || grid_cols_ <= 0 || grid_rows_ <= 0)
        return false;

    cell_x = static_cast<int>(pt.x * grid_cell_width_inv_);
    cell_y = static_cast<int>(pt.y * grid_cell_height_inv_);

    if (cell_x < 0 || cell_x >= grid_cols_ || cell_y < 0 || cell_y >= grid_rows_)
        return false;

    return true;
}

void Frame::buildFeatureGrid()
{
    feature_grid_.clear();
    grid_cols_ = 0;
    grid_rows_ = 0;
    grid_cell_width_inv_ = 0.0f;
    grid_cell_height_inv_ = 0.0f;

    if (img_.empty())
        return;

    grid_cols_ = 
        static_cast<int>(std::ceil(static_cast<float>(img_.cols) / kGridCellSizePx));
    grid_rows_ = 
        static_cast<int>(std::ceil(static_cast<float>(img_.rows) / kGridCellSizePx));

    grid_cell_width_inv_ = static_cast<float>(grid_cols_) / img_.cols;
    grid_cell_height_inv_ = static_cast<float>(grid_rows_) / img_.rows;

    feature_grid_.assign(grid_cols_ * grid_rows_, {});

    const std::size_t reserve_num = feature_grid_.empty() 
                                    ? 0 
                                    : (features_.size() + feature_grid_.size() - 1) 
                                        / feature_grid_.size();

    for (auto& cell : feature_grid_)
        cell.reserve(reserve_num);

    for (int i = 0; i < static_cast<int>(features_.size()); i++)
    {
        const std::shared_ptr<Feature>& feature = features_[i];
        if (feature == nullptr)
            continue;

        int cell_x = 0, cell_y = 0;
        if (!pointToGridCell(feature->getKeyPoint().pt, cell_x, cell_y))
            continue;

        feature_grid_[cell_y * grid_cols_ + cell_x].push_back(i);
    }
}

std::vector<int> Frame::getFeatureIndicesInArea(const cv::Point2d& pt, 
                                                float radius, 
                                                int min_level, 
                                                int max_level) const 
{
    std::vector<int> indices;

    appendFeatureIndicesInArea(pt, radius, min_level, max_level, indices);
    return indices;
}

void Frame::appendFeatureIndicesInArea(const cv::Point2d& pt,
                                       float radius,
                                       int min_level,
                                       int max_level,
                                       std::vector<int>& indices) const
{
    indices.clear();

    if (feature_grid_.empty() || img_.empty())
        return;

    const int min_cell_x = 
        std::max(0, static_cast<int>((pt.x - radius) * grid_cell_width_inv_));
    const int max_cell_x = 
        std::min(grid_cols_ - 1, static_cast<int>((pt.x + radius) * grid_cell_width_inv_));
    const int min_cell_y = 
        std::max(0, static_cast<int>((pt.y - radius) * grid_cell_height_inv_));
    const int max_cell_y = 
        std::min(grid_rows_ - 1, static_cast<int>((pt.y + radius) * grid_cell_height_inv_));

    if (min_cell_x > max_cell_x || min_cell_y > max_cell_y)
        return;

    for (int y = min_cell_y; y <= max_cell_y; y++)
    {
        for (int x = min_cell_x; x <= max_cell_x; x++)
        {
            const std::vector<int>& cell = feature_grid_[y * grid_cols_ + x];
            for (const int feature_idx : cell)
            {
                if (feature_idx < 0 || feature_idx >= static_cast<int>(features_.size()))
                    continue;

                const std::shared_ptr<Feature>& feature = features_[feature_idx];
                if (feature == nullptr)
                    continue;

                const int level = feature->getLevel();
                if (level < min_level || level > max_level)
                    continue;

                indices.push_back(feature_idx);
            }
        }
    }

}

} // namespace mini_orb_slam
