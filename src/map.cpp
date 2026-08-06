#include "map.h"

namespace mini_orb_slam
{

namespace  
{

bool computeRelativePoseConstraint(const std::shared_ptr<Frame>& from_keyframe,
                                   const std::shared_ptr<Frame>& to_keyframe,
                                   cv::Mat& R_21,
                                   cv::Mat& t_21)
{
    R_21.release();
    t_21.release();


    if (from_keyframe == nullptr || to_keyframe == nullptr)
        return false;

    cv::Mat R_from, t_from;
    cv::Mat R_to, t_to;

    from_keyframe->copyPose(R_from, t_from);
    to_keyframe->copyPose(R_to, t_to);

    if (R_from.empty() || t_from.empty() || R_to.empty() || t_to.empty())
        return false;

    R_21 = R_to * R_from.t();
    t_21 = t_to - R_21 * t_from;
    
    return true;
}

} // namespace

void Map::addKeyframe(const std::shared_ptr<Frame>& keyframe)
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return;

    if (std::find(keyframes_.begin(), keyframes_.end(), keyframe) != keyframes_.end())
        return; 

    const std::shared_ptr<Frame> previous_keyframe = getLastKeyframe();
    keyframes_.push_back(keyframe);

    if (previous_keyframe == nullptr)
        return;

    cv::Mat R_21, t_21;

    if (!computeRelativePoseConstraint(previous_keyframe, keyframe, R_21, t_21))
        return;

    constexpr int kSequentialEdgeWeight = 1000;

    addPoseGraphConstraint(previous_keyframe, 
                           keyframe, 
                           kSequentialEdgeWeight, 
                           R_21, 
                           t_21, 
                           1.0, 
                           PoseGraphConstraintKind::SEQUENTIAL);      
}

void Map::recordCovisibilityConstraints(const std::shared_ptr<Frame>& keyframe)
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return;

    if (std::find(keyframes_.begin(), keyframes_.end(), keyframe) == keyframes_.end())
        return;

    constexpr int kMinCovisibilityWeight = 30;
    constexpr std::size_t kMaxCovisibilityNeighbors = 10;

    const std::vector<std::shared_ptr<Frame>> neighbors = 
        keyframe->getBestCovisibilityKeyframes(kMaxCovisibilityNeighbors, kMinCovisibilityWeight);

    for (const auto& neighbor : neighbors)
    {
        if (neighbor == nullptr || !neighbor->isKeyframe() ||
            std::find(keyframes_.begin(), keyframes_.end(), neighbor) == keyframes_.end())
        {
            continue;
        }

        const int weight = keyframe->getConnectionWeight(neighbor->getId());
        if (weight < kMinCovisibilityWeight)
            continue;

        cv::Mat R_21, t_21;

        if (!computeRelativePoseConstraint(keyframe, neighbor, R_21, t_21))
            continue;

        addPoseGraphConstraint(keyframe, 
                               neighbor, 
                               weight, 
                               R_21, 
                               t_21, 
                               1.0, 
                               PoseGraphConstraintKind::COVISIBILITY);
    }
}

bool Map::addPoseGraphConstraint(const std::shared_ptr<Frame>& from_keyframe, 
                                const std::shared_ptr<Frame> &to_keyframe, 
                                int weight, 
                                const cv::Mat &R_21, 
                                const cv::Mat &t_21, 
                                double scale, 
                                PoseGraphConstraintKind kind)
{
    if (from_keyframe == nullptr || to_keyframe == nullptr || from_keyframe == to_keyframe ||
        weight <= 0 || !std::isfinite(scale) || scale <= 1e-8 ||
        R_21.rows != 3 || R_21.cols != 3 || t_21.rows != 3 || t_21.cols != 1)
    {
        return false;
    }

    pose_graph_constraints_.erase(
        std::remove_if(
            pose_graph_constraints_.begin(),
            pose_graph_constraints_.end(),
            [](const PoseGraphConstraint& constraint)
            {
                return constraint.from_keyframe.expired() || constraint.to_keyframe.expired();
            }),
        pose_graph_constraints_.end());

    cv::Mat stored_R;
    cv::Mat stored_t;
    R_21.convertTo(stored_R, CV_64F);
    t_21.convertTo(stored_t, CV_64F);

    for (auto& constraint : pose_graph_constraints_)
    {
        const std::shared_ptr<Frame> existing_from = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> existing_to = constraint.to_keyframe.lock();

        const bool same_endpoints = (existing_from == from_keyframe && existing_to == to_keyframe) ||
                                    (existing_from == to_keyframe && existing_to == from_keyframe);

        if (same_endpoints && constraint.kind == kind)
        {
            constraint.from_keyframe = from_keyframe;
            constraint.to_keyframe = to_keyframe;
            constraint.weight = weight;
            constraint.R_21 = stored_R;
            constraint.t_21 = stored_t;
            constraint.scale = scale;

            return true;
        }
    }

    pose_graph_constraints_.push_back(
        PoseGraphConstraint{from_keyframe, to_keyframe, weight, stored_R, stored_t, scale, kind});

    return true;
}

bool Map::hasPoseGraphConstraint(const std::shared_ptr<Frame>& from_keyframe, 
                                 const std::shared_ptr<Frame> &to_keyframe, 
                                 PoseGraphConstraintKind kind) const
{
    if (from_keyframe == nullptr || to_keyframe == nullptr || from_keyframe == to_keyframe)
        return false;

    for (const auto& constraint : pose_graph_constraints_)
    {
        if (constraint.kind != kind)
            continue;

        const std::shared_ptr<Frame> from = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> to = constraint.to_keyframe.lock();

        const bool same_endpoints = (from == from_keyframe && to == to_keyframe) ||
                                    (from == to_keyframe && to == from_keyframe);

        if (same_endpoints)
            return true;
    }

    return false;
}

} // namespace mini_orb_slam
