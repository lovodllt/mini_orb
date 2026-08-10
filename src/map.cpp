#include "map.h"

#include <unordered_set>

namespace mini_orb_slam
{

namespace  
{

constexpr int kMinCovisibilityWeight = 30;
constexpr std::size_t kMaxCovisibilityNeighbors = 10;

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

Map::PoseGraphConstraintKey Map::makePoseGraphConstraintKey(
    const std::shared_ptr<Frame>& from_keyframe,
    const std::shared_ptr<Frame>& to_keyframe,
    PoseGraphConstraintKind kind)
{
    const std::size_t from_id = from_keyframe != nullptr ? from_keyframe->getId() : 0;
    const std::size_t to_id = to_keyframe != nullptr ? to_keyframe->getId() : 0;
    return PoseGraphConstraintKey{std::min(from_id, to_id), std::max(from_id, to_id), kind};
}

void Map::rebuildPoseGraphConstraintIndexes()
{
    pose_graph_constraint_indices_.clear();
    incident_constraint_indices_.clear();

    for (std::size_t index = 0; index < pose_graph_constraints_.size(); ++index)
    {
        const PoseGraphConstraint& constraint = pose_graph_constraints_[index];
        const std::shared_ptr<Frame> from_keyframe = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> to_keyframe = constraint.to_keyframe.lock();
        if (from_keyframe == nullptr || to_keyframe == nullptr)
            continue;

        pose_graph_constraint_indices_[makePoseGraphConstraintKey(
            from_keyframe, to_keyframe, constraint.kind)] = index;
        incident_constraint_indices_[from_keyframe->getId()].insert(index);
        incident_constraint_indices_[to_keyframe->getId()].insert(index);
    }
}

bool Map::removeExpiredPoseGraphConstraints()
{
    const std::size_t before = pose_graph_constraints_.size();
    pose_graph_constraints_.erase(
        std::remove_if(
            pose_graph_constraints_.begin(),
            pose_graph_constraints_.end(),
            [](const PoseGraphConstraint& constraint)
            {
                return constraint.from_keyframe.expired() || constraint.to_keyframe.expired();
            }),
        pose_graph_constraints_.end());

    const bool changed = pose_graph_constraints_.size() != before;
    if (changed)
        rebuildPoseGraphConstraintIndexes();
    return changed;
}

void Map::addKeyframe(const std::shared_ptr<Frame>& keyframe)
{
    if (keyframe == nullptr || !keyframe->isKeyframe())
        return;

    if (std::find(keyframes_.begin(), keyframes_.end(), keyframe) != keyframes_.end())
        return; 

    const std::shared_ptr<Frame> previous_keyframe = getLastKeyframe();
    keyframes_.push_back(keyframe);
    markModified();

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

void Map::reconcileCovisibilityConstraints(
    const std::vector<std::shared_ptr<Frame>>& keyframes)
{
    std::unordered_set<std::size_t> affected_ids;
    affected_ids.reserve(keyframes.size() * 2 + 1);
    for (const auto& keyframe : keyframes)
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
            affected_ids.insert(keyframe->getId());
    }
    if (affected_ids.empty())
        return;

    bool removed_constraints = removeExpiredPoseGraphConstraints();

    // The incident index already identifies every edge whose validity can
    // change when one of these keyframes changes. Avoid evaluating unrelated
    // sequential/covisibility/loop edges on every BA commit.
    std::unordered_set<std::size_t> affected_constraint_indices;
    affected_constraint_indices.reserve(affected_ids.size() * 16 + 1);
    for (const std::size_t keyframe_id : affected_ids)
    {
        const auto incident = incident_constraint_indices_.find(keyframe_id);
        if (incident == incident_constraint_indices_.end())
            continue;
        affected_constraint_indices.insert(incident->second.begin(),
                                           incident->second.end());
    }

    std::unordered_set<std::size_t> removed_constraint_indices;
    removed_constraint_indices.reserve(affected_constraint_indices.size());
    for (const std::size_t index : affected_constraint_indices)
    {
        if (index >= pose_graph_constraints_.size())
            continue;

        const PoseGraphConstraint& constraint = pose_graph_constraints_[index];
        if (constraint.kind != PoseGraphConstraintKind::COVISIBILITY)
            continue;

        const std::shared_ptr<Frame> from_keyframe = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> to_keyframe = constraint.to_keyframe.lock();
        if (from_keyframe == nullptr || to_keyframe == nullptr ||
            !from_keyframe->isKeyframe() || !to_keyframe->isKeyframe() ||
            from_keyframe->getConnectionWeight(to_keyframe->getId()) <
                kMinCovisibilityWeight)
        {
            removed_constraint_indices.insert(index);
        }
    }

    if (!removed_constraint_indices.empty())
    {
        std::vector<PoseGraphConstraint> retained_constraints;
        retained_constraints.reserve(pose_graph_constraints_.size() -
                                     removed_constraint_indices.size());
        for (std::size_t index = 0; index < pose_graph_constraints_.size(); ++index)
        {
            if (removed_constraint_indices.count(index) == 0)
                retained_constraints.push_back(std::move(pose_graph_constraints_[index]));
        }
        pose_graph_constraints_ = std::move(retained_constraints);
        rebuildPoseGraphConstraintIndexes();
        removed_constraints = true;
    }

    for (const auto& keyframe : keyframes)
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
            recordCovisibilityConstraints(keyframe);
    }

    if (removed_constraints)
        markModified();
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

    removeExpiredPoseGraphConstraints();

    cv::Mat stored_R;
    cv::Mat stored_t;
    R_21.convertTo(stored_R, CV_64F);
    t_21.convertTo(stored_t, CV_64F);

    const PoseGraphConstraintKey key =
        makePoseGraphConstraintKey(from_keyframe, to_keyframe, kind);
    const auto existing = pose_graph_constraint_indices_.find(key);
    if (existing != pose_graph_constraint_indices_.end())
    {
        PoseGraphConstraint& constraint = pose_graph_constraints_[existing->second];
        constraint.from_keyframe = from_keyframe;
        constraint.to_keyframe = to_keyframe;
        constraint.weight = weight;
        constraint.R_21 = stored_R;
        constraint.t_21 = stored_t;
        constraint.scale = scale;
        markModified();
        return true;
    }

    pose_graph_constraints_.push_back(
        PoseGraphConstraint{from_keyframe, to_keyframe, weight, stored_R, stored_t, scale, kind});
    const std::size_t index = pose_graph_constraints_.size() - 1;
    pose_graph_constraint_indices_[key] = index;
    incident_constraint_indices_[from_keyframe->getId()].insert(index);
    incident_constraint_indices_[to_keyframe->getId()].insert(index);

    markModified();

    return true;
}

bool Map::hasPoseGraphConstraint(const std::shared_ptr<Frame>& from_keyframe, 
                                 const std::shared_ptr<Frame> &to_keyframe, 
                                 PoseGraphConstraintKind kind) const
{
    if (from_keyframe == nullptr || to_keyframe == nullptr || from_keyframe == to_keyframe)
        return false;

    return pose_graph_constraint_indices_.count(
        makePoseGraphConstraintKey(from_keyframe, to_keyframe, kind)) != 0;
}

std::size_t Map::refreshPoseGraphMeasurements()
{
    return refreshPoseGraphMeasurements(std::vector<std::shared_ptr<Frame>>{});
}

std::size_t Map::refreshPoseGraphMeasurements(
    const std::vector<std::shared_ptr<Frame>>& dirty_keyframes)
{
    std::unordered_set<std::size_t> dirty_keyframe_ids;
    dirty_keyframe_ids.reserve(dirty_keyframes.size() * 2 + 1);
    for (const auto& keyframe : dirty_keyframes)
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
            dirty_keyframe_ids.insert(keyframe->getId());
    }

    const bool incremental = !dirty_keyframe_ids.empty();
    std::unordered_set<std::size_t> constraint_indices;
    if (incremental)
    {
        for (const std::size_t keyframe_id : dirty_keyframe_ids)
        {
            const auto incident = incident_constraint_indices_.find(keyframe_id);
            if (incident != incident_constraint_indices_.end())
                constraint_indices.insert(incident->second.begin(), incident->second.end());
        }
    }
    std::size_t refreshed_num = 0;

    const auto refresh_constraint = [&refreshed_num](PoseGraphConstraint& constraint)
    {
        if (constraint.kind == PoseGraphConstraintKind::LOOP)
            return;

        const std::shared_ptr<Frame> from_keyframe = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> to_keyframe = constraint.to_keyframe.lock();
        cv::Mat relative_R;
        cv::Mat relative_t;
        if (from_keyframe == nullptr || to_keyframe == nullptr ||
            !computeRelativePoseConstraint(from_keyframe, to_keyframe,
                                           relative_R, relative_t))
        {
            return;
        }

        constraint.R_21 = relative_R;
        constraint.t_21 = relative_t;
        constraint.scale = 1.0;
        refreshed_num++;
    };

    if (incremental)
    {
        for (const std::size_t index : constraint_indices)
        {
            if (index < pose_graph_constraints_.size())
                refresh_constraint(pose_graph_constraints_[index]);
        }
    }
    else
    {
        for (auto& constraint : pose_graph_constraints_)
            refresh_constraint(constraint);
    }

    if (refreshed_num > 0)
        markModified();

    return refreshed_num;
}

} // namespace mini_orb_slam
