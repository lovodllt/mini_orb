#include <algorithm>

#include "keyframe_database.h"

namespace mini_orb_slam
{

KeyframeDatabase::KeyframeDatabase(const std::shared_ptr<BoWVocabulary>& vocabulary)
    : vocabulary_(vocabulary) {}

void KeyframeDatabase::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    keyframe_ids_.clear();
}

bool KeyframeDatabase::addKeyframe(const std::shared_ptr<Frame>& keyframe)
{
    if (vocabulary_ == nullptr || !vocabulary_->isLoaded())
        return false;

    if (keyframe == nullptr || !keyframe->isKeyframe() || !keyframe->hasBoW())
        return false;

    std::lock_guard<std::mutex> lock(mutex_);

    if (!keyframe_ids_.insert(keyframe->getId()).second)
        return false;

    keyframes_.push_back(keyframe);
    return true;
}

std::vector<KeyframeQueryResult> KeyframeDatabase::query(
    const std::shared_ptr<Frame>& frame, 
    std::size_t max_result,
    double min_score) const
{
    std::vector<KeyframeQueryResult> results;

    if (vocabulary_ == nullptr || !vocabulary_->isLoaded() ||
        frame == nullptr || !frame->hasBoW())
    {
        return results;
    }

    std::vector<std::weak_ptr<Frame>> keyframe_snapshot;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        keyframe_snapshot = keyframes_;
    }

    results.reserve(keyframe_snapshot.size());

    for (const auto& weak_frame : keyframe_snapshot)
    {
        const std::shared_ptr<Frame> keyframe = weak_frame.lock();
        if (keyframe == nullptr || !keyframe->isKeyframe() || !keyframe->hasBoW())
            continue;

        if (keyframe->getId() == frame->getId())
            continue;

        const double score = vocabulary_->score(frame->getBowVector(), 
                                                keyframe->getBowVector());

        if (score < min_score)
            continue;

        results.push_back({keyframe, score});
    }

    std::sort(results.begin(), results.end(),
                [](const KeyframeQueryResult& a, const KeyframeQueryResult& b)
                {
                    if (a.score != b.score)
                        return a.score > b.score;

                    return a.keyframe->getId() < b.keyframe->getId();
                });

    if (max_result > 0 && results.size() > max_result)
        results.resize(max_result);

    return results;
}

} // namespace mini_orb_slam
