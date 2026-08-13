#ifndef MINI_ORB_SLAM_INCLUDE_KEYFRAME_DATABASE_H_
#define MINI_ORB_SLAM_INCLUDE_KEYFRAME_DATABASE_H_

#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "bow_vocabulary.h"
#include "frame.h"

namespace mini_orb_slam
{

struct KeyframeQueryResult
{
    std::shared_ptr<Frame> keyframe;
    double score{0.0};
};

class KeyframeDatabase
{
public:
    explicit KeyframeDatabase(const std::shared_ptr<BoWVocabulary>& vocabulary);

    void clear();

    bool addKeyframe(const std::shared_ptr<Frame>& keyframe);

    std::vector<KeyframeQueryResult> query(const std::shared_ptr<Frame>& frame,
                                           std::size_t max_result,
                                           double min_score = 0.0) const;

private:
    std::shared_ptr<BoWVocabulary> vocabulary_;
    std::vector<std::weak_ptr<Frame>> keyframes_;
    std::unordered_set<std::size_t> keyframe_ids_;

    mutable std::mutex mutex_;
};

} // namespace mini_orb_slam


#endif // MINI_ORB_SLAM_INCLUDE_KEYFRAME_DATABASE_H_
