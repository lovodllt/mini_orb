#ifndef MINI_ORB_SLAM_INCLUDE_BOW_VOCABULARY_H_
#define MINI_ORB_SLAM_INCLUDE_BOW_VOCABULARY_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

namespace mini_orb_slam
{

using BowVector = DBoW2::BowVector;
using FeatureVector = DBoW2::FeatureVector;
using ORBVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

class BoWVocabulary
{
public:
    bool loadFromTextFile(const std::string& path);
    bool isLoaded() const { return vocabulary_ != nullptr; }

    bool transform(const cv::Mat& descriptors,
                   BowVector& bow_vector,
                   FeatureVector& feature_vector,
                   int levelsup = 4) const;

    double score(const BowVector& lhs, const BowVector& rhs) const;

private:
    static void changeStructure(const cv::Mat& plain, std::vector<cv::Mat>& out);

    std::shared_ptr<ORBVocabulary> vocabulary_;
};

} // namespace mini_orb_slam


#endif // MINI_ORB_SLAM_INCLUDE_BOW_VOCABULARY_H_
