#include "bow_vocabulary.h"

namespace mini_orb_slam
{

bool BoWVocabulary::loadFromTextFile(const std::string& path)
{
    std::shared_ptr<ORBVocabulary> vocabulary = std::make_shared<ORBVocabulary>();
    if (!vocabulary->loadFromTextFile(path))
        return false;

    vocabulary_ = vocabulary;
    return true;
}

bool BoWVocabulary::transform(const cv::Mat& descriptors,
                              BowVector& bow_vector,
                              FeatureVector& feature_vector,
                              int levelsup) const
{
    bow_vector.clear();
    feature_vector.clear();

    if (vocabulary_ == nullptr || descriptors.empty())
        return false;

    std::vector<cv::Mat> features;
    changeStructure(descriptors, features);
    if (features.empty())
        return false;

    vocabulary_->transform(features, bow_vector, feature_vector, levelsup);
    return true;
}

double BoWVocabulary::score(const BowVector& lhs, const BowVector& rhs) const
{
    if (vocabulary_ == nullptr)
        return 0.0;

    return vocabulary_->score(lhs, rhs);
}

void BoWVocabulary::changeStructure(const cv::Mat& plain, std::vector<cv::Mat>& out)
{
    out.clear();
    out.reserve(plain.rows);

    for (int i = 0; i < plain.rows; i++)
        out.push_back(plain.row(i));
}

} // namespace mini_orb_slam
