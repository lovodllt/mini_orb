#ifndef MINI_ORB_SLAM_INCLUDE_FRAME_H_
#define MINI_ORB_SLAM_INCLUDE_FRAME_H_

#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "camera.h"
#include "feature.h"
#include "bow_vocabulary.h"

namespace mini_orb_slam
{

class Frame
{
public:
    Frame(std::size_t id, double timestamp, const cv::Mat& img, const std::shared_ptr<Camera>& camera);

    void setFeatures(const std::vector<std::shared_ptr<Feature>>& features, const cv::Mat& descriptors);
    void clearFeatures();

    void setPose(const cv::Mat& R_cw, const cv::Mat& t_cw);
    void copyPose(cv::Mat& R_cw, cv::Mat& t_cw) const;
    const cv::Mat& getRcw() const { return R_cw_; }
    const cv::Mat& getTcw() const { return t_cw_; }
    cv::Mat getRwc() const;
    cv::Point3d getCameraCenter() const;

    bool isKeyframe() const { return is_keyframe_; }
    void setKeyframe(bool is_keyframe) { is_keyframe_ = is_keyframe; }

    std::size_t getId() const { return id_; }
    double getTimestamp() const { return timestamp_; }
    const cv::Mat& getImg() const { return img_; }
    const std::shared_ptr<Camera>& getCamera() const { return camera_; }

    bool hasFeatures() const { return !features_.empty() && !descriptors_.empty(); }
    std::size_t getFeatureNum() const { return features_.size(); }

    const std::vector<std::shared_ptr<Feature>>& getFeatures() const { return features_; }
    const std::vector<cv::KeyPoint>& getKeypoints() const { return keypoints_; }
    const cv::Mat& getDescriptors() const { return descriptors_; }
    const std::vector<int>& getPyramidLevels() const { return pyramid_levels_; }

    std::vector<int> getFeatureIndicesInArea(const cv::Point2d& pt,
                                             float radius,
                                             int min_level,
                                             int max_level) const;

    void updateConnections();

    std::vector<std::shared_ptr<Frame>> copyConnectedKeyframes(int min_weight = 1) const;
    std::vector<std::shared_ptr<Frame>> copyBestCovisibilityKeyframes(std::size_t max_num, 
                                                                      int min_weight = 1) const;

    int copyConnectionWeight(std::size_t keyframe_id) const;

    std::vector<std::shared_ptr<Frame>> getConnectedKeyframes(int min_weight = 1) const;
    std::vector<std::shared_ptr<Frame>> getBestCovisibilityKeyframes(std::size_t max_num,
                                                                     int min_weight = 1) const;

    int getConnectionWeight(std::size_t keyframe_id) const;

    void computeBoW(const std::shared_ptr<BoWVocabulary>& vocabulary);
    void clearBoW();

    bool hasBoW() const { return has_bow_; }
    const BowVector& getBowVector() const { return bow_vector_; }
    const FeatureVector& getFeatureVector() const { return feature_vector_; }

private:
    void updateFeatureCache();

    void buildFeatureGrid();
    bool pointToGridCell(const cv::Point2f& pt, int& cell_x, int& cell_y) const;

    static constexpr int kGridCellSizePx = 40;
    int grid_cols_{0};
    int grid_rows_{0};
    float grid_cell_width_inv_{0.0f};
    float grid_cell_height_inv_{0.0f};
    std::vector<std::vector<int>> feature_grid_;

    struct KeyframeConnection
    {
        std::weak_ptr<Frame> keyframe;
        int weight{0};
    };

    std::unordered_map<std::size_t, KeyframeConnection> connected_keyframes_;

    mutable std::mutex pose_mutex_;
    mutable std::mutex connection_mutex_;

    std::size_t id_{0};
    double timestamp_{0.0};
    cv::Mat img_;
    std::shared_ptr<Camera> camera_;

    std::vector<std::shared_ptr<Feature>> features_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<int> pyramid_levels_;

    cv::Mat R_cw_;
    cv::Mat t_cw_;
    bool is_keyframe_{false};

    BowVector bow_vector_;
    FeatureVector feature_vector_;
    bool has_bow_{false};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_FRAME_H_
