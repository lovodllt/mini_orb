#ifndef MINI_ORB_SLAM_INCLUDE_CAMERA_H_
#define MINI_ORB_SLAM_INCLUDE_CAMERA_H_

#include <string>
#include <vector>

#include <ros/ros.h>
#include <Eigen/Core>
#include <sensor_msgs/CameraInfo.h>
#include <opencv4/opencv2/core.hpp>

namespace mini_orb_slam
{

class Camera
{
public:
    Camera() = default;

    bool loadParams(ros::NodeHandle& nh);
    bool isValid() const { return is_valid_; }

    bool setCameraInfo(const sensor_msgs::CameraInfo& camera_info);

    bool undistrortImage(const cv::Mat& input, cv::Mat& output) const;

    Eigen::Vector3d Pixel2Camera(const Eigen::Vector2d& pixel, double depth) const;
    Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d& pc) const;

    bool projectCameraPoint(double x, double y, double z,
                            double& u, double& v) const;

    const std::string& getCameraName() const { return camera_name_; }
    const cv::Mat& getK() const { return K_; }
    const cv::Mat& getD() const { return D_; }

    int getImageWidth() const { return img_width_; }
    int getImageHeight() const { return img_height_; }

private:
    bool parseMatrix(const std::vector<double>& data, int row, int col, cv::Mat& matrix);
    bool cacheIntrinsics();

    bool setCalibration(int image_width, 
                        int image_height, 
                        const cv::Mat& raw_K, 
                        const cv::Mat& raw_D);

    bool buildUndistortMaps();

    bool is_valid_{false};
    int img_width_{0};
    int img_height_{0};

    std::string camera_name_;

    cv::Mat raw_K_;
    cv::Mat raw_D_;

    // undistorted camera intrinsics
    cv::Mat K_;
    cv::Mat D_;

    cv::Mat undistort_map1_;
    cv::Mat undistort_map2_;

    bool has_distortion_{false};

    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_CAMERA_H_
