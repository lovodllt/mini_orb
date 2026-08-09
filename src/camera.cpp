#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"

namespace mini_orb_slam
{
   
bool Camera::loadParams(ros::NodeHandle& nh)
{
    nh.param("camera_name", camera_name_, std::string("camera"));

    int image_width = 0;
    int image_height = 0;

    nh.param("image_width", image_width, 0);
    nh.param("image_height", image_height, 0);

    std::vector<double> K_data;
    std::vector<double> D_data;

    if (!nh.getParam("K", K_data))
        nh.getParam("camera_matrix/data", K_data);

    if (!nh.getParam("D", D_data))
        nh.getParam("distortion_coefficients/data", D_data);

    cv::Mat raw_k, raw_D;

    if (!parseMatrix(K_data, 3, 3, raw_k))
    {
        is_valid_ = false;
        return false;
    }

    // EuRoC cam0 uses the four-parameter plumb_bob model. Keep the
    // coefficient count intact so setCalibration can validate all OpenCV
    // distortion layouts (4, 5, 8, 12 and 14).
    const int distortion_num = static_cast<int>(D_data.size());
    if (distortion_num != 4 && distortion_num != 5 && distortion_num != 8 &&
        distortion_num != 12 && distortion_num != 14)
    {
        ROS_ERROR("Invalid distortion coefficients size: %d", distortion_num);
        is_valid_ = false;
        return false;
    }

    if (!parseMatrix(D_data, distortion_num, 1, raw_D))
    {
        is_valid_ = false;
        return false;
    }

    return setCalibration(image_width, image_height, raw_k, raw_D);
}

bool Camera::setCameraInfo(const sensor_msgs::CameraInfo& camera_info)
{
    if (camera_info.width == 0 || camera_info.height == 0)
    {
        ROS_ERROR("CameraInfo has invalid image size.");
        return false;
    }

    if (!camera_info.distortion_model.empty() &&
        camera_info.distortion_model != "plumb_bob" &&
        camera_info.distortion_model != "rational_polynomial")
    {
        ROS_ERROR_STREAM("Unsupported distortion model: " << camera_info.distortion_model);
        return false;
    }

    cv::Mat raw_K(3, 3, CV_64F);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            raw_K.at<double>(r, c) = camera_info.K[r * 3 + c];
        }
    }

    const int distortion_num = camera_info.D.empty() ? 5 : camera_info.D.size();
    cv::Mat raw_D = cv::Mat::zeros(distortion_num, 1, CV_64F);

    for (int i = 0; i < camera_info.D.size(); i++)
        raw_D.at<double>(i, 0) = camera_info.D[i];

    return setCalibration(camera_info.width, camera_info.height, raw_K, raw_D);
}

bool Camera::setCalibration(int image_width,
                            int image_height, 
                            const cv::Mat& raw_K, 
                            const cv::Mat& raw_D)
{
    if (image_width <= 0 || image_height <= 0 ||
        raw_K.rows != 3 || raw_K.cols != 3)
    {
        ROS_ERROR("Invalid camera calibration dimensions.");
        is_valid_ = false;
        return false;
    }

    const int distortion_num = static_cast<int>(raw_D.total());
    const bool valid_distortion_num = 
        distortion_num == 4 || distortion_num == 5 || distortion_num == 8 || distortion_num == 12 || distortion_num == 14;

    if (!valid_distortion_num)
    {
        ROS_ERROR("Invalid distortion coefficients size: %d", distortion_num);
        is_valid_ = false;
        return false;
    }

    raw_K.convertTo(raw_K_, CV_64F);

    cv::Mat raw_D_64;
    raw_D.convertTo(raw_D_64, CV_64F);
    raw_D_ = raw_D_64.reshape(1, distortion_num).clone();

    img_width_ = image_width;
    img_height_ = image_height;

    K_ = raw_K_.clone();
    D_ = cv::Mat::zeros(raw_D_.rows, 1, CV_64F);

    has_distortion_ = cv::norm(raw_D_, cv::NORM_INF) > 1e-8;

    if (!cacheIntrinsics() || !buildUndistortMaps())
    {
        is_valid_ = false;
        return false;
    }

    is_valid_ = true;
    return true;
}

bool Camera::buildUndistortMaps()
{
    undistort_map1_.release();
    undistort_map2_.release();

    if (!has_distortion_)
        return true;

    cv::initUndistortRectifyMap(raw_K_, 
                                raw_D_, 
                                cv::Mat::eye(3, 3, CV_64F), 
                                K_, 
                                cv::Size(img_width_, img_height_), 
                                CV_16SC2, 
                                undistort_map1_, 
                                undistort_map2_);

    return !undistort_map1_.empty() && !undistort_map2_.empty();
}

bool Camera::undistrortImage(const cv::Mat& input, cv::Mat &output) const
{
    output.release();

    if (!is_valid_ || input.empty() ||
        input.rows != img_height_ || input.cols != img_width_)
    {
        return false;
    }

    if (!has_distortion_)
    {
        output = input.clone();
        return true;
    }

    cv::remap(input, output, undistort_map1_, undistort_map2_, cv::INTER_LINEAR);

    return !output.empty();
}

bool Camera::parseMatrix(const std::vector<double>& data, int row, int col, cv::Mat& matrix)
{
    if (static_cast<int>(data.size()) != row * col)
    {
        ROS_ERROR("Data size does not match expected matrix dimensions: expected %d, got %lu", row * col, data.size());
        return false;
    }

    matrix = cv::Mat(row, col, CV_64F);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            matrix.at<double>(i, j) = data[i * col + j];
        }
    }

    return true;
}

bool Camera::cacheIntrinsics()
{
    if (K_.rows != 3 || K_.cols != 3)
        return false;

    fx_ = K_.at<double>(0, 0);
    fy_ = K_.at<double>(1, 1);
    cx_ = K_.at<double>(0, 2);
    cy_ = K_.at<double>(1, 2);

    if (fx_ <= 0 || fy_ <= 0)
    {
        ROS_ERROR("Invalid focal lengths fx=%f, fy=%f", fx_, fy_);
        return false;
    }

    return true;
}

Eigen::Vector3d Camera::Pixel2Camera(const Eigen::Vector2d& pixel, double depth) const
{
    Eigen::Vector3d pc;
    pc(0) = (pixel(0) - cx_) * depth / fx_;
    pc(1) = (pixel(1) - cy_) * depth / fy_;
    pc(2) = depth;

    return pc;
}

Eigen::Vector2d Camera::Camera2Pixel(const Eigen::Vector3d& pc) const
{
    Eigen::Vector2d pixel(0.0, 0.0);

    if (pc(2) <= 0)
    {
        ROS_ERROR("Invalid depth value: %f", pc(2));
        return pixel;
    }

    pixel(0) = fx_ * pc(0) / pc(2) + cx_;
    pixel(1) = fy_ * pc(1) / pc(2) + cy_;

    return pixel;
}

} // namespace mini_orb_slam
