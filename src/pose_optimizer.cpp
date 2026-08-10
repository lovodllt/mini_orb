#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <opencv2/calib3d.hpp>

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/edge_project_xyz.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include "pose_optimizer.h"

namespace mini_orb_slam
{

namespace
{

cv::Matx33d matToMatx33d(const cv::Mat& mat)
{
    cv::Matx33d result = cv::Matx33d::zeros();

    if (mat.rows != 3 || mat.cols != 3)
        return result;

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
            result(r, c) = mat.at<double>(r, c);
    }

    return result;
}

bool cvToEigenRotation(const cv::Mat& R_cv, Eigen::Matrix3d& R_eigen)
{
    if (R_cv.rows != 3 || R_cv.cols != 3)
        return false;

    cv::Mat R_64;
    R_cv.convertTo(R_64, CV_64F);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
            R_eigen(r, c) = R_64.at<double>(r, c);
    }

    return R_eigen.allFinite();
}

bool cvToEigenTranslation(const cv::Mat& t_cv, Eigen::Vector3d& t_eigen)
{
    if (t_cv.rows != 3 || t_cv.cols != 1)
        return false;

    cv::Mat t_64;
    t_cv.convertTo(t_64, CV_64F);

    for (int r = 0; r < 3; r++)
        t_eigen(r) = t_64.at<double>(r, 0);

    return t_eigen.allFinite();
}

bool makeSim3(const cv::Mat& R_cv, const cv::Mat& t_cv, double scale, g2o::Sim3& sim3)
{
    if (!std::isfinite(scale) || scale <= 1e-8)
        return false;

    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    if (!cvToEigenRotation(R_cv, R) || !cvToEigenTranslation(t_cv, t))
        return false;

    sim3 = g2o::Sim3(R, t, scale);
    return true;
}

cv::Mat eigenToCvRotation(const Eigen::Matrix3d& R)
{
    cv::Mat result(3, 3, CV_64F);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
            result.at<double>(r, c) = R(r, c);
    }

    return result;
}

cv::Mat eigenToCvTranslation(const Eigen::Vector3d& t)
{
    return (cv::Mat_<double>(3, 1) << t(0), t(1), t(2));
}

Eigen::Vector3d pointToEigen(const cv::Point3d& p)
{
    return Eigen::Vector3d(p.x, p.y, p.z);
}

struct PoseGraphVertexState
{
    std::shared_ptr<Frame> keyframe;
    cv::Mat R_cw_before;
    cv::Mat t_cw_before;
    g2o::VertexSim3Expmap* vertex{nullptr};
};

} // namespace

PoseOptimizer::PoseOptimizer(const std::shared_ptr<Camera>& camera,
                             double scale_factor,
                             int levels_num) 
    : camera_(camera), 
      scale_factor_(scale_factor > 1.0 ? scale_factor : 1.2), 
      levels_num_(levels_num > 0 ? levels_num : 8) {}

bool PoseOptimizer::projectWorldPointToFrame(const std::shared_ptr<Frame>& frame,
                                             const cv::Point3d& point_world,
                                             cv::Point2d& projected_point,
                                             cv::Point3d& point_camera) const
{
    if (camera_ == nullptr ||
        frame == nullptr)
    {
        return false;
    }

    cv::Mat R_cw;
    cv::Mat t_cw;
    frame->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return false;

    const cv::Mat point_world_mat = 
        (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);

    const cv::Mat point_camera_mat = R_cw * point_world_mat + t_cw;

    const double x = point_camera_mat.at<double>(0, 0);
    const double y = point_camera_mat.at<double>(1, 0);
    const double z = point_camera_mat.at<double>(2, 0);

    if (z <= 1e-6)
        return false;

    const cv::Mat& K = camera_->getK();
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    projected_point.x = fx * x / z + cx;
    projected_point.y = fy * y / z + cy;
    point_camera = cv::Point3d(x, y, z);

    return true;
}

cv::Matx23d PoseOptimizer::computeProjectionJacobian(const cv::Point3d& point_camera) const
{
    cv::Matx23d J = cv::Matx23d::zeros();

    if (camera_ == nullptr || point_camera.z <= 1e-6)
        return J;

    const cv::Mat& K = camera_->getK();
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);

    const double x = point_camera.x;
    const double y = point_camera.y;
    const double z = point_camera.z;
    const double z2 = z * z;

    J(0, 0) = fx / z;
    J(0, 1) = 0.0;
    J(0, 2) = -fx * x / z2;

    J(1, 0) = 0.0;
    J(1, 1) = fy / z;
    J(1, 2) = -fy * y / z2;

    return J;
}

cv::Matx33d PoseOptimizer::skewSymmetric(const cv::Point3d& p) const
{
    return cv::Matx33d(0.0, -p.z,  p.y,
                       p.z,  0.0, -p.x,
                      -p.y,  p.x,  0.0);
}

bool PoseOptimizer::linearizeObservation(const LocalBAObservation& observation,
                                         const cv::Point3d& point_world,
                                         cv::Vec2d& residual,
                                         cv::Matx<double, 2, 6>& J_pose,
                                         cv::Matx<double, 2, 3>& J_point) const
{
    residual = cv::Vec2d(0.0, 0.0);
    J_pose = cv::Matx<double, 2, 6>::zeros();
    J_point = cv::Matx<double, 2, 3>::zeros();

    if (observation.keyframe == nullptr)
        return false;

    cv::Point2d projected_point;
    cv::Point3d point_camera;
    if (!projectWorldPointToFrame(observation.keyframe,
                                  point_world,
                                  projected_point,
                                  point_camera))
    {
        return false;
    }

    residual[0] = static_cast<double >(observation.img_point.x) - projected_point.x;
    residual[1] = static_cast<double >(observation.img_point.y) - projected_point.y;
    
    const cv::Matx23d J_proj = computeProjectionJacobian(point_camera);

    cv::Mat R_cw;
    cv::Mat t_cw;
    observation.keyframe->copyPose(R_cw, t_cw);

    if (R_cw.empty() || t_cw.empty())
        return false;

    const cv::Matx33d R_cw_mat = matToMatx33d(R_cw);

    const cv::Matx33d skew = skewSymmetric(point_camera);

    J_point = -(J_proj * R_cw_mat);

    const cv::Matx23d J_trans = -J_proj;
    const cv::Matx23d J_rot = J_proj * skew;

    for (int r = 0; r < 2; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            J_pose(r, c) = J_rot(r, c);
            J_pose(r, c + 3) = J_trans(r, c);
        }
    }

    return true;
}

double PoseOptimizer::computeMeanReprojectionError(const std::vector<cv::Point3d>& object_points,
                                                   const std::vector<cv::Point2f>& img_points,
                                                   const cv::Mat& rvec,
                                                   const cv::Mat& tvec) const
{
    if (camera_ == nullptr ||
        object_points.empty() ||
        object_points.size() != img_points.size() ||
        rvec.empty() || 
        tvec.empty())
    {
        return 0.0;
    }

    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points,
                      rvec,
                      tvec,
                      camera_->getK(),
                      cv::Mat(),
                      projected_points);

    if (projected_points.size() != img_points.size())
        return 0.0;

    double total_error = 0.0;
    for (std::size_t i = 0; i < projected_points.size(); i++)
    {
        const cv::Point2d observed_point(img_points[i].x, img_points[i].y);
        total_error += cv::norm(projected_points[i] - observed_point);
    }

    return total_error / projected_points.size();
}

double PoseOptimizer::computeMeanPointReprojectionError(
    const cv::Point3d& point_world,
    const std::vector<std::shared_ptr<Frame>>& frames,
    const std::vector<cv::Point2f>& img_points) const
{
    if (camera_ == nullptr || frames.empty() || frames.size() != img_points.size())
        return 0.0;

    std::vector<cv::Point3d> object_points(1, point_world);
    double total_error = 0.0;
    int valid_num = 0;

    for (std::size_t i = 0; i <frames.size(); i++)
    {
        const std::shared_ptr<Frame> frame = frames[i];
        if (frame == nullptr)
            continue;

        cv::Mat R_cw;
        cv::Mat t_cw;
        frame->copyPose(R_cw, t_cw);

        if (R_cw.empty() || t_cw.empty())
            continue;

        cv::Mat rvec;
        cv::Rodrigues(R_cw, rvec);

        std::vector<cv::Point2d> projected_points;
        cv::projectPoints(object_points,
                            rvec,
                            t_cw,
                            camera_->getK(),
                            cv::Mat(),
                            projected_points);

        if (projected_points.size() != 1)
            continue;

        const cv::Point2d observed_point(img_points[i].x, img_points[i].y);
        total_error += cv::norm(projected_points[0] - observed_point);
        valid_num++;
    }

    if (valid_num <= 0)
        return 0.0;

    return total_error / valid_num;
}

double PoseOptimizer::computeObservationReprojectionError(const LocalBAObservation& observation) const
{
    if (camera_ == nullptr ||
        observation.keyframe == nullptr ||
        observation.map_point == nullptr ||
        observation.map_point->isBad())
    {
        return -1.0;
    }

    cv::Point2d projected_point;
    cv::Point3d camera_point;
    if (!projectWorldPointToFrame(observation.keyframe,
                                  observation.map_point->getPos(),
                                  projected_point,
                                  camera_point))
    {
        return -1.0;
    }

    return cv::norm(projected_point - cv::Point2d(observation.img_point));
}

double PoseOptimizer::computeMeanObservationReprojectionError(
    const std::vector<LocalBAObservation>& observations) const
{
    if (observations.empty())
        return 0.0;

    double total_error = 0.0;
    int valid_num = 0;

    for (const auto& observation : observations)
    {
        const double error = computeObservationReprojectionError(observation);
        if (error < 0.0)
            continue;

        total_error += error;
        valid_num++;
    }

    if (valid_num <= 0)
        return 0.0;

    return total_error / valid_num;
}

std::vector<PoseOptimizer::LocalBAObservation> PoseOptimizer::selectInlierObservations(
    const std::vector<LocalBAObservation>& observations,
    double max_reproj_error) const
{
    std::vector<LocalBAObservation> inliers;
    inliers.reserve(observations.size());

    constexpr double kLevelScaleFactor = 1.2;

    for (const auto& observation : observations)
    {
        const double error = computeObservationReprojectionError(observation);
        if (error < 0.0)
            continue;

        const int level = std::max(0, observation.feature_level);
        const double level_threshold = 
            max_reproj_error * std::pow(kLevelScaleFactor, static_cast<int>(level));

        if (error > level_threshold)
            continue; 

        inliers.push_back(observation);
    }

    return inliers;
}

std::vector<PoseOptimizer::LocalBAObservation> PoseOptimizer::collectObservationsForKeyframe(
    const std::vector<LocalBAObservation>& observations,
    const std::shared_ptr<Frame>& keyframe) const
{
    std::vector<LocalBAObservation> keyframe_observations;
    if (keyframe == nullptr)
        return keyframe_observations;

    keyframe_observations.reserve(observations.size());

    for (const auto& observation : observations)
    {
        if (observation.keyframe == keyframe &&
            observation.map_point != nullptr &&
            !observation.map_point->isBad())
        {
            keyframe_observations.push_back(observation);
        }
    }

    return keyframe_observations;
}

std::vector<PoseOptimizer::LocalBAObservation> PoseOptimizer::collectObservationsForMapPoint(
    const std::vector<LocalBAObservation>& observations,
    const std::shared_ptr<MapPoint>& map_point) const
{
    std::vector<LocalBAObservation> map_point_observations;
    if (map_point == nullptr || map_point->isBad())
        return map_point_observations;

    map_point_observations.reserve(observations.size());

    for (const auto& observation : observations)
    {
        if (observation.map_point == map_point &&
            observation.keyframe != nullptr &&
            observation.keyframe->isKeyframe())
        {
            map_point_observations.push_back(observation);
        }
    }

    return map_point_observations;
}

double PoseOptimizer::computeTotalBAReprojectionError(
    const LocalBAContext& context) const
{
    if (context.observations.empty())
        return 0.0;

    double total_error = 0.0;
    int valid_num = 0;

    for (const auto& observation : context.observations)
    {
        const double error = computeObservationReprojectionError(observation);
        if (error < 0.0)
            continue;

        total_error += error;
        valid_num++;
    }

    if (valid_num <= 0)
        return 0.0;

    return total_error / valid_num;
}

bool PoseOptimizer::validateLocalBACandidate(const PnPResult& tracking_seed, 
                                             const cv::Mat& candidate_R_cw,
                                             const cv::Mat& candidate_t_cw,
                                             const std::unordered_map<std::size_t, cv::Point3d>&
                                                 candidate_map_point_positions,
                                             double& seed_reproj_error,
                                             double& candidate_seed_reproj_error) const
{
    seed_reproj_error = 0.0;
    candidate_seed_reproj_error = 0.0;

    if (camera_ == nullptr || !tracking_seed.success ||
        tracking_seed.inlier_indices.empty() ||
        tracking_seed.object_points.size() != tracking_seed.img_points.size() ||
        tracking_seed.object_points.size() != tracking_seed.candidate_map_points.size() ||
        candidate_R_cw.rows != 3 || candidate_R_cw.cols != 3 ||
        candidate_t_cw.rows != 3 || candidate_t_cw.cols != 1)
    {
        return false;
    }

    std::vector<cv::Point3d> seed_object_points;
    std::vector<cv::Point3d> candidate_object_points;
    std::vector<cv::Point2f> seed_img_points;
    seed_object_points.reserve(tracking_seed.inlier_indices.rows);
    candidate_object_points.reserve(tracking_seed.inlier_indices.rows);
    seed_img_points.reserve(tracking_seed.inlier_indices.rows);

    for (int i = 0; i < tracking_seed.inlier_indices.rows; i++)
    {
        const int index = tracking_seed.inlier_indices.at<int>(i, 0);
        if (index < 0 || index >= tracking_seed.object_points.size())
            continue;

        const std::shared_ptr<MapPoint>& map_point = tracking_seed.candidate_map_points[index];
        if (map_point == nullptr)
            continue;

        const auto candidate_point_it = candidate_map_point_positions.find(map_point->getId());
        seed_object_points.push_back(tracking_seed.object_points[index]);
        candidate_object_points.push_back(candidate_point_it == candidate_map_point_positions.end()
            ? tracking_seed.object_points[index]
            : candidate_point_it->second);
        seed_img_points.push_back(tracking_seed.img_points[index]);
    }

    constexpr std::size_t kMinValidationInliers = 20;
    if (seed_object_points.size() < kMinValidationInliers ||
        candidate_object_points.size() != seed_object_points.size() ||
        tracking_seed.R.rows != 3 || tracking_seed.R.cols != 3 ||
        tracking_seed.tvec.rows != 3 || tracking_seed.tvec.cols != 1)
    {
        return false;
    }

    cv::Mat seed_R;
    cv::Mat seed_rvec;
    cv::Mat candidate_R;
    cv::Mat candidate_rvec;
    cv::Mat candidate_t;

    tracking_seed.R.convertTo(seed_R, CV_64F);
    candidate_R_cw.convertTo(candidate_R, CV_64F);
    candidate_t_cw.convertTo(candidate_t, CV_64F);

    try
    {
        cv::Rodrigues(seed_R, seed_rvec);
        cv::Rodrigues(candidate_R, candidate_rvec);
    }
    catch (const cv::Exception&)
    {
        return false;
    }

    for (const auto& point : candidate_object_points)
    {
        const cv::Mat point_world = 
            (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
        const cv::Mat point_camera = candidate_R * point_world + candidate_t;

        if (point_camera.at<double>(2, 0) <= 1e-6)
            return false;
    }

    seed_reproj_error = computeMeanReprojectionError(seed_object_points,
                                                     seed_img_points,
                                                     tracking_seed.rvec,
                                                     tracking_seed.tvec);

    candidate_seed_reproj_error = computeMeanReprojectionError(candidate_object_points,
                                                               seed_img_points,
                                                               candidate_rvec,
                                                               candidate_t);

    if (!std::isfinite(seed_reproj_error) || seed_reproj_error <= 0.0 ||
        !std::isfinite(candidate_seed_reproj_error) || candidate_seed_reproj_error <= 0.0)
    {
        return false;
    }

    // Current-project reprojection must be evaluated against the candidate
    // point estimates from the same BA transaction, not the pre-BA map.
    constexpr double kMaxSeedErrorIncreasePx = 0.25;
    constexpr double kMaxSeedErrorIncreaseRatio = 1.10;

    const double max_allowed_error = 
        seed_reproj_error + 
        std::max(kMaxSeedErrorIncreasePx, 
                 kMaxSeedErrorIncreaseRatio * seed_reproj_error - seed_reproj_error);

    return (candidate_seed_reproj_error <= max_allowed_error);
}

PnPResult PoseOptimizer::optimize(const PnPResult& input_result) const
{
    return optimizeImpl(input_result, true);
}

PnPResult PoseOptimizer::optimizeWithPosePrior(const PnPResult& input_result) const
{
    return optimizeImpl(input_result, false);
}

PnPResult PoseOptimizer::optimizeImpl(const PnPResult& input_result,
                                      bool run_pnp_ransac) const
{
    PnPResult result = input_result;

    if (camera_ == nullptr ||
        result.object_points.size() < 6 ||
        result.object_points.size() != result.img_points.size())
    {
        return {};
    }

    auto normalizePoseVector = [](cv::Mat& vec) -> bool
    {
        if (vec.empty())
            return false;

        if (vec.rows == 1 && vec.cols == 1)
            vec = vec.t();

        vec.convertTo(vec, CV_64F);
        return vec.rows == 3 && vec.cols == 1;
    };

    const bool has_inital_pose =
        normalizePoseVector(result.rvec) && normalizePoseVector(result.tvec);

    if (run_pnp_ransac)
    {
        // OpenCV is used only to obtain a robust initial pose. The accepted
        // pose and inlier set come from the four-round g2o refinement below.
        const int ransac_method = has_inital_pose ? cv::SOLVEPNP_ITERATIVE
                                                   : cv::SOLVEPNP_EPNP;
        bool success = false;
        try
        {
            success = cv::solvePnPRansac(
                          result.object_points,
                          result.img_points,
                          camera_->getK(),
                          cv::Mat(),
                          result.rvec,
                          result.tvec,
                          has_inital_pose,
                          100,
                          8.0,
                          0.99,
                          result.inlier_indices,
                          ransac_method);
        }
        catch (const cv::Exception&)
        {
            return {};
        }

        if (!success || result.inlier_indices.rows < 6)
            return {};
    }
    else if (!has_inital_pose)
    {
        return {};
    }

    cv::Mat initial_R;
    cv::Rodrigues(result.rvec, initial_R);
    Eigen::Matrix3d initial_R_eigen;
    Eigen::Vector3d initial_t_eigen;
    if (!cvToEigenRotation(initial_R, initial_R_eigen) ||
        !cvToEigenTranslation(result.tvec, initial_t_eigen))
    {
        return {};
    }

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));

    auto* pose_vertex = new g2o::VertexSE3Expmap();
    pose_vertex->setId(0);
    pose_vertex->setEstimate(g2o::SE3Quat(initial_R_eigen, initial_t_eigen));
    if (!optimizer.addVertex(pose_vertex))
    {
        delete pose_vertex;
        return {};
    }

    struct PoseEdge
    {
        int input_index{-1};
        g2o::EdgeSE3ProjectXYZOnlyPose* edge{nullptr};
    };
    std::vector<PoseEdge> edges;
    edges.reserve(result.object_points.size());

    const cv::Mat& K = camera_->getK();
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    constexpr double kMonoChi2Threshold = 5.991;
    const double huber_delta = std::sqrt(kMonoChi2Threshold);

    for (int i = 0; i < static_cast<int>(result.object_points.size()); ++i)
    {
        const cv::Point3d& point = result.object_points[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;

        auto* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        edge->setVertex(0, pose_vertex);
        edge->setMeasurement(Eigen::Vector2d(result.img_points[i].x,
                                             result.img_points[i].y));

        double inv_sigma2 = 1.0;
        if (i < static_cast<int>(result.candidate_features.size()) &&
            result.candidate_features[i] != nullptr)
        {
            const int level = std::clamp(result.candidate_features[i]->getLevel(),
                                         0, levels_num_ - 1);
            inv_sigma2 = std::pow(scale_factor_, -2.0 * level);
        }
        edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);
        auto* robust_kernel = new g2o::RobustKernelHuber();
        robust_kernel->setDelta(huber_delta);
        edge->setRobustKernel(robust_kernel);
        edge->fx = fx;
        edge->fy = fy;
        edge->cx = cx;
        edge->cy = cy;
        edge->Xw = Eigen::Vector3d(point.x, point.y, point.z);

        if (!optimizer.addEdge(edge))
        {
            delete edge;
            continue;
        }
        edges.push_back({i, edge});
    }

    if (edges.size() < 6)
        return {};

    // ORB-SLAM2 logic reference: four 10-iteration rounds, classify edges
    // by the 2-DoF chi-square gate, and remove the robust kernel for the last
    // refinement rounds. Outlier decisions are local to this PnP result.
    for (int iteration = 0; iteration < 4; ++iteration)
    {
        if (!optimizer.initializeOptimization(0))
            return {};
        optimizer.optimize(10);

        for (auto& pose_edge : edges)
        {
            auto* edge = pose_edge.edge;
            edge->computeError();
            const bool inlier = std::isfinite(edge->chi2()) &&
                                edge->chi2() <= kMonoChi2Threshold &&
                                edge->isDepthPositive();
            edge->setLevel(inlier ? 0 : 1);
            if (iteration == 1)
                edge->setRobustKernel(nullptr);
        }
    }

    g2o::SE3Quat optimized_pose = pose_vertex->estimate();
    result.R = eigenToCvRotation(optimized_pose.rotation().toRotationMatrix());
    result.rvec.release();
    cv::Rodrigues(result.R, result.rvec);
    result.tvec = eigenToCvTranslation(optimized_pose.translation());

    std::vector<cv::Point3d> inlier_object_points;
    std::vector<cv::Point2f> inlier_img_points;
    std::vector<int> inlier_indices;
    inlier_object_points.reserve(edges.size());
    inlier_img_points.reserve(edges.size());
    inlier_indices.reserve(edges.size());

    for (const auto& pose_edge : edges)
    {
        auto* edge = pose_edge.edge;
        edge->computeError();
        if (edge->level() != 0 || !std::isfinite(edge->chi2()) ||
            edge->chi2() > kMonoChi2Threshold || !edge->isDepthPositive())
        {
            continue;
        }
        inlier_indices.push_back(pose_edge.input_index);
        inlier_object_points.push_back(result.object_points[pose_edge.input_index]);
        inlier_img_points.push_back(result.img_points[pose_edge.input_index]);
    }

    int positive_depth_num = 0;
    for (const auto& point : inlier_object_points)
    {
        const cv::Mat point_camera = result.R *
            (cv::Mat_<double>(3, 1) << point.x, point.y, point.z) + result.tvec;
        if (point_camera.at<double>(2, 0) > 1e-6)
            positive_depth_num++;
    }

    if (inlier_indices.size() < 6 || positive_depth_num < 6 ||
        static_cast<double>(positive_depth_num) / inlier_indices.size() < 0.90)
    {
        return {};
    }

    result.inlier_indices = cv::Mat(static_cast<int>(inlier_indices.size()), 1, CV_32S);
    for (int i = 0; i < result.inlier_indices.rows; ++i)
        result.inlier_indices.at<int>(i, 0) = inlier_indices[i];

    result.inlier_num = static_cast<int>(inlier_indices.size());
    result.ransac_reproj_error = computeMeanReprojectionError(
        inlier_object_points, inlier_img_points, result.rvec, result.tvec);
    result.optimized_reproj_error = result.ransac_reproj_error;
    result.optimized = true;
    result.success = true;
    return result;
}

std::vector<std::shared_ptr<Frame>> PoseOptimizer::collectLocalBAKeyframes(
    const std::shared_ptr<Frame>& cur_keyframe) const
{
    std::vector<std::shared_ptr<Frame>> local_keyframes;

    if (cur_keyframe == nullptr || !cur_keyframe->isKeyframe())
        return local_keyframes;

    std::unordered_set<std::size_t> added_ids;
    added_ids.reserve(32);

    auto append_unique = 
        [&local_keyframes, &added_ids](const std::shared_ptr<Frame>& keyframe)
        {
            if (keyframe == nullptr || !keyframe->isKeyframe())
                return;

            if (!added_ids.insert(keyframe->getId()).second)
                return;

            local_keyframes.push_back(keyframe);
        };

    append_unique(cur_keyframe);

    const std::vector<std::shared_ptr<Frame>> connected_keyframes = cur_keyframe->getConnectedKeyframes(1);

    for (const auto& keyframe : connected_keyframes)
        append_unique(keyframe);

    return local_keyframes;
}

std::vector<std::shared_ptr<MapPoint>> PoseOptimizer::collectLocalBAMapPoints(
    const std::vector<std::shared_ptr<Frame>>& local_keyframes) const
{
    std::vector<std::shared_ptr<MapPoint>> local_map_points;
    std::unordered_set<std::size_t> map_point_ids;
    map_point_ids.reserve(256);

    for (const auto& keyframe : local_keyframes)
    {
        if (keyframe == nullptr)
            continue;

        for (const auto& feature : keyframe->getFeatures())
        {
            if (feature == nullptr)
                continue;

            const std::shared_ptr<MapPoint> map_point = feature->getMapPoint();
            if (map_point == nullptr || map_point->isBad())
                continue;

            if (map_point_ids.insert(map_point->getId()).second)
                local_map_points.push_back(map_point);
        }
    }

    return local_map_points;
}

std::vector<std::shared_ptr<Frame>> PoseOptimizer::collectFixedBAKeyframes(
    const std::vector<std::shared_ptr<Frame>>& local_keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& local_map_points,
    const std::unordered_map<std::size_t,
                             std::vector<std::shared_ptr<Feature>>>&
        map_point_observation_snapshots) const
{
    std::vector<std::shared_ptr<Frame>> fixed_keyframes;
    std::unordered_set<std::size_t> local_keyframe_ids;
    std::unordered_set<std::size_t> fixed_keyframe_ids;

    local_keyframe_ids.reserve(local_keyframes.size() * 2 + 1);
    fixed_keyframe_ids.reserve(64);

    for (const auto& keyframe : local_keyframes)
    {
        if (keyframe != nullptr)
            local_keyframe_ids.insert(keyframe->getId());
    }

    for (const auto& map_point : local_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const auto snapshot_it = map_point_observation_snapshots.find(map_point->getId());
        if (snapshot_it == map_point_observation_snapshots.end())
            continue;

        const auto& observations = snapshot_it->second;

        for (const auto& feature : observations)
        {
            if (feature == nullptr)
                continue;

            const std::shared_ptr<Frame> keyframe = feature->getFrame();
            if (keyframe == nullptr || !keyframe->isKeyframe())
                continue;

            if (local_keyframe_ids.count(keyframe->getId()) > 0)
                continue;

            if (fixed_keyframe_ids.insert(keyframe->getId()).second)
                fixed_keyframes.push_back(keyframe);
        }
    }

    return fixed_keyframes;
}

std::vector<PoseOptimizer::LocalBAObservation> PoseOptimizer::collectLocalBAObservations(
    const std::vector<std::shared_ptr<Frame>>& local_keyframes,
    const std::vector<std::shared_ptr<Frame>>& fixed_keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& local_map_points,
    const std::unordered_map<std::size_t,
                             std::vector<std::shared_ptr<Feature>>>&
        map_point_observation_snapshots) const
{
    std::vector<LocalBAObservation> observations;

    std::unordered_set<std::size_t> local_keyframe_ids;
    std::unordered_set<std::size_t> fixed_keyframe_ids;
    local_keyframe_ids.reserve(local_keyframes.size() * 2 + 1);
    fixed_keyframe_ids.reserve(fixed_keyframes.size() * 2 + 1);

    for (const auto& keyframe : local_keyframes)
    {
        if (keyframe != nullptr)
            local_keyframe_ids.insert(keyframe->getId());
    }

    for (const auto& keyframe : fixed_keyframes)
    {
        if (keyframe != nullptr)
            fixed_keyframe_ids.insert(keyframe->getId());
    }

    std::unordered_map<std::size_t, int> observation_count_by_point;

    // Keep the observation order identical to the previous candidate/filter
    // path, but build the final vector only once.  Local BA is invoked for
    // every admitted keyframe, so copying the complete edge set here becomes
    // increasingly expensive as the map grows.
    observations.reserve(local_map_points.size() * 4);
    observation_count_by_point.reserve(local_map_points.size() * 2 + 1);

    for (const auto& map_point : local_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const auto snapshot_it = map_point_observation_snapshots.find(map_point->getId());
        if (snapshot_it == map_point_observation_snapshots.end())
            continue;

        const auto& keyframe_observations = snapshot_it->second;

        for (const auto& feature : keyframe_observations)
        {
            if (feature == nullptr || feature->getMapPoint() != map_point)
                continue;

            const std::shared_ptr<Frame> keyframe = feature->getFrame();
            if (keyframe == nullptr || !keyframe->isKeyframe())
                continue;

            const std::size_t keyframe_id = keyframe->getId();
            const bool in_local = local_keyframe_ids.count(keyframe_id) > 0;
            const bool in_fixed = fixed_keyframe_ids.count(keyframe_id) > 0;

            if (!in_local && !in_fixed)
                continue;

            LocalBAObservation observation;
            observation.keyframe = keyframe;
            observation.map_point = map_point;
            observation.feature = feature;
            observation.img_point = feature->getKeyPoint().pt;
            observation.feature_level = feature->getLevel();

            observations.push_back(observation);
            observation_count_by_point[map_point->getId()]++;
        }
    }

    // ORB-SLAM2 only creates a point vertex when it has at least two valid
    // observations in the local BA graph.  Remove under-constrained points
    // in place, preserving the exact edge order for all retained points.
    observations.erase(
        std::remove_if(observations.begin(), observations.end(),
                       [&observation_count_by_point](const LocalBAObservation& observation)
                       {
                           if (observation.map_point == nullptr)
                               return true;

                           const auto count_it = observation_count_by_point.find(
                               observation.map_point->getId());
                           return count_it == observation_count_by_point.end() ||
                                  count_it->second < 2;
                       }),
        observations.end());

    return observations;
}

PoseOptimizer::LocalBAContext PoseOptimizer::buildLocalBAContext(
    const std::shared_ptr<Map>& map,
    const std::shared_ptr<Frame>& cur_frame) const
{
    LocalBAContext context;

    if (map == nullptr)
        return context;

    context.local_keyframes = collectLocalBAKeyframes(cur_frame);
    if (context.local_keyframes.empty())
        return context;

    context.local_map_points = collectLocalBAMapPoints(context.local_keyframes);

    context.map_point_observation_snapshots.reserve(context.local_map_points.size() * 2 + 1);
    for (const auto& map_point : context.local_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        context.map_point_observation_snapshots.emplace(
            map_point->getId(), map_point->getKeyframeObservations());
    }

    context.fixed_keyframes = 
        collectFixedBAKeyframes(context.local_keyframes,
                                context.local_map_points,
                                context.map_point_observation_snapshots);

    context.observations = collectLocalBAObservations(context.local_keyframes,
                                                      context.fixed_keyframes,
                                                      context.local_map_points,
                                                      context.map_point_observation_snapshots);

    if (context.observations.empty())
        return context;

    std::unordered_set<std::size_t> local_keyframe_ids;
    local_keyframe_ids.reserve(context.local_keyframes.size() * 2 + 1);

    for (const auto& keyframe : context.local_keyframes)
    {
        if (keyframe != nullptr)
            local_keyframe_ids.insert(keyframe->getId());
    }

    std::unordered_set<std::size_t> observed_map_point_ids;
    std::unordered_set<std::size_t> observed_fixed_keyframe_ids;

    observed_map_point_ids.reserve(context.local_map_points.size() * 2 + 1);
    observed_fixed_keyframe_ids.reserve(context.fixed_keyframes.size() * 2 + 1);

    for (const auto& observation : context.observations)
    {
        if (observation.map_point != nullptr)
            observed_map_point_ids.insert(observation.map_point->getId());

        if (observation.keyframe != nullptr && 
            local_keyframe_ids.count(observation.keyframe->getId()) == 0)
        {
            observed_fixed_keyframe_ids.insert(observation.keyframe->getId());
        }
    }

    context.local_map_points.erase(
        std::remove_if(
            context.local_map_points.begin(), context.local_map_points.end(),
            [&observed_map_point_ids](const std::shared_ptr<MapPoint>& map_point)
            {
                return map_point == nullptr || observed_map_point_ids.count(map_point->getId()) == 0;
            }),
        context.local_map_points.end()
    );

    context.fixed_keyframes.erase(
        std::remove_if(
            context.fixed_keyframes.begin(), context.fixed_keyframes.end(),
            [&observed_fixed_keyframe_ids](const std::shared_ptr<Frame>& keyframe)
            {
                return keyframe == nullptr || observed_fixed_keyframe_ids.count(keyframe->getId()) == 0;
            }),
        context.fixed_keyframes.end()
    );

    for (const auto& keyframe : map->getKeyframes())
    {
        if (keyframe != nullptr && keyframe->isKeyframe())
        {
            context.map_origin_keyframe = keyframe;
            break;
        }
    }

    auto snapshot_keyframe_pose = [&context](const std::shared_ptr<Frame>& keyframe)
    {
        if (keyframe == nullptr ||
            context.keyframe_poses.count(keyframe->getId()) > 0)
        {
            return;
        }

        cv::Mat R_cw;
        cv::Mat t_cw;
        keyframe->copyPose(R_cw, t_cw);
        if (!R_cw.empty() && !t_cw.empty())
            context.keyframe_poses.emplace(
                keyframe->getId(), LocalBAContext::PoseSnapshot{R_cw, t_cw});
    };

    for (const auto& keyframe : context.local_keyframes)
        snapshot_keyframe_pose(keyframe);
    for (const auto& keyframe : context.fixed_keyframes)
        snapshot_keyframe_pose(keyframe);

    context.map_point_positions.reserve(context.local_map_points.size() * 2 + 1);
    for (const auto& map_point : context.local_map_points)
    {
        if (map_point != nullptr && !map_point->isBad())
            context.map_point_positions.emplace(map_point->getId(), map_point->getPos());
    }

    context.map_version = map->getVersion();

    return context;
}

bool PoseOptimizer::refineKeyframePose(
    const std::shared_ptr<Frame>& keyframe,
    const std::vector<LocalBAObservation>& observations) const
{
    if (camera_ == nullptr || keyframe == nullptr || !keyframe->isKeyframe())
        return false;

    if (observations.size() < 6)
        return false;

    auto computePoseMeanError = 
        [this, &observations](const cv::Mat& R_cw, const cv::Mat& t_cw) -> double
        {
            if (R_cw.empty() || t_cw.empty())
                return 0.0;

            const cv::Mat& K = camera_->getK();
            const double fx = K.at<double>(0, 0);
            const double fy = K.at<double>(1, 1);
            const double cx = K.at<double>(0, 2);
            const double cy = K.at<double>(1, 2);

            double total_error = 0.0;
            int valid_num = 0;

            for (const auto& observation : observations)
            {
                if (observation.map_point == nullptr || 
                    observation.map_point->isBad())
                    continue;

                const cv::Point3d& point_world = observation.map_point->getPos();
                const cv::Mat point_world_mat = 
                    (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);

                const cv::Mat point_camera_mat = R_cw * point_world_mat + t_cw;

                const double x = point_camera_mat.at<double>(0, 0);
                const double y = point_camera_mat.at<double>(1, 0);
                const double z = point_camera_mat.at<double>(2, 0);

                if (z <= 1e-6)
                    continue;

                const cv::Point2d projected_point(fx * x / z + cx, fy * y / z + cy);

                total_error += cv::norm(projected_point - cv::Point2d(observation.img_point));
                valid_num++;
            }

            if (valid_num <= 0)
                return 0.0;

            return total_error / valid_num;
        };

    cv::Mat cur_R;
    cv::Mat cur_t;
    keyframe->copyPose(cur_R, cur_t);

    if (cur_R.empty() || cur_t.empty())
        return false;

    double cur_error = computePoseMeanError(cur_R, cur_t);

    constexpr int KMaxGNIterations = 8;
    constexpr double KDamping = 1e-6;
    constexpr double kStepThreshold = 1e-6;
    constexpr double kMaxAcceptablePointReprojError = 6.0;

    bool updated = false;
    
    for (int iter = 0; iter < KMaxGNIterations; iter++)
    {
        cv::Matx<double, 6, 6> H = cv::Matx<double, 6, 6>::zeros();
        cv::Vec<double, 6> b(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        int valid_obs_num = 0;

        for (const auto& observation : observations)
        {
            if (observation.map_point == nullptr || observation.map_point->isBad())
                continue;

            cv::Vec2d residual;
            cv::Matx<double, 2, 6> J_pose;
            cv::Matx<double, 2, 3> J_point;

            if (!linearizeObservation(observation, 
                                      observation.map_point->getPos(), 
                                      residual,
                                      J_pose, 
                                      J_point))
            {
                continue;
            }

            H += J_pose.t() * J_pose;
            b += J_pose.t() * residual;
            valid_obs_num++;
        }

        if (valid_obs_num < 6)
            break;

        for (int i = 0; i < 6; i++)
            H(i, i) += KDamping;

        const cv::Matx<double, 6, 1> b_mat(b);
        const cv::Mat rhs = -cv::Mat(b_mat);
        cv::Mat delta_mat;

        if (!cv::solve(cv::Mat(H), rhs, delta_mat, cv::DECOMP_CHOLESKY) &&
            !cv::solve(cv::Mat(H), rhs, delta_mat, cv::DECOMP_SVD))
        {
            break;
        }

        const cv::Vec3d delta_rot(delta_mat.at<double>(0, 0),
                                  delta_mat.at<double>(1, 0),
                                  delta_mat.at<double>(2, 0));
        const cv::Vec3d delta_trans(delta_mat.at<double>(3, 0),
                                    delta_mat.at<double>(4, 0),
                                    delta_mat.at<double>(5, 0));

        const double step_norm = 
            std::sqrt(delta_rot.dot(delta_rot) + delta_trans.dot(delta_trans));
        if (step_norm < kStepThreshold)
            break;

        cv::Mat delta_rvec = (cv::Mat_<double>(3, 1) << delta_rot[0], delta_rot[1], delta_rot[2]);
        cv::Mat delta_R;
        cv::Rodrigues(delta_rvec, delta_R);
        
        const cv::Mat candidate_R = delta_R * cur_R;
        const cv::Mat candidate_t = 
            delta_R * cur_t + (cv::Mat_<double>(3, 1) << delta_trans[0], delta_trans[1], delta_trans[2]);
        
        bool all_positive_depth = true;
        for (const auto& observation : observations)
        {
            if (observation.map_point == nullptr || observation.map_point->isBad())
                continue;

            const cv::Point3d& point_world = observation.map_point->getPos();
            const cv::Mat point_world_mat = 
                (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);

            const cv::Mat point_camera_mat = candidate_R * point_world_mat + candidate_t;
            if (point_camera_mat.at<double>(2, 0) <= 1e-6)
            {
                all_positive_depth = false;
                break;
            }
        }

        if (!all_positive_depth)
            break;

        const double candidate_error = computePoseMeanError(candidate_R, candidate_t);
        if (candidate_error <= 0.0 || candidate_error > kMaxAcceptablePointReprojError)
            break;

        if (cur_error > 0.0 && candidate_error > cur_error + 1e-6)
            break;

        cur_R = candidate_R;
        cur_t = candidate_t;
        cur_error = candidate_error;
        keyframe->setPose(cur_R, cur_t);
        updated = true;
    }

    return updated;
}

bool PoseOptimizer::refineLocalMapPoint(
    const std::shared_ptr<MapPoint>& map_point,
    const std::vector<LocalBAObservation>& observations) const
{
    if (camera_ == nullptr || map_point == nullptr || map_point->isBad())
        return false;

    if (observations.size() < 2)
        return false;

    std::vector<std::shared_ptr<Frame>> obs_frames;
    std::vector<cv::Point2f> img_points;
    obs_frames.reserve(observations.size());
    img_points.reserve(observations.size());

    for (const auto& observation : observations)
    {
        if (observation.keyframe == nullptr)
            continue;

        obs_frames.push_back(observation.keyframe);
        img_points.push_back(observation.img_point);
    }

    if (obs_frames.size() < 2)
        return false;

    cv::Point3d cur_point = map_point->getPos();
    double cur_error = computeMeanPointReprojectionError(cur_point, obs_frames, img_points);

    constexpr int KMaxGNIterations = 5;
    constexpr double KDamping = 1e-6;
    constexpr double kStepThreshold = 1e-6;
    constexpr double kMaxAcceptablePointReprojError = 4.0;

    bool updated = false;

    for (int iter = 0; iter < KMaxGNIterations; iter++)
    {
        cv::Matx33d H = cv::Matx33d::zeros();
        cv::Vec3d b(0.0, 0.0, 0.0); 
        int valid_obs_num = 0;

        for (const auto& observation : observations)
        {
            cv::Vec2d residual;
            cv::Matx<double, 2, 6> J_pose;
            cv::Matx<double, 2, 3> J_point;

            if (!linearizeObservation(observation, cur_point, residual, J_pose, J_point))
                continue;

            H += J_point.t() * J_point;
            b += J_point.t() * residual;
            valid_obs_num++;
        }

        if (valid_obs_num < 2)
            break;

        H(0, 0) += KDamping;
        H(1, 1) += KDamping;
        H(2, 2) += KDamping;

        const cv::Matx<double, 3, 1> b_mat(b);
        const cv::Mat rhs = -cv::Mat(b_mat);
        cv::Mat delta_mat;

        if (!cv::solve(cv::Mat(H), rhs, delta_mat, cv::DECOMP_CHOLESKY) &&
            !cv::solve(cv::Mat(H), rhs, delta_mat, cv::DECOMP_SVD))
        {
            break;
        }

        const cv::Vec3d delta(delta_mat.at<double>(0, 0),
                              delta_mat.at<double>(1, 0),
                              delta_mat.at<double>(2, 0));

        if (cv::norm(delta) < kStepThreshold)
            break;

        const cv::Point3d candidate_point(cur_point.x + delta[0],
                                          cur_point.y + delta[1],
                                          cur_point.z + delta[2]);

        bool all_positive_depth = true;
        for (const auto& observation : observations)
        {
            cv::Point2d projected_point;
            cv::Point3d point_camera;

            if (!projectWorldPointToFrame(observation.keyframe,
                                          candidate_point,
                                          projected_point,
                                          point_camera))
            {
                all_positive_depth = false;
                break;
            }
        }

        if (!all_positive_depth)
            break;

        const double candidate_error =
            computeMeanPointReprojectionError(candidate_point, obs_frames, img_points);

        if (candidate_error <= 0.0 || candidate_error > kMaxAcceptablePointReprojError)
            break;

        if (cur_error > 0.0 && candidate_error > cur_error + 1e-6)
            break;

        cur_point = candidate_point;
        cur_error = candidate_error;
        updated = true;
    }

    if (!updated)
        return false;

    map_point->setPos(cur_point);
    return true;
}

int PoseOptimizer::refineLocalMapPoints(const LocalBAContext& context, 
                                        const std::vector<LocalBAObservation>& observations,
                                        int outer_iters) const
{
    if (context.local_map_points.empty() || observations.empty())
        return 0;

    int refined_total = 0;

    for (int iter = 0; iter < outer_iters; iter++)
    {
        int refined_this_round = 0;

        for (const auto& map_point : context.local_map_points)
        {
            const std::vector<LocalBAObservation> map_point_observations = 
                collectObservationsForMapPoint(observations, map_point);

            if (map_point_observations.size() < 2)
                continue;

            if (refineLocalMapPoint(map_point, map_point_observations))
                refined_this_round++;
        }

        if (refined_this_round == 0)
            break;

        refined_total += refined_this_round;
    }

    return refined_total;
}

int PoseOptimizer::optimizeLocalKeyframes(const LocalBAContext& context,
                                          const std::vector<LocalBAObservation>& observations,
                                          int outer_iters) const
{
    if (context.local_keyframes.empty() || context.local_map_points.empty())
        return 0;

    std::unordered_set<std::size_t> local_map_point_ids;
    local_map_point_ids.reserve(context.local_map_points.size() * 2 + 1);

    for (const auto& map_point : context.local_map_points)
    {
        if (map_point != nullptr && !map_point->isBad())
            local_map_point_ids.insert(map_point->getId());
    }

    int optimized_total = 0;

    for (int iter = 0; iter < outer_iters; iter++)
    {
        int optimized_this_round = 0;

        for (const auto& keyframe : context.local_keyframes)
        {
            const std::vector<LocalBAObservation> keyframe_observations = 
                collectObservationsForKeyframe(observations, keyframe);

            if (keyframe_observations.size() < 6)
                continue;

            if (refineKeyframePose(keyframe, keyframe_observations))
                optimized_this_round++;
        }

        if (optimized_this_round == 0)
            break;

        optimized_total += optimized_this_round;
    }

    return optimized_total;
}

InitialMapOptimizationResult PoseOptimizer::optimizeInitialMap(const std::shared_ptr<Map>& map, 
                                                               const std::shared_ptr<Frame>& ref_frame, 
                                                               const std::shared_ptr<Frame>& cur_frame) const
{
    InitialMapOptimizationResult result;

    if (camera_ == nullptr || map == nullptr ||
        ref_frame == nullptr || cur_frame == nullptr)
    {
        return result;
    }

    cv::Mat ref_R;
    cv::Mat ref_t;
    cv::Mat cur_R;
    cv::Mat cur_t;

    ref_frame->copyPose(ref_R, ref_t);
    cur_frame->copyPose(cur_R, cur_t);

    if (ref_R.empty() || ref_t.empty() ||
        cur_R.empty() || cur_t.empty())
    {
        return result;
    }

    const cv::Mat& K = camera_->getK();
    if (K.rows != 3 || K.cols != 3)
        return result;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    
    if (!std::isfinite(fx) || !std::isfinite(fy) ||
        !std::isfinite(cx) || !std::isfinite(cy))
    {
        return result;
    }

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));

    auto add_pose_vertex = 
        [&optimizer](int id, const std::shared_ptr<Frame>& frame, bool fixed) -> g2o::VertexSE3Expmap*
        {
            Eigen::Matrix3d R_cw;
            Eigen::Vector3d t_cw;

            cv::Mat R_cv;
            cv::Mat t_cv;
            frame->copyPose(R_cv, t_cv);

            if (R_cv.empty() || t_cv.empty() ||
                !cvToEigenRotation(R_cv, R_cw) ||
                !cvToEigenTranslation(t_cv, t_cw))
            {
                return nullptr;
            }

            auto* vertex = new g2o::VertexSE3Expmap();
            vertex->setId(id);
            vertex->setEstimate(g2o::SE3Quat(R_cw, t_cw));
            vertex->setFixed(fixed);

            if (!optimizer.addVertex(vertex))
            {
                delete vertex;
                return nullptr;
            }

            return vertex;
        };

    g2o::VertexSE3Expmap* ref_vertex = add_pose_vertex(0, ref_frame, true);
    g2o::VertexSE3Expmap* cur_vertex = add_pose_vertex(1, cur_frame, false);

    if (ref_vertex == nullptr || cur_vertex == nullptr)
        return result;

    struct InitialMapGraphEdge
    {
        std::shared_ptr<MapPoint> map_point;
        g2o::EdgeSE3ProjectXYZ* edge{nullptr};
    };

    constexpr double kMonoChi2Threshold = 5.991;
    const double kHuberDelta = std::sqrt(kMonoChi2Threshold);

    int next_vertex_id = 2;

    std::unordered_map<std::size_t, g2o::VertexPointXYZ*> point_vertices;
    std::unordered_map<std::size_t, std::shared_ptr<MapPoint>> graph_map_points;
    std::vector<InitialMapGraphEdge> graph_edges;

    point_vertices.reserve(map->getMapPointNum() * 2 + 1);
    graph_map_points.reserve(map->getMapPointNum() * 2 + 1);
    graph_edges.reserve(map->getMapPointNum() * 2);

    for (const auto& map_point : map->getMapPoints())
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::shared_ptr<Feature> ref_feature = map_point->getRefFeature();
        const std::shared_ptr<Feature> cur_feature = map_point->getCurFeature();

        if (ref_feature == nullptr || cur_feature == nullptr ||
            ref_feature->getMapPoint() != map_point ||
            cur_feature->getMapPoint() != map_point ||
            ref_feature->getFrame() != ref_frame ||
            cur_feature->getFrame() != cur_frame)
        {
            continue;
        }

        const cv::Point3d& point = map_point->getPos();
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;

        auto* point_vertex = new g2o::VertexPointXYZ();
        point_vertex->setId(next_vertex_id++);
        point_vertex->setEstimate(Eigen::Vector3d(point.x, point.y, point.z));
        point_vertex->setMarginalized(true);

        if (!optimizer.addVertex(point_vertex))
        {
            delete point_vertex;
            continue;
        }

        point_vertices[map_point->getId()] = point_vertex;
        graph_map_points[map_point->getId()] = map_point;

        auto add_observation = 
            [&optimizer, &graph_edges, map_point, point_vertex, fx, fy, cx, cy, kHuberDelta, this]
            (const std::shared_ptr<Feature>& feature, g2o::VertexSE3Expmap* pose_vertex) 
            {
                const int octave = std::clamp(feature->getLevel(), 0, levels_num_ - 1);
                const double inv_sigma2 = std::pow(scale_factor_, -2.0 * octave);

                if (!std::isfinite(inv_sigma2) || inv_sigma2 <= 0.0)
                    return;

                auto* edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setVertex(0, point_vertex);
                edge->setVertex(1, pose_vertex);

                const cv::Point2f& point = feature->getKeyPoint().pt;
                edge->setMeasurement(Eigen::Vector2d(point.x, point.y));
                edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);

                auto* robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(kHuberDelta);
                edge->setRobustKernel(robust_kernel);

                edge->fx = fx;
                edge->fy = fy;
                edge->cx = cx;
                edge->cy = cy;

                if (!optimizer.addEdge(edge))
                {
                    delete edge;
                    return;
                }

                graph_edges.push_back({map_point, edge});
            };

        add_observation(ref_feature, ref_vertex);
        add_observation(cur_feature, cur_vertex);
    }

    if (graph_edges.size() < 30)
        return result;

    auto compute_mean_error = 
        [](const std::vector<InitialMapGraphEdge>& edges, bool only_inliers) -> double
        {
            double total_error = 0.0;
            int valid_num = 0;

            for (const auto& graph_edge : edges)
            {
                if (graph_edge.edge == nullptr)
                    continue;

                const double chi2 = graph_edge.edge->chi2();

                if (only_inliers &&
                   (!std::isfinite(chi2) || chi2 > kMonoChi2Threshold ||
                    !graph_edge.edge->isDepthPositive()))
                {
                    continue;
                }

                const Eigen::Vector2d error = graph_edge.edge->error();
                if (!error.allFinite())
                    continue;

                total_error += error.norm();
                valid_num++;
            }

            return valid_num > 0 ? total_error / valid_num : 0.0;
        };

    if (!optimizer.initializeOptimization(0))
        return result;

    optimizer.computeActiveErrors();
    result.mean_reproj_error_before = compute_mean_error(graph_edges, false);

    for (const auto& graph_edge : graph_edges)
    {
        if (graph_edge.edge->chi2() > kMonoChi2Threshold || !graph_edge.edge->isDepthPositive())
        {
            graph_edge.edge->setLevel(1);
        }

        graph_edge.edge->setRobustKernel(nullptr);
    }

    if (!optimizer.initializeOptimization(0) || optimizer.optimize(10) <= 0)
        return result;

    optimizer.computeActiveErrors();

    std::unordered_map<std::size_t, int> edge_num_by_point;
    std::unordered_map<std::size_t, int> inlier_num_by_point;

    for (const auto& graph_edge : graph_edges)
    {
        const std::size_t point_id = graph_edge.map_point->getId();
        edge_num_by_point[point_id]++;

        const double chi2 = graph_edge.edge->chi2();
        if (std::isfinite(chi2) &&
            chi2 <= kMonoChi2Threshold &&
            graph_edge.edge->isDepthPositive())
        {
            inlier_num_by_point[point_id]++;
            result.inlier_edge_num++;
        }
    }

    result.edge_num = static_cast<int>(graph_edges.size());
    result.mean_reproj_error_after = compute_mean_error(graph_edges, true);

    for (const auto& item : edge_num_by_point)
    {
        const std::size_t point_id = item.first;

        if (item.second != 2 || inlier_num_by_point[point_id] != 2)
            continue;

        const auto map_point_it = graph_map_points.find(point_id);
        if (map_point_it != graph_map_points.end())
            result.inlier_map_points.push_back(map_point_it->second);
    }

    const g2o::SE3Quat& ref_estimate = ref_vertex->estimate();
    const g2o::SE3Quat& cur_estimate = cur_vertex->estimate();

    const Eigen::Matrix3d optimized_ref_R =
        ref_estimate.rotation().toRotationMatrix();
    const Eigen::Vector3d optimized_ref_t = ref_estimate.translation();
    const Eigen::Matrix3d optimized_cur_R =
        cur_estimate.rotation().toRotationMatrix();
    const Eigen::Vector3d optimized_cur_t = cur_estimate.translation();

    if (!optimized_ref_R.allFinite() || !optimized_ref_t.allFinite() ||
        !optimized_cur_R.allFinite() || !optimized_cur_t.allFinite())
    {
        return result;
    }

    std::vector<std::pair<std::shared_ptr<MapPoint>, cv::Point3d>> optimized_points;
    optimized_points.reserve(result.inlier_map_points.size());

    for (const auto& item : point_vertices)
    {
        const Eigen::Vector3d& estimate = item.second->estimate();
        if (!estimate.allFinite())
            return {};

        const auto map_point_it = graph_map_points.find(item.first);
        if (map_point_it == graph_map_points.end())
            return {};

        optimized_points.emplace_back(
            map_point_it->second,
            cv::Point3d(estimate[0], estimate[1], estimate[2])
        );
    }

    result.optimized_ref_R = eigenToCvRotation(optimized_ref_R);
    result.optimized_ref_t = eigenToCvTranslation(optimized_ref_t);
    result.optimized_cur_R = eigenToCvRotation(optimized_cur_R);
    result.optimized_cur_t = eigenToCvTranslation(optimized_cur_t);

    result.optimized_map_points.clear();
    result.optimized_map_points.reserve(optimized_points.size());

    for (const auto& optimized_point : optimized_points)
    {
        if (optimized_point.first == nullptr)
            continue;

        result.optimized_map_points.push_back({optimized_point.first, optimized_point.second});
    }

    result.success = 
        !result.inlier_map_points.empty() &&
        !result.optimized_map_points.empty() &&
        !result.optimized_ref_R.empty() &&
        !result.optimized_cur_R.empty() &&
        std::isfinite(result.mean_reproj_error_before) &&
        std::isfinite(result.mean_reproj_error_after);

    return result;  
}

LocalBAResult PoseOptimizer::optimizeLocalMap(const std::shared_ptr<Map>& map, 
                                              const std::shared_ptr<Frame>& cur_keyframe,
                                              const PnPResult& tracking_seed) const
{
    LocalBAResult result;

    if (camera_ == nullptr || map == nullptr || cur_keyframe == nullptr)
    {   
        return result;
    }

    const auto context_start = std::chrono::steady_clock::now();
    LocalBAContext context;
    {
        std::lock_guard<std::mutex> map_lock(map->getMutex());
        if (!cur_keyframe->isKeyframe())
            return result;

        context = buildLocalBAContext(map, cur_keyframe);
    }

    result.context_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - context_start).count();
    result.local_keyframe_num = context.local_keyframes.size();
    result.fixed_keyframe_num = context.fixed_keyframes.size();
    result.local_map_point_num = context.local_map_points.size();
    result.observation_num = context.observations.size();

    std::unordered_map<std::size_t, std::size_t> ba_edges_by_map_point;
    ba_edges_by_map_point.reserve(context.local_map_points.size() * 2 + 1);
    for (const auto& observation : context.observations)
    {
        if (observation.map_point != nullptr)
            ba_edges_by_map_point[observation.map_point->getId()]++;
    }
    for (const auto& entry : ba_edges_by_map_point)
    {
        if (entry.second == 2)
            result.map_points_with_2_ba_edges++;
        else if (entry.second == 3)
            result.map_points_with_3_ba_edges++;
        else if (entry.second >= 4)
            result.map_points_with_4_or_more_ba_edges++;
    }

    if (context.local_keyframes.size() < 3 ||
        context.local_map_points.size() < 20 ||
        context.observations.size() < 30)
    {
        return result;
    }

    const cv::Mat& K = camera_->getK();

    if (K.rows != 3 || K.cols != 3)
        return result;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    if (!std::isfinite(fx) || !std::isfinite(fy) ||
        !std::isfinite(cx) || !std::isfinite(cy))
    {
        return result;
    }

    const auto graph_build_start = std::chrono::steady_clock::now();
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));

    int next_vertex_id = 0;
    bool has_fixed_pose = false;

    std::unordered_set<std::size_t> local_keyframe_ids;
    local_keyframe_ids.reserve(context.local_keyframes.size() * 2 + 1);

    std::unordered_map<std::size_t, g2o::VertexSE3Expmap*> pose_vertices;
    pose_vertices.reserve(
        context.local_keyframes.size() + context.fixed_keyframes.size() * 2 + 1);

    std::shared_ptr<Frame> fallback_anchor;
    g2o::VertexSE3Expmap* fallback_anchor_vertex = nullptr;
   
        auto add_pose_vertex = 
        [&optimizer, &next_vertex_id, &context](const std::shared_ptr<Frame>& keyframe,
                                      bool fixed) -> g2o::VertexSE3Expmap*
        {
            Eigen::Matrix3d R_cw;
            Eigen::Vector3d t_cw;

            if (keyframe == nullptr)
                return nullptr;

            const auto pose_it = context.keyframe_poses.find(keyframe->getId());
            if (pose_it == context.keyframe_poses.end() ||
                !cvToEigenRotation(pose_it->second.R_cw, R_cw) ||
                !cvToEigenTranslation(pose_it->second.t_cw, t_cw))
            {
                return nullptr;
            }

            auto* vertex = new g2o::VertexSE3Expmap();
            vertex->setId(next_vertex_id++);
            vertex->setEstimate(g2o::SE3Quat(R_cw, t_cw));
            vertex->setFixed(fixed);

            if (!optimizer.addVertex(vertex))
            {
                delete vertex;
                return nullptr;
            }

            return vertex;
        };

    for (const auto& keyframe : context.local_keyframes)
    {
        if (keyframe == nullptr ||
            !local_keyframe_ids.insert(keyframe->getId()).second)
        {
            continue;
        }

        const bool is_map_origin = (keyframe == context.map_origin_keyframe);
        g2o::VertexSE3Expmap* vertex = add_pose_vertex(keyframe, is_map_origin);

        if (vertex == nullptr)
            return result;

        pose_vertices[keyframe->getId()] = vertex;
        has_fixed_pose = has_fixed_pose || is_map_origin;

        if (fallback_anchor == nullptr || keyframe->getId() < fallback_anchor->getId())
        {
            fallback_anchor = keyframe;
            fallback_anchor_vertex = vertex;
        }
    }

    for (const auto& keyframe : context.fixed_keyframes)
    {
        if (keyframe == nullptr ||
            local_keyframe_ids.count(keyframe->getId()) > 0 ||
            pose_vertices.count(keyframe->getId()) > 0)
        {
            continue;
        }

        g2o::VertexSE3Expmap* vertex = add_pose_vertex(keyframe, true);

        if (vertex == nullptr)
            continue;

        pose_vertices[keyframe->getId()] = vertex;
        has_fixed_pose = true;
    }

    if (pose_vertices.empty() || fallback_anchor_vertex == nullptr)
        return result;

    if (!has_fixed_pose)
        fallback_anchor_vertex->setFixed(true);

    std::unordered_map<std::size_t, g2o::VertexPointXYZ*> point_vertices;
    point_vertices.reserve(context.local_map_points.size() * 2 + 1);

    for (const auto& map_point : context.local_map_points)
    {
        if (map_point == nullptr)
            continue;

        const auto point_it = context.map_point_positions.find(map_point->getId());
        if (point_it == context.map_point_positions.end())
            continue;

        const cv::Point3d& point = point_it->second;

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;

        auto* vertex = new g2o::VertexPointXYZ();
        vertex->setId(next_vertex_id++);
        vertex->setEstimate(Eigen::Vector3d(point.x, point.y, point.z));
        vertex->setMarginalized(true);

        if (!optimizer.addVertex(vertex))
        {
            delete vertex;
            return result;
        }

        point_vertices[map_point->getId()] = vertex;
    }

    struct LocalBAGraphEdge
    {
        LocalBAObservation observation;
        g2o::EdgeSE3ProjectXYZ* edge;
    };

    constexpr double kMonoChi2Threshold = 5.991;
    const double kHuberDelta = std::sqrt(kMonoChi2Threshold);

    std::vector<LocalBAGraphEdge> graph_edges;
    graph_edges.reserve(context.observations.size());

    for (const auto& observation : context.observations)
    {
        if (observation.keyframe == nullptr || observation.map_point == nullptr ||
            observation.feature == nullptr)
        {
            continue;
        }

        const auto pose_it = pose_vertices.find(observation.keyframe->getId());
        const auto point_it = point_vertices.find(observation.map_point->getId());

        if (pose_it == pose_vertices.end() || point_it == point_vertices.end())
            continue;

        const int octave = std::clamp(observation.feature_level, 0, levels_num_ - 1);
        const double inv_sigma2 = std::pow(scale_factor_, -2.0 * octave);

        if (!std::isfinite(inv_sigma2) || inv_sigma2 <= 0.0)
            continue;

        auto* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, point_it->second);
        edge->setVertex(1, pose_it->second);
        edge->setMeasurement(
            Eigen::Vector2d(observation.img_point.x, observation.img_point.y));
        edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);

        auto* robust_kernel = new g2o::RobustKernelHuber();
        robust_kernel->setDelta(kHuberDelta);
        edge->setRobustKernel(robust_kernel);

        edge->fx = fx;
        edge->fy = fy;
        edge->cx = cx;
        edge->cy = cy;

        if (!optimizer.addEdge(edge))
        {
            delete edge;
            continue;
        }

        graph_edges.push_back({observation, edge});
    }

    if (graph_edges.size() < 30)
        return result;

    result.edge_num = graph_edges.size();
    result.graph_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - graph_build_start).count();

    const auto solve_start = std::chrono::steady_clock::now();
    if (!optimizer.initializeOptimization(0))
        return result;

    if (optimizer.optimize(5) <= 0)
        return result;

    optimizer.computeActiveErrors();

    for (const auto& graph_edge : graph_edges)
    {
        if (graph_edge.edge->chi2() > kMonoChi2Threshold ||
            !graph_edge.edge->isDepthPositive())
        {
            graph_edge.edge->setLevel(1);
        }

        graph_edge.edge->setRobustKernel(nullptr);
    }

    if (!optimizer.initializeOptimization(0))
        return result;

    if (optimizer.optimize(10) <= 0)
        return result;

    optimizer.computeActiveErrors();

    result.solver_success = true;
    result.solve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - solve_start).count();

    for (const auto& graph_edge : graph_edges)
    {
        const double chi2 = graph_edge.edge->chi2();

        if (!std::isfinite(chi2) || chi2 > kMonoChi2Threshold || 
            !graph_edge.edge->isDepthPositive())
        {
            result.rejected_edge_num++;
        }
    }

    struct OptimizedPose
    {
        std::shared_ptr<Frame> keyframe;
        cv::Mat R_cw;
        cv::Mat t_cw;
    };

    struct OptimizedPoint
    {
        std::shared_ptr<MapPoint> map_point;
        cv::Point3d position;
    };

    std::vector<OptimizedPose> optimized_poses;
    optimized_poses.reserve(pose_vertices.size());

    for (const auto& keyframe : context.local_keyframes)
    {
        if (keyframe == nullptr)
            continue;

        const auto vertex_it = pose_vertices.find(keyframe->getId());

        if (vertex_it == pose_vertices.end())
            continue;

        const g2o::SE3Quat& estimate = vertex_it->second->estimate();
        const Eigen::Matrix3d R_cw = estimate.rotation().toRotationMatrix();
        const Eigen::Vector3d t_cw = estimate.translation();

        if (!R_cw.allFinite() || !t_cw.allFinite())
            return result;

        optimized_poses.push_back(
            {keyframe, eigenToCvRotation(R_cw), eigenToCvTranslation(t_cw)});
    }

    std::vector<OptimizedPoint> optimized_points;
    optimized_points.reserve(point_vertices.size());

    for (const auto& map_point : context.local_map_points)
    {
        if (map_point == nullptr)
            continue;

        const auto vertex_it = point_vertices.find(map_point->getId());

        if (vertex_it == point_vertices.end())
            continue;

        const Eigen::Vector3d& estimate = vertex_it->second->estimate();

        if (!estimate.allFinite())
            return result;

        optimized_points.push_back(
            {map_point, cv::Point3d(estimate[0], estimate[1], estimate[2])});
    }

    std::unordered_map<std::size_t, cv::Point3d> candidate_map_point_positions;
    candidate_map_point_positions.reserve(optimized_points.size() * 2 + 1);
    for (const auto& optimized_point : optimized_points)
        candidate_map_point_positions.emplace(optimized_point.map_point->getId(), optimized_point.position);

    const auto current_pose_it = 
        std::find_if(optimized_poses.begin(), optimized_poses.end(),
                     [&cur_keyframe](const OptimizedPose& pose)
                     {
                         return pose.keyframe == cur_keyframe;
                     });

    if (current_pose_it == optimized_poses.end())
        return result;

    const auto validation_start = std::chrono::steady_clock::now();
    const bool candidate_accepted = validateLocalBACandidate(
        tracking_seed,
        current_pose_it->R_cw,
        current_pose_it->t_cw,
        candidate_map_point_positions,
        result.seed_reproj_error,
        result.candidate_seed_reproj_error);
    result.validation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_start).count();

    if (!candidate_accepted)
    {
        result.rejection_reason = "candidate_tracking_seed_validation";
        return result;
    }

    // The expensive solve used only the immutable context above. Re-enter the
    // map transaction only to validate and publish a complete candidate.
    const auto commit_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> map_lock(map->getMutex());
    if (map->getVersion() != context.map_version || !cur_keyframe->isKeyframe())
    {
        result.rejection_reason = "stale_map_snapshot";
        return result;
    }

    result.accepted = true;

    for (const auto& optimized_pose : optimized_poses)
        optimized_pose.keyframe->setPose(optimized_pose.R_cw, optimized_pose.t_cw);

    for (const auto& optimized_point : optimized_points)
        optimized_point.map_point->setPos(optimized_point.position);

    std::unordered_set<const Feature*> detached_features_ptrs;
    detached_features_ptrs.reserve(context.observations.size());

    std::unordered_set<std::size_t> touched_map_point_ids;
    touched_map_point_ids.reserve(context.local_map_points.size() * 2 + 1);

    // A BA pose/position update does not change a MapPoint's descriptor.  The
    // descriptor only needs recomputation when this transaction removes an
    // observation.  ORB-SLAM2's Local BA follows the same separation: it
    // refreshes geometric normal/depth state for optimized points, while
    // distinctive descriptors are maintained when observations change.
    std::vector<std::shared_ptr<MapPoint>> touched_map_points;
    touched_map_points.reserve(graph_edges.size());

    std::vector<std::shared_ptr<Frame>> detached_observation_owners;
    detached_observation_owners.reserve(graph_edges.size());
    std::unordered_set<std::size_t> detached_observation_owner_ids;
    detached_observation_owner_ids.reserve(graph_edges.size());

    for (const auto& graph_edge : graph_edges)
    {
        const double chi2 = graph_edge.edge->chi2();

        if (!std::isfinite(chi2) || chi2 > kMonoChi2Threshold ||
            !graph_edge.edge->isDepthPositive())
        {
            const std::shared_ptr<Feature>& feature = graph_edge.observation.feature;
            const std::shared_ptr<MapPoint>& map_point = graph_edge.observation.map_point;

            if (feature == nullptr || map_point == nullptr ||
                !detached_features_ptrs.insert(feature.get()).second ||
                feature->getMapPoint() != map_point)
            {
                continue;
            }

            const std::shared_ptr<Frame> owner_keyframe = feature->getFrame();
            if (owner_keyframe != nullptr && owner_keyframe->isKeyframe() &&
                detached_observation_owner_ids.insert(owner_keyframe->getId()).second)
            {
                detached_observation_owners.push_back(owner_keyframe);
            }

            feature->setMapPoint(nullptr);
            map_point->removeObservation(feature);

            if (touched_map_point_ids.insert(map_point->getId()).second)
                touched_map_points.push_back(map_point);
        }
    }

    for (const auto& map_point : touched_map_points)
    {
        if (map_point != nullptr && !map_point->isBad() &&
            map_point->getKeyframeObservationCount() < 2)
        {
            map_point->setBad(true);
        }
    }

    map->removeBadMapPoints();

    const auto view_statistics_start = std::chrono::steady_clock::now();
    for (const auto& map_point : context.local_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        map_point->updateViewStatistics(scale_factor_, levels_num_);
    }
    result.view_statistics_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - view_statistics_start).count();

    const auto descriptor_refresh_start = std::chrono::steady_clock::now();
    // Only MapPoints whose observations were actually detached can have a
    // changed representative descriptor. Iterate that compact set directly
    // instead of rescanning the entire local BA point set and hashing IDs.
    for (const auto& map_point : touched_map_points)
    {
        if (map_point == nullptr || map_point->isBad())
        {
            continue;
        }

        map_point->updateRepresentativeDescriptor();
        result.descriptor_refresh_num++;
    }
    result.descriptor_refresh_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - descriptor_refresh_start).count();

    // R110-A: rebuild exactly the topology touched by BA observation removal.
    // Connections must be refreshed before selecting neighbors; otherwise the
    // old R112 ordering can miss a newly affected covisibility endpoint.
    std::vector<std::shared_ptr<Frame>> topology_keyframes;
    std::unordered_set<std::size_t> topology_keyframe_ids;
    std::unordered_set<std::size_t> topology_connections_refreshed;
    topology_keyframe_ids.reserve(detached_observation_owners.size() * 8 + 1);
    topology_connections_refreshed.reserve(detached_observation_owners.size() + 1);
    for (const auto& owner_keyframe : detached_observation_owners)
    {
        if (owner_keyframe == nullptr || !owner_keyframe->isKeyframe())
            continue;

        owner_keyframe->updateConnections();
        topology_connections_refreshed.insert(owner_keyframe->getId());
        if (topology_keyframe_ids.insert(owner_keyframe->getId()).second)
            topology_keyframes.push_back(owner_keyframe);

        const std::vector<std::shared_ptr<Frame>> neighbors =
            owner_keyframe->getConnectedKeyframes(5);
        for (const auto& neighbor : neighbors)
        {
            if (neighbor != nullptr && neighbor->isKeyframe() &&
                topology_keyframe_ids.insert(neighbor->getId()).second)
            {
                topology_keyframes.push_back(neighbor);
            }
        }
    }

    for (const auto& keyframe : topology_keyframes)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe())
            continue;
        if (topology_connections_refreshed.count(keyframe->getId()) > 0)
            continue;
        keyframe->updateConnections();
    }

    map->reconcileCovisibilityConstraints(topology_keyframes);

    // R110-B: only Local BA-optimized poses can invalidate the relative
    // measurement. Map indexes restrict this operation to their incident
    // sequential/covisibility edges; Loop Closing retains full refreshes.
    map->refreshPoseGraphMeasurements(context.local_keyframes);

    map->markModified();
    result.commit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - commit_start).count();

    return result;
}


bool PoseOptimizer::computeRelativePoseConstraint(const std::shared_ptr<Frame>& from_keyframe,
                                                  const std::shared_ptr<Frame>& to_keyframe,
                                                  cv::Mat& R_21, 
                                                  cv::Mat& t_21) const
{
    R_21.release();
    t_21.release();

    if (from_keyframe == nullptr || to_keyframe == nullptr)
        return false;

    cv::Mat R_from;
    cv::Mat t_from;
    cv::Mat R_to;
    cv::Mat t_to;

    from_keyframe->copyPose(R_from, t_from);
    to_keyframe->copyPose(R_to, t_to);

    if (R_from.empty() || t_from.empty() ||
        R_to.empty() || t_to.empty())
    {
        return false;
    }

    R_21 = R_to * R_from.t();
    t_21 = t_to - R_21 * t_from;

    return true;
}

bool PoseOptimizer::optimizeEssentialGraph(const std::vector<std::shared_ptr<Frame>>& map_keyframes,
                                           const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                           const std::vector<PoseGraphConstraint>& constraints,
                                           const std::shared_ptr<Frame>& anchor_keyframe) const
{
    if (anchor_keyframe == nullptr || !anchor_keyframe->isKeyframe())
        return false;

    if (map_keyframes.size() < 3 || constraints.size() < 3)
        return false;

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));

    std::vector<PoseGraphVertexState> states;
    states.reserve(map_keyframes.size());

    std::unordered_map<std::size_t, int> vertex_ids;
    vertex_ids.reserve(map_keyframes.size() * 2 + 1);

    for (const auto& keyframe : map_keyframes)
    {
        if (keyframe == nullptr || !keyframe->isKeyframe())
            continue;

        cv::Mat R_cw;
        cv::Mat t_cw;
        keyframe->copyPose(R_cw, t_cw);

        g2o::Sim3 initial_estimate;
        if (!makeSim3(R_cw, t_cw, 1.0, initial_estimate))
            continue;

        const int vertex_id = static_cast<int>(states.size());

        auto* vertex = new g2o::VertexSim3Expmap();
        vertex->setId(vertex_id);
        vertex->setEstimate(initial_estimate);
        vertex->_fix_scale = false;
        vertex->setFixed(keyframe == anchor_keyframe);

        if (!optimizer.addVertex(vertex))
        {
            delete vertex;
            return false;
        }

        vertex_ids[keyframe->getId()] = vertex_id;
        states.push_back({keyframe, 
                          R_cw,
                          t_cw,
                          vertex});
    }

    if (states.size() < 3 || vertex_ids.count(anchor_keyframe->getId()) == 0)
    {
        return false;
    }

    int valid_edge_num = 0;

    for (const auto& constraint : constraints)
    {
        const std::shared_ptr<Frame> from_keyframe = constraint.from_keyframe.lock();
        const std::shared_ptr<Frame> to_keyframe = constraint.to_keyframe.lock();

        if (from_keyframe == nullptr || to_keyframe == nullptr)
            continue;

        const auto from_it = vertex_ids.find(from_keyframe->getId());
        const auto to_it = vertex_ids.find(to_keyframe->getId());

        if (from_it == vertex_ids.end() || to_it == vertex_ids.end())
            continue;

        g2o::Sim3 measurement;
        if (!makeSim3(constraint.R_21, constraint.t_21, constraint.scale, measurement))
            continue;

        auto* edge = new g2o::EdgeSim3();
        edge->setVertex(0, optimizer.vertex(from_it->second));
        edge->setVertex(1, optimizer.vertex(to_it->second));
        edge->setMeasurement(measurement);

        edge->setInformation(Eigen::Matrix<double, 7, 7>::Identity());

        if (!optimizer.addEdge(edge))
        {
            delete edge;
            continue;
        }

        valid_edge_num++;
    }

    if (valid_edge_num < 3)
        return false;

    if (!optimizer.initializeOptimization())
        return false;

    constexpr int kMaxIterations = 20;
    if (optimizer.optimize(kMaxIterations) <= 0)
        return false;

    // Build all corrected map-point positions in the optimizer candidate.  Do
    // not mutate the live map until every pose and point has passed validation;
    // otherwise a late invalid Sim3 vertex can leave a partial graph correction
    // behind even though this function reports failure.
    struct OptimizedMapPoint
    {
        std::shared_ptr<MapPoint> map_point;
        cv::Point3d position;
    };
    struct OptimizedPoseState
    {
        std::shared_ptr<Frame> keyframe;
        cv::Mat R_cw;
        cv::Mat t_cw;
    };
    std::vector<OptimizedMapPoint> optimized_map_points;
    optimized_map_points.reserve(map_points.size());

    // fixed camera poses, solve the map point
    for (const auto& map_point : map_points)
    {
        if (map_point == nullptr || map_point->isBad())
            continue;

        const std::shared_ptr<Feature> ref_feature = 
            map_point->selectRefFeatureCandidate();

        if (ref_feature == nullptr)
            continue;

        const std::shared_ptr<Frame> ref_keyframe = ref_feature->getFrame();
        if (ref_keyframe == nullptr || !ref_keyframe->isKeyframe())
            continue;

        const auto ref_it = vertex_ids.find(ref_keyframe->getId());
        if (ref_it == vertex_ids.end())
            continue;

        const PoseGraphVertexState& ref_state = states[ref_it->second];

        Eigen::Matrix3d R_before;
        Eigen::Vector3d t_before;

        if (!cvToEigenRotation(ref_state.R_cw_before, R_before) ||
            !cvToEigenTranslation(ref_state.t_cw_before, t_before))
        {
            continue;
        }

        const Eigen::Vector3d point_world_before = pointToEigen(map_point->getPos());

        const Eigen::Vector3d point_camera = R_before * point_world_before + t_before;

        const Eigen::Vector3d point_world_after = 
            ref_state.vertex->estimate().inverse().map(point_camera);

        if (!point_world_after.allFinite())
            continue;

        optimized_map_points.push_back({
            map_point,
            cv::Point3d(point_world_after[0],
                        point_world_after[1],
                        point_world_after[2])});
    }

    std::vector<OptimizedPoseState> optimized_poses;
    optimized_poses.reserve(states.size());
    for (const auto& state : states)
    {
        const g2o::Sim3& optimized_sim3 = state.vertex->estimate();
        const double scale = optimized_sim3.scale();

        if (!std::isfinite(scale) || scale <= 1e-8)
            return false;

        const Eigen::Matrix3d R_cw = 
            optimized_sim3.rotation().toRotationMatrix();
        const Eigen::Vector3d t_cw = 
            optimized_sim3.translation() / scale;

        if (!R_cw.allFinite() || !t_cw.allFinite())
            return false;

        optimized_poses.push_back({state.keyframe,
                                   eigenToCvRotation(R_cw),
                                   eigenToCvTranslation(t_cw)});
    }

    if (optimized_poses.size() != states.size())
        return false;

    // The caller holds the map transaction while invoking this function.  The
    // final writes are deliberately grouped here so the official map observes
    // either the complete essential-graph correction or no correction at all.
    for (const auto& optimized_pose : optimized_poses)
        optimized_pose.keyframe->setPose(optimized_pose.R_cw, optimized_pose.t_cw);

    for (const auto& optimized_point : optimized_map_points)
        optimized_point.map_point->setPos(optimized_point.position);

    // Sequential/covisibility measurements depend on the corrected poses;
    // refresh them after the complete commit while preserving the verified loop
    // edge measurement.
    // (The map version is advanced by the caller's surrounding transaction.)

    return true;
}


} // namespace mini_orb_slam
