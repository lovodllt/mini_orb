#ifndef MINI_ORB_SLAM_INCLUDE_POSE_OPTIMIZER_H_
#define MINI_ORB_SLAM_INCLUDE_POSE_OPTIMIZER_H_

#include <memory>
#include <vector>
#include <cmath>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

#include <opencv4/opencv2/core.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "camera.h"
#include "common.h"
#include "map.h"

namespace mini_orb_slam 
{

class PoseOptimizer
{
public:
    explicit PoseOptimizer(const std::shared_ptr<Camera>& camera,
                           double scale_factor,
                           int levels_num);

    PnPResult optimize(const PnPResult& input_result) const;

    // ORB-SLAM2 logic reference: TrackLocalMap refines the pose supplied by
    // the seed tracker directly against all local-map observations.
    PnPResult optimizeWithPosePrior(const PnPResult& input_result) const;

    InitialMapOptimizationResult optimizeInitialMap(
        const std::shared_ptr<Map>& map,
        const std::shared_ptr<Frame>& ref_frame,
        const std::shared_ptr<Frame>& cur_frame) const;

    LocalBAResult optimizeLocalMap(const std::shared_ptr<Map>& map, 
                                   const std::shared_ptr<Frame>& cur_keyframe,
                                   const PnPResult& tracking_seed,
                                   bool* abort_flag = nullptr) const;

    bool optimizeEssentialGraph(const std::vector<std::shared_ptr<Frame>>& map_keyframes,
                                const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                const std::vector<PoseGraphConstraint>& constraints,
                                const std::shared_ptr<Frame>& anchor_keyframe) const;

private:
    PnPResult optimizeImpl(const PnPResult& input_result,
                            bool run_pnp_ransac) const;

    struct LocalBAObservation
    {
        std::shared_ptr<Frame> keyframe;
        std::shared_ptr<MapPoint> map_point;
        std::shared_ptr<Feature> feature;
        cv::Point2f img_point;
        int feature_level{0};
    };

    struct LocalBAContext
    {
        struct PoseSnapshot
        {
            cv::Mat R_cw;
            cv::Mat t_cw;
        };

        std::vector<std::shared_ptr<Frame>> local_keyframes;
        std::vector<std::shared_ptr<Frame>> fixed_keyframes;
        std::vector<std::shared_ptr<MapPoint>> local_map_points;
        std::vector<LocalBAObservation> observations;
        std::shared_ptr<Frame> map_origin_keyframe;
        std::unordered_map<std::size_t, PoseSnapshot> keyframe_poses;
        std::unordered_map<std::size_t, cv::Point3d> map_point_positions;
        std::size_t map_version{0};
    };

    bool projectWorldPointToFrame(const std::shared_ptr<Frame>& frame,
                                  const cv::Point3d& point_world,
                                  cv::Point2d& projected_point,
                                  cv::Point3d& point_camera) const;

    double computeMeanReprojectionError(const std::vector<cv::Point3d>& object_points,
                                        const std::vector<cv::Point2f>& img_points,
                                        const cv::Mat& rvec,
                                        const cv::Mat& tvec) const; 

    double computeMeanPointReprojectionError(
        const cv::Point3d& point_world,
        const std::vector<std::shared_ptr<Frame>>& frames,
        const std::vector<cv::Point2f>& img_points) const;

    double computeObservationReprojectionError(const LocalBAObservation& observation) const;
    double computeMeanObservationReprojectionError(
        const std::vector<LocalBAObservation>& observations) const;

    double computeTotalBAReprojectionError(const LocalBAContext& context) const;

    bool validateLocalBACandidate(const PnPResult& tracking_seed, 
                                  const cv::Mat& candidate_R_cw,
                                  const cv::Mat& candidate_t_cw,
                                  const std::unordered_map<std::size_t, cv::Point3d>&
                                      candidate_map_point_positions,
                                  double& seed_reproj_error,
                                  double& candidate_seed_reproj_error) const;

    std::vector<std::shared_ptr<Frame>> collectLocalBAKeyframes(
        const std::shared_ptr<Frame>& cur_keyframe) const;

    std::vector<std::shared_ptr<MapPoint>> collectLocalBAMapPoints(
        const std::vector<std::shared_ptr<Frame>>& local_keyframes) const;

    std::vector<std::shared_ptr<Frame>> collectFixedBAKeyframes(
        const std::vector<std::shared_ptr<Frame>>& local_keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& local_map_points) const;

    std::vector<LocalBAObservation> collectLocalBAObservations(
        const std::vector<std::shared_ptr<Frame>>& local_keyframes,
        const std::vector<std::shared_ptr<Frame>>& fixed_keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& local_map_points) const;

    std::vector<LocalBAObservation> selectInlierObservations(
        const std::vector<LocalBAObservation>& observations,
        double max_reproj_error) const;

    std::vector<LocalBAObservation> collectObservationsForKeyframe(
        const std::vector<LocalBAObservation>& observations,
        const std::shared_ptr<Frame>& keyframe) const;

    std::vector<LocalBAObservation> collectObservationsForMapPoint(
        const std::vector<LocalBAObservation>& observations,
        const std::shared_ptr<MapPoint>& map_point) const;

    // Caller holds Map::getMutex(). All geometry used by the solver is copied
    // here, so g2o never reads mutable map state after the lock is released.
    LocalBAContext buildLocalBAContext(const std::shared_ptr<Map>& map,
                                       const std::shared_ptr<Frame>& cur_frame) const;

    cv::Matx23d computeProjectionJacobian(const cv::Point3d& point_camera) const;
    cv::Matx33d skewSymmetric(const cv::Point3d& p) const;

    bool linearizeObservation(const LocalBAObservation& observation,
                              const cv::Point3d& point_world,
                              cv::Vec2d& result,
                              cv::Matx<double, 2, 6>& J_pose,
                              cv::Matx<double, 2, 3>& J_point) const;

    bool refineKeyframePose(
        const std::shared_ptr<Frame>& keyframe,
        const std::vector<LocalBAObservation>& observations) const;  

    bool refineLocalMapPoint(
        const std::shared_ptr<MapPoint>& map_point,
        const std::vector<LocalBAObservation>& observations) const;

    int refineLocalMapPoints(const LocalBAContext& context,
                             const std::vector<LocalBAObservation>& observations,
                             int outer_iters) const;

    int optimizeLocalKeyframes(const LocalBAContext& context,
                               const std::vector<LocalBAObservation>& observations,
                               int outer_iters) const;

    bool computeRelativePoseConstraint(const std::shared_ptr<Frame>& from_keyframe,
                                       const std::shared_ptr<Frame>& to_keyframe,
                                       cv::Mat& R_21,
                                       cv::Mat& t_21) const;

    std::shared_ptr<Camera> camera_;
    double scale_factor_{1.2};
    int levels_num_{8};
};

} // namespace mini_orb_slam

#endif  // MINI_ORB_SLAM_INCLUDE_POSE_OPTIMIZER_H_
