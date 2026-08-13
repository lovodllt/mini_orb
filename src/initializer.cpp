#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv4/opencv2/calib3d.hpp>

#include "initializer.h"

namespace  
{

constexpr double kNormalizedRansacThreshold = 1e-3;
constexpr double kHomographyChi2Threshold = 5.991;
constexpr double kFundamentalChi2Threshold = 3.841;
constexpr double kFundamentalScoreChi2Threshold = 5.991;
constexpr double kModelInlierSigma2 = kNormalizedRansacThreshold * kNormalizedRansacThreshold;

bool projectionByHomography(const cv::Mat& H,
                            const cv::Point2f& point,
                            cv::Point2f& projected)
{
    if (H.rows != 3 || H.cols != 3)
        return false;

    cv::Mat H64;
    H.convertTo(H64, CV_64F);

    const double x = point.x;
    const double y = point.y;
    const double w = H64.at<double>(2, 0) * x + 
                     H64.at<double>(2, 1) * y +
                     H64.at<double>(2, 2);

    if (!std::isfinite(w) || std::abs(w) <= 1e-6)
        return false;

    const double u = (H64.at<double>(0, 0) * x + 
                      H64.at<double>(0, 1) * y +
                      H64.at<double>(0, 2)) / w;
    const double v = (H64.at<double>(1, 0) * x +
                      H64.at<double>(1, 1) * y +
                      H64.at<double>(1, 2)) / w;

    if (!std::isfinite(u) || !std::isfinite(v))
        return false;

    projected = cv::Point2f(static_cast<float>(u), static_cast<float>(v));

    return true;
}

double scoreHomographyModel(const std::vector<cv::Point2f>& ref_points,
                            const std::vector<cv::Point2f>& cur_points,
                            const cv::Mat& H21,
                            cv::Mat& inlier_mask)
{
    inlier_mask = cv::Mat::zeros(static_cast<int>(ref_points.size()), 1, CV_8U);

    if (ref_points.size() != cur_points.size() || ref_points.size() < 8 ||
        H21.rows != 3 || H21.cols != 3)
    {
        return 0.0;
    }

    cv::Mat H12;
    if (cv::invert(H21, H12, cv::DECOMP_SVD) == 0.0)
        return 0.0;

    const double max_error2 = kHomographyChi2Threshold * kModelInlierSigma2;
    double score = 0.0;

    for (std::size_t i = 0; i < ref_points.size(); i++)
    {
        cv::Point2f projected_ref, projected_cur;

        if (!projectionByHomography(H21, ref_points[i], projected_cur) ||
            !projectionByHomography(H12, cur_points[i], projected_ref))
        {
            continue;
        }

        const cv::Point2f ref_error = ref_points[i] - projected_ref;
        const cv::Point2f cur_error = cur_points[i] - projected_cur;

        const double ref_error2 = ref_error.dot(ref_error);
        const double cur_error2 = cur_error.dot(cur_error);

        if (ref_error2 > max_error2 || cur_error2 > max_error2)
            continue;

        inlier_mask.at<uchar>(static_cast<int>(i), 0) = 1;
        score += 2.0 * kHomographyChi2Threshold - (ref_error2 + cur_error2) / kModelInlierSigma2;
    }

    return score;
}

double scoreFundamentalModel(const std::vector<cv::Point2f>& ref_points,
                             const std::vector<cv::Point2f>& cur_points,
                             const cv::Mat& F21,
                             cv::Mat& inlier_mask)
{
    inlier_mask = cv::Mat::zeros(static_cast<int>(ref_points.size()), 1, CV_8U);

    if (ref_points.size() != cur_points.size() || ref_points.size() < 8 ||
        F21.rows != 3 || F21.cols != 3)
    {
        return 0.0;
    }

    cv::Mat F64;
    F21.convertTo(F64, CV_64F);

    const double max_error2 = kFundamentalChi2Threshold * kModelInlierSigma2;
    double score = 0.0;

    for (std::size_t i = 0; i < ref_points.size(); i++)
    {
        const cv::Vec3d x1(ref_points[i].x, ref_points[i].y, 1.0);
        const cv::Vec3d x2(cur_points[i].x, cur_points[i].y, 1.0);

        const cv::Matx33d F(
            F64.at<double>(0, 0), F64.at<double>(0, 1), F64.at<double>(0, 2),
            F64.at<double>(1, 0), F64.at<double>(1, 1), F64.at<double>(1, 2),
            F64.at<double>(2, 0), F64.at<double>(2, 1), F64.at<double>(2, 2));  
        
        const cv::Vec3d line2 = F * x1;
        const cv::Vec3d line1 = F.t() * x2;
        const double residual = x2.dot(line2);

        const double denom2 = line2[0] * line2[0] + line2[1] * line2[1];
        const double denom1 = line1[0] * line1[0] + line1[1] * line1[1];

        if (denom1 < 1e-12 || denom2 < 1e-12)
            continue;

        const double error2 = residual * residual / denom2;
        const double error1 = residual * residual / denom1;

        if (error1 > max_error2 || error2 > max_error2)
            continue;

        inlier_mask.at<uchar>(i, 0) = 1;
        score += 2.0 * kFundamentalScoreChi2Threshold - (error1 + error2) / kModelInlierSigma2;
    }

    return score;
}

using PoseHypothesis = std::pair<cv::Mat, cv::Mat>; // R, t

bool decomposeEssential(const cv::Mat& essential, std::vector<PoseHypothesis>& hypotheses)
{
    hypotheses.clear();

    if (essential.rows != 3 || essential.cols != 3)
        return false;

    cv::Mat E;
    essential.convertTo(E, CV_64F);

    cv::Mat w;
    cv::Mat U;
    cv::Mat Vt;
    cv::SVD::compute(E, w, U, Vt, cv::SVD::FULL_UV);

    if (U.rows != 3 || U.cols != 3 || Vt.rows != 3 || Vt.cols != 3)
        return false;

    cv::Mat t = U.col(2).clone();
    const double t_norm = cv::norm(t);
    if (!std::isfinite(t_norm) || t_norm <= 1e-12)
        return false;

    t /= t_norm;

    const cv::Mat W = (cv::Mat_<double>(3, 3) <<
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0);

    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;

    if (cv::determinant(R1) < 0.0)
        R1 = -R1;

    if (cv::determinant(R2) < 0.0)
        R2 = -R2;

    if (!cv::checkRange(R1) || !cv::checkRange(R2) || !cv::checkRange(t))
        return false;

    hypotheses.emplace_back(R1, t);
    hypotheses.emplace_back(R2, t);
    hypotheses.emplace_back(R1, -t);
    hypotheses.emplace_back(R2, -t);

    return true;
}

bool decomposeHomography(const cv::Mat& homography, std::vector<PoseHypothesis>& hypotheses)
{
    hypotheses.clear();

    if (homography.rows != 3 || homography.cols != 3)
        return false;

    cv::Mat H;
    homography.convertTo(H, CV_64F);

    cv::Mat w;
    cv::Mat U;
    cv::Mat Vt;
    cv::SVD::compute(H, w, U, Vt, cv::SVD::FULL_UV);

    if (w.rows != 3 || w.cols != 1 || U.rows != 3 || U.cols != 3 || Vt.rows != 3 || Vt.cols != 3)
        return false;

    const double d1 = w.at<double>(0, 0);
    const double d2 = w.at<double>(1, 0);
    const double d3 = w.at<double>(2, 0);

    if (!std::isfinite(d1) || !std::isfinite(d2) || !std::isfinite(d3) ||
        d1 <= 1e-12 || d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
    {
        return false;
    }

    const double d13_squared = d1 * d1 - d3 * d3;
    const double d12_squared = d1 * d1 - d2 * d2;
    const double d23_squared = d2 * d2 - d3 * d3;

    if (d13_squared <= 1e-12 || d12_squared <= 1e-12 || d23_squared <= 1e-12)
        return false;

    const double orientation_sign = cv::determinant(U) * cv::determinant(Vt);
    if (!std::isfinite(orientation_sign) || std::abs(orientation_sign) < 0.5)
        return false;

    const double x1_abs = std::sqrt(d12_squared / d13_squared);
    const double x3_abs = std::sqrt(d23_squared / d13_squared);

    const double x1[4] = { x1_abs, x1_abs, -x1_abs, -x1_abs };
    const double x3[4] = { x3_abs, -x3_abs, x3_abs, -x3_abs };

    const double numerator = std::sqrt(d12_squared * d23_squared);
    const double theta_denominator = (d1 + d3) * d2;
    const double phi_denominator = (d1 - d3) * d2;

    if (std::abs(theta_denominator) <= 1e-12 || std::abs(phi_denominator) <= 1e-12)
        return false;

    const double theta_sine_abs = numerator / theta_denominator;
    const double theta_cosine = (d2 * d2 + d1 * d3) / theta_denominator;

    const double phi_sine_abs = numerator / phi_denominator;
    const double phi_cosine = (d1 * d3 - d2 * d2) / phi_denominator;

    if (!std::isfinite(theta_sine_abs) || !std::isfinite(theta_cosine) ||
        !std::isfinite(phi_sine_abs) || !std::isfinite(phi_cosine))
    {
        return false;
    }

    const double theta_sine[4] = 
        { theta_sine_abs, -theta_sine_abs, -theta_sine_abs, theta_sine_abs };
    const double phi_sine[4] = 
        { phi_sine_abs, -phi_sine_abs, -phi_sine_abs, phi_sine_abs };

    for (int i = 0; i < 4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_64F);
        Rp.at<double>(0, 0) = theta_cosine;
        Rp.at<double>(0, 2) = -theta_sine[i];
        Rp.at<double>(2, 0) = theta_sine[i];
        Rp.at<double>(2, 2) = theta_cosine;

        cv::Mat R = orientation_sign * U * Rp * Vt;

        const cv::Mat tp = (cv::Mat_<double>(3, 1) << x1[i], 0.0, -x3[i]) * (d1 - d3);
        cv::Mat t = U * tp;

        const double t_norm = cv::norm(t);
        if (t_norm > 1e-12 && cv::checkRange(R) && cv::checkRange(t))
            hypotheses.emplace_back(R, t / t_norm);
    }

    for (int i = 0; i < 4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_64F);
        Rp.at<double>(0, 0) = phi_cosine;
        Rp.at<double>(0, 2) = phi_sine[i];
        Rp.at<double>(1, 1) = -1.0;
        Rp.at<double>(2, 0) = phi_sine[i];
        Rp.at<double>(2, 2) = -phi_cosine;

        cv::Mat R = orientation_sign * U * Rp * Vt;

        const cv::Mat tp = (cv::Mat_<double>(3, 1) << x1[i], 0.0, x3[i]) * (d1 + d3);
        cv::Mat t = U * tp;

        const double t_norm = cv::norm(t);
        if (t_norm > 1e-12 && cv::checkRange(R) && cv::checkRange(t))
            hypotheses.emplace_back(R, t / t_norm);
    }

    return hypotheses.size() == 8;
}

} // namespace


namespace mini_orb_slam
{

Initializer::Initializer(const std::shared_ptr<Camera>& camera) : camera_(camera) {}

PoseRecoveryResult Initializer::recoverPoseFromFrames(
    const std::shared_ptr<Frame>& ref_frame, 
    const std::shared_ptr<Frame>& cur_frame, 
    const std::vector<std::pair<int, int>>& match_indices) const
{
    PoseRecoveryResult result;

    if (camera_ == nullptr || ref_frame == nullptr || cur_frame == nullptr)
        return result;
    
    const std::vector<cv::KeyPoint>& ref_kps = ref_frame->getKeypoints();
    const std::vector<cv::KeyPoint>& cur_kps = cur_frame->getKeypoints();

    std::vector<cv::Point2f> ref_points, cur_points;
    std::vector<int> valid_ref_indices, valid_cur_indices;

    ref_points.reserve(match_indices.size());
    cur_points.reserve(match_indices.size());
    valid_ref_indices.reserve(match_indices.size());
    valid_cur_indices.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int ref_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (ref_idx < 0 || cur_idx < 0 ||
            ref_idx >= static_cast<int>(ref_kps.size()) ||
            cur_idx >= static_cast<int>(cur_kps.size()))
        {
            continue;
        }

        ref_points.push_back(ref_kps[ref_idx].pt);
        cur_points.push_back(cur_kps[cur_idx].pt);
        valid_ref_indices.push_back(ref_idx);
        valid_cur_indices.push_back(cur_idx);
    }

    if (ref_points.size() < 8)
        return result;

    std::vector<cv::Point2f> ref_norm_points, cur_norm_points;
    cv::undistortPoints(ref_points, ref_norm_points, camera_->getK(), camera_->getD());
    cv::undistortPoints(cur_points, cur_norm_points, camera_->getK(), camera_->getD());

    cv::Mat homography_ransac_mask;
    cv::Mat fundenmental_ransac_mask;

    const cv::Mat H21 = cv::findHomography(ref_norm_points,
                                           cur_norm_points,
                                           cv::RANSAC,
                                           kNormalizedRansacThreshold,
                                           homography_ransac_mask);

    const cv::Mat F21 = cv::findFundamentalMat(ref_norm_points,
                                               cur_norm_points,
                                               cv::FM_RANSAC,
                                               kNormalizedRansacThreshold,
                                               0.99,
                                               fundenmental_ransac_mask);

    cv::Mat homography_mask;
    cv::Mat fundamental_mask;

    const double score_h = H21.empty() 
        ? 0.0 : scoreHomographyModel(ref_norm_points, cur_norm_points, H21, homography_mask);

    const double score_f = F21.empty()
        ? 0.0 : scoreFundamentalModel(ref_norm_points, cur_norm_points, F21, fundamental_mask);

    const double score_sum = score_h + score_f;
    if (score_sum <= 0.0)
        return result;

    struct ReconstructionCandidate
    {
        PoseRecoveryResult pose;
        int triangulated_num{0};
        bool triangulation_success{false};
    };

    auto build_candidate = 
        [this, &ref_points, &cur_points, 
         &ref_norm_points, &cur_norm_points, 
         &valid_ref_indices, &valid_cur_indices](
            const cv::Mat& R,
            const cv::Mat& t,
            const cv::Mat& inlier_mask,
            const cv::Mat& essential_matrix,
            TwoViewModel model,
            double model_score) -> ReconstructionCandidate
        {
            ReconstructionCandidate candidate;

            if (R.rows != 3 || R.cols != 3 ||
                t.rows != 3 || t.cols != 1 ||
                inlier_mask.total() != ref_points.size())
            {
                return candidate;
            }

            candidate.pose.E = essential_matrix.clone();
            candidate.pose.R = R.clone();
            candidate.pose.t = t.clone();
            candidate.pose.inlier_mask = inlier_mask.clone();
            candidate.pose.model = model;
            candidate.pose.model_score = model_score;
            candidate.pose.model_inlier_num = cv::countNonZero(inlier_mask);

            candidate.pose.ref_inlier_points.reserve(ref_points.size());
            candidate.pose.cur_inlier_points.reserve(cur_points.size());
            candidate.pose.ref_norm_inlier_points.reserve(ref_norm_points.size());
            candidate.pose.cur_norm_inlier_points.reserve(cur_norm_points.size());
            candidate.pose.ref_feature_indices.reserve(ref_points.size());
            candidate.pose.cur_feature_indices.reserve(cur_points.size());

            for (int i = 0; i < ref_points.size(); i++)
            {
                if (inlier_mask.at<uchar>(i, 0) == 0)
                    continue;

                candidate.pose.ref_inlier_points.push_back(ref_points[i]);
                candidate.pose.cur_inlier_points.push_back(cur_points[i]);
                candidate.pose.ref_norm_inlier_points.push_back(ref_norm_points[i]);
                candidate.pose.cur_norm_inlier_points.push_back(cur_norm_points[i]);
                candidate.pose.ref_feature_indices.push_back(valid_ref_indices[i]);
                candidate.pose.cur_feature_indices.push_back(valid_cur_indices[i]);
            }

            if (candidate.pose.ref_inlier_points.size() < 8)
                return candidate;

            candidate.pose.success = true;

            const TriangulationResult triangulation = triangulateFromPose(candidate.pose);

            candidate.triangulated_num = triangulation.candidate_good_num;
            candidate.triangulation_success = triangulation.success;

            return candidate;
        };

    const double homography_ratio = score_h / score_sum;
    const bool use_homography = homography_ratio > 0.40;

    const cv::Mat selected_model = use_homography ? H21 : F21;
    const cv::Mat selected_mask = use_homography ? homography_mask : fundamental_mask;

    if (selected_model.empty() || selected_mask.empty() || cv::countNonZero(selected_mask) < 8)
        return result;

    std::vector<PoseHypothesis> hypotheses;

    const bool decomposition_success = use_homography
        ? decomposeHomography(selected_model, hypotheses)
        : decomposeEssential(selected_model, hypotheses);

    if (!decomposition_success)
        return result;

    std::vector<ReconstructionCandidate> candidates;
    candidates.reserve(hypotheses.size());

    const TwoViewModel selected_type = use_homography
        ? TwoViewModel::HOMOGRAPHY
        : TwoViewModel::FUNDAMENTAL;

    const double selected_score = use_homography ? score_h : score_f;
    const cv::Mat selected_essential = use_homography ? cv::Mat() : F21;

    for (const auto& hypothesis : hypotheses)
    {
        ReconstructionCandidate candidate = build_candidate(hypothesis.first, 
                                                            hypothesis.second, 
                                                            selected_mask, 
                                                            selected_essential, 
                                                            selected_type, 
                                                            selected_score);

        if (candidate.pose.success)
            candidates.push_back(std::move(candidate));
    }

    if (candidates.empty())
        return result;

    int best_idx = -1;
    int second_best_idx = -1;

    for (int i = 0; i < candidates.size(); i++)
    {
        if (best_idx < 0 ||
            candidates[i].triangulated_num > candidates[best_idx].triangulated_num)
        {
            second_best_idx = best_idx;
            best_idx = i;
        }
        else if (second_best_idx < 0 ||
                 candidates[i].triangulated_num > 
                 candidates[second_best_idx].triangulated_num)
        {
            second_best_idx = i;    
        }
    }

    if (best_idx < 0)
        return result;

    const ReconstructionCandidate& best_candidate = candidates[best_idx];
    if (!best_candidate.triangulation_success)
        return result;

    const int minimum_good_num = 
        std::max(50, static_cast<int>(std::ceil(0.9 * best_candidate.pose.model_inlier_num)));

    if (best_candidate.triangulated_num < minimum_good_num)
        return result;

    const int second_best_num = (second_best_idx < 0) 
        ? 0 
        : candidates[second_best_idx].triangulated_num;

    if (use_homography)
    {
        if (second_best_num >= static_cast<int>(0.75 * best_candidate.triangulated_num))
            return result;
    }
    else 
    {
        int similar_hypothesis_num = 0;

        for (const auto& candidate : candidates)
        {
            if (candidate.triangulated_num > 
                static_cast<int>(0.7 * best_candidate.triangulated_num))
            {
                similar_hypothesis_num++;
            }
        }

        if (similar_hypothesis_num > 1)
            return result;
    }

    return best_candidate.pose;
}

TriangulationResult Initializer::triangulateFromPose(const PoseRecoveryResult& pose_result) const
{
    TriangulationResult result;

    if (!pose_result.success || camera_ == nullptr ||
        pose_result.R.rows != 3 || pose_result.R.cols != 3 ||
        pose_result.t.rows != 3 || pose_result.t.cols != 1)
    {
        return result;
    }

    if (pose_result.ref_inlier_points.size() != pose_result.cur_inlier_points.size() ||
        pose_result.ref_inlier_points.size() < 2)
    {
        return result;
    }

    constexpr double kReprojErrorThreshold = 2.0;
    constexpr double kReprojErrorThresholdPx = 2.0;
    constexpr double kMinParallaxDeg = 1.0;
    constexpr double kMaxCosParallax = 0.99998;

    std::vector<double> cos_parallaxes;
    cos_parallaxes.reserve(pose_result.ref_inlier_points.size());

    cv::Mat K;
    camera_->getK().convertTo(K, CV_64F);

    const cv::Mat P_ref = cv::Mat::eye(3, 4, CV_64F);
    K.copyTo(P_ref(cv::Rect(0, 0, 3, 3)));

    cv::Mat P_cur = cv::Mat::zeros(3, 4, CV_64F);
    pose_result.R.copyTo(P_cur(cv::Rect(0, 0, 3, 3)));
    pose_result.t.copyTo(P_cur(cv::Rect(3, 0, 1, 3)));
    P_cur = K * P_cur;

    cv::Mat points_4d_raw;
    cv::triangulatePoints(P_ref,
                          P_cur,
                          pose_result.ref_inlier_points,
                          pose_result.cur_inlier_points,
                          points_4d_raw);

    cv::Mat points_4d;
    points_4d_raw.convertTo(points_4d, CV_64F);

    if (points_4d.empty() || points_4d.rows != 4)
        return result;

    result.raw_point_num = points_4d.cols;
    result.points_3d.reserve(points_4d.cols);
    result.ref_points.reserve(points_4d.cols);
    result.cur_points.reserve(points_4d.cols);
    result.ref_feature_indices.reserve(points_4d.cols);
    result.cur_feature_indices.reserve(points_4d.cols);

    const cv::Mat camera_center_ref = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat camera_center_cur = -pose_result.R.t() * pose_result.t;

    for (int i = 0; i < points_4d.cols; i++)
    {
        const double w = points_4d.at<double>(3, i);

        if (!std::isfinite(w) || std::abs(w) < 1e-6)
            continue;

        const cv::Mat point_ref = (cv::Mat_<double>(3, 1) << 
                                      points_4d.at<double>(0, i) / w,
                                      points_4d.at<double>(1, i) / w,
                                      points_4d.at<double>(2, i) / w);

        if (!cv::checkRange(point_ref))
            continue;

        const cv::Mat point_cur = pose_result.R * point_ref + pose_result.t;

        const cv::Mat ray_ref = point_ref - camera_center_ref;
        const cv::Mat ray_cur = point_ref - camera_center_cur;

        const double norm_ref = cv::norm(ray_ref);
        const double norm_cur = cv::norm(ray_cur);

        if (norm_ref < 1e-6 || norm_cur < 1e-6)
            continue;

        double cos_parallax = ray_ref.dot(ray_cur) / (norm_ref * norm_cur);
        cos_parallax = std::clamp(cos_parallax, -1.0, 1.0);

        const double depth_ref = point_ref.at<double>(2, 0);
        const double depth_cur = point_cur.at<double>(2, 0);

        if (depth_ref <= 0.0 && cos_parallax < kMaxCosParallax)
            continue;

        if (depth_cur <= 0.0 && cos_parallax < kMaxCosParallax)
            continue;

        result.positive_depth_num++;

        const cv::Mat point_ref_homo = (cv::Mat_<double>(4, 1) << 
                                            point_ref.at<double>(0, 0), 
                                            point_ref.at<double>(1, 0),
                                            point_ref.at<double>(2, 0), 
                                            1.0);
        
        const cv::Mat proj_ref = P_ref * point_ref_homo;
        const cv::Mat proj_cur = P_cur * point_ref_homo;

        const double ref_w = proj_ref.at<double>(2, 0);
        const double cur_w = proj_cur.at<double>(2, 0);

        if (std::abs(ref_w) <= 1e-6 || std::abs(cur_w) <= 1e-6)
            continue;

        const cv::Point2d reproj_ref(proj_ref.at<double>(0, 0) / ref_w,
                                     proj_ref.at<double>(1, 0) / ref_w);
        const cv::Point2d reproj_cur(proj_cur.at<double>(0, 0) / cur_w,
                                     proj_cur.at<double>(1, 0) / cur_w);

        const cv::Point2f& obs_ref = pose_result.ref_inlier_points[i];
        const cv::Point2f& obs_cur = pose_result.cur_inlier_points[i];

        const double err_ref = cv::norm(reproj_ref - cv::Point2d(obs_ref.x, obs_ref.y));
        const double err_cur = cv::norm(reproj_cur - cv::Point2d(obs_cur.x, obs_cur.y));

        if (!std::isfinite(err_ref) || err_ref > kReprojErrorThresholdPx ||
            !std::isfinite(err_cur) || err_cur > kReprojErrorThresholdPx)
        {
            continue;
        }

        result.candidate_good_num++;
        cos_parallaxes.push_back(cos_parallax);

        const double parallax_deg = std::acos(cos_parallax) * 180 / CV_PI;

        result.mean_reproj_error_ref += err_ref;
        result.mean_reproj_error_cur += err_cur;
        result.mean_parallax_deg += parallax_deg;
        result.reproj_valid_num++;

        if (cos_parallax < kMaxCosParallax)
        {
            result.good_parallax_num++;

            result.points_3d.emplace_back(point_ref.at<double>(0, 0),
                                          point_ref.at<double>(1, 0),
                                          point_ref.at<double>(2, 0));

            result.ref_points.push_back(obs_ref);
            result.cur_points.push_back(obs_cur);

            result.ref_feature_indices.push_back(pose_result.ref_feature_indices[i]);
            result.cur_feature_indices.push_back(pose_result.cur_feature_indices[i]);
        }
    }

    if (result.reproj_valid_num > 0)
    {
        result.mean_reproj_error_ref /= result.reproj_valid_num;

        result.mean_reproj_error_cur /= result.reproj_valid_num;

        result.mean_parallax_deg /= result.reproj_valid_num;
    }

    if (!cos_parallaxes.empty())
    {
        std::sort(cos_parallaxes.begin(), cos_parallaxes.end());

        const std::size_t idx = std::min<std::size_t>(50, cos_parallaxes.size() - 1);

        const double selected_cos = std::clamp(cos_parallaxes[idx], -1.0, 1.0);

        result.check_rt_parallax_deg = std::acos(selected_cos) * 180 / CV_PI;
    }

    if (!pose_result.ref_inlier_points.empty())
    {
        result.good_point_ratio = 
            static_cast<double>(result.points_3d.size()) / pose_result.ref_inlier_points.size();
    }

    if (result.candidate_good_num > 0)
    {
        result.good_parallax_ratio = 
            static_cast<double>(result.good_parallax_num) / result.candidate_good_num;
    }

    constexpr int kMinCandidateGoodNum = 50;

    result.success = 
        result.candidate_good_num >= kMinCandidateGoodNum &&
        std::isfinite(result.mean_reproj_error_ref) &&
        std::isfinite(result.mean_reproj_error_cur) &&
        std::isfinite(result.check_rt_parallax_deg) &&
        result.check_rt_parallax_deg >= kMinParallaxDeg;

    return result;
}

TriangulationResult Initializer::triangulateFromMatchedFrames(
    const std::shared_ptr<Frame>& ref_frame,
    const std::shared_ptr<Frame>& cur_frame,
    const std::vector<std::pair<int, int>>& match_indices) const
{
    TriangulationResult result;

    if (camera_ == nullptr || ref_frame == nullptr || cur_frame == nullptr)
        return result;

    cv::Mat R_ref;
    cv::Mat t_ref;
    cv::Mat R_cur;
    cv::Mat t_cur;
    ref_frame->copyPose(R_ref, t_ref);
    cur_frame->copyPose(R_cur, t_cur);

    if (R_ref.empty() || t_ref.empty() || R_cur.empty() || t_cur.empty())
    {
        return result;
    }

    const std::vector<cv::KeyPoint>& ref_kps = ref_frame->getKeypoints();
    const std::vector<cv::KeyPoint>& cur_kps = cur_frame->getKeypoints();

    std::vector<cv::Point2f> ref_points, cur_points;
    std::vector<int> valid_ref_indices, valid_cur_indices;

    ref_points.reserve(match_indices.size());
    cur_points.reserve(match_indices.size());
    valid_ref_indices.reserve(match_indices.size());
    valid_cur_indices.reserve(match_indices.size());

    for (const auto& match_idx : match_indices)
    {
        const int ref_idx = match_idx.first;
        const int cur_idx = match_idx.second;

        if (ref_idx < 0 || cur_idx < 0 ||
            ref_idx >= static_cast<int>(ref_kps.size()) ||
            cur_idx >= static_cast<int>(cur_kps.size()))
        {
            continue;
        }

        ref_points.push_back(ref_kps[ref_idx].pt);
        cur_points.push_back(cur_kps[cur_idx].pt);
        valid_ref_indices.push_back(ref_idx);
        valid_cur_indices.push_back(cur_idx);
    }

    if (ref_points.size() < 2)
        return result;

    std::vector<cv::Point2f> ref_norm_points, cur_norm_points;
    cv::undistortPoints(ref_points, ref_norm_points, camera_->getK(), camera_->getD());
    cv::undistortPoints(cur_points, cur_norm_points, camera_->getK(), camera_->getD());

    const cv::Mat R_ref_wc = R_ref.t();
    const cv::Mat R_cur_wc = R_cur.t();

    const cv::Mat R_cr = R_cur * R_ref_wc;
    const cv::Mat t_cr = t_cur - R_cr * t_ref;

    cv::Mat P_ref_norm = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P_cur_norm = cv::Mat::zeros(3, 4, CV_64F);
    R_cr.copyTo(P_cur_norm(cv::Rect(0, 0, 3, 3)));
    t_cr.copyTo(P_cur_norm(cv::Rect(3, 0, 1, 3)));

    cv::Mat K;
    camera_->getK().convertTo(K, CV_64F);
    const cv::Mat P_ref_pix = K * P_ref_norm;
    const cv::Mat P_cur_pix = K * P_cur_norm;

    cv::Mat points_4d_raw;
    cv::triangulatePoints(P_ref_norm,
                          P_cur_norm,
                          ref_norm_points,
                          cur_norm_points,
                          points_4d_raw);

    cv::Mat points_4d;
    points_4d_raw.convertTo(points_4d, CV_64F);

    if (points_4d.empty() || points_4d.rows != 4)
        return result;

    result.raw_point_num = points_4d.cols;

    result.points_3d.reserve(points_4d.cols);
    result.ref_points.reserve(points_4d.cols);
    result.cur_points.reserve(points_4d.cols);
    result.ref_feature_indices.reserve(points_4d.cols);
    result.cur_feature_indices.reserve(points_4d.cols);

    double total_reproj_error_ref = 0.0;
    double total_reproj_error_cur = 0.0;
    double total_parallax_deg = 0.0;
    int parallax_eval_num = 0;

    const double reproj_err_threshold = 2.0;
    const double min_parallax_deg = 1.0;

    const cv::Mat camera_center_ref = -R_ref_wc * t_ref;
    const cv::Mat camera_center_cur = -R_cur_wc * t_cur;

    for (int i = 0; i < points_4d.cols; i++)
    {
        const double w = points_4d.at<double>(3, i);
        if (!std::isfinite(w) || std::abs(w) < 1e-6)
            continue;

        const double x_ref = points_4d.at<double>(0, i) / w;
        const double y_ref = points_4d.at<double>(1, i) / w;
        const double z_ref = points_4d.at<double>(2, i) / w;

        if (!std::isfinite(x_ref) || !std::isfinite(y_ref) || !std::isfinite(z_ref))
            continue;

        cv::Mat point_ref = (cv::Mat_<double>(3, 1) << x_ref, y_ref, z_ref);
        cv::Mat point_cur = R_cr * point_ref + t_cr;

        const double depth_ref = z_ref;
        const double depth_cur = point_cur.at<double>(2, 0);

        if (depth_ref <= 0 || depth_cur <= 0)
            continue;

        result.positive_depth_num++;

        const cv::Mat point_world = R_ref_wc * point_ref - R_ref_wc * t_ref;
        const cv::Mat point_ref_homo = (cv::Mat_<double>(4, 1) << x_ref, y_ref, z_ref, 1.0);

        const cv::Mat proj_ref = P_ref_pix * point_ref_homo;
        const cv::Mat proj_cur = P_cur_pix * point_ref_homo;

        const double proj_ref_w = proj_ref.at<double>(2, 0);
        const double proj_cur_w = proj_cur.at<double>(2, 0);
        if (std::abs(proj_ref_w) < 1e-6 || std::abs(proj_cur_w) < 1e-6)
            continue;

        const double u_ref = proj_ref.at<double>(0, 0) / proj_ref_w;
        const double v_ref = proj_ref.at<double>(1, 0) / proj_ref_w;
        const double u_cur = proj_cur.at<double>(0, 0) / proj_cur_w;
        const double v_cur = proj_cur.at<double>(1, 0) / proj_cur_w;

        const cv::Point2f& obs_ref = ref_points[i];
        const cv::Point2f& obs_cur = cur_points[i];

        const double err_ref = cv::norm(cv::Point2d(u_ref, v_ref) - cv::Point2d(obs_ref.x, obs_ref.y));
        const double err_cur = cv::norm(cv::Point2d(u_cur, v_cur) - cv::Point2d(obs_cur.x, obs_cur.y));

        if (!std::isfinite(err_ref) || !std::isfinite(err_cur))
            continue;

        total_reproj_error_ref += err_ref;
        total_reproj_error_cur += err_cur;
        result.reproj_valid_num++;

        cv::Mat ray_ref = point_world - camera_center_ref;
        cv::Mat ray_cur = point_world - camera_center_cur;

        const double norm_ref = cv::norm(ray_ref);
        const double norm_cur = cv::norm(ray_cur);
        if (norm_ref < 1e-6 || norm_cur < 1e-6)
            continue;

        double cos_theta = ray_ref.dot(ray_cur) / (norm_ref * norm_cur);
        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));

        const double parallax_deg = std::acos(cos_theta) * 180.0 / CV_PI;
        total_parallax_deg += parallax_deg;
        parallax_eval_num++;

        if (parallax_deg >= min_parallax_deg)
            result.good_parallax_num++;

        const bool small_reproj_error = (err_ref <= reproj_err_threshold) && (err_cur <= reproj_err_threshold);
        const bool enough_parallax = (parallax_deg >= min_parallax_deg);

        if (!small_reproj_error || !enough_parallax)
            continue;

        result.points_3d.emplace_back(point_world.at<double>(0, 0),
                                      point_world.at<double>(1, 0),
                                      point_world.at<double>(2, 0));
        result.ref_points.push_back(ref_points[i]);
        result.cur_points.push_back(cur_points[i]);
        result.ref_feature_indices.push_back(valid_ref_indices[i]);
        result.cur_feature_indices.push_back(valid_cur_indices[i]);
    }

    if (result.reproj_valid_num > 0)
    {
        result.mean_reproj_error_ref =
            total_reproj_error_ref / static_cast<double>(result.reproj_valid_num);
        result.mean_reproj_error_cur =
            total_reproj_error_cur / static_cast<double>(result.reproj_valid_num);
    }

    if (parallax_eval_num > 0)
    {
        result.mean_parallax_deg = 
            total_parallax_deg / static_cast<double>(parallax_eval_num);
    }

    if (!ref_points.empty())
    {
        result.good_point_ratio =
            static_cast<double>(result.points_3d.size()) / static_cast<double>(ref_points.size());
    }

    if (result.positive_depth_num > 0)
    {
        result.good_parallax_ratio =
            static_cast<double>(result.good_parallax_num) / static_cast<double>(result.positive_depth_num);
    }

    result.success = 
        !result.points_3d.empty() &&
        result.points_3d.size() >= 30 &&
        result.good_point_ratio >= 0.5 &&
        result.mean_reproj_error_ref <= reproj_err_threshold &&
        result.mean_reproj_error_cur <= reproj_err_threshold &&
        result.mean_parallax_deg >= min_parallax_deg;

    return result;
}

} // namespace mini_orb_slam
