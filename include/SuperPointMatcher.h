/**
* This file is part of ORB-SLAM3
*
* Modified to use SuperPoint features and descriptors.
*
* This file declares the SuperPointMatcher class, which is used to match SuperPoint features
* between frames and keyframes.
*
*/

#ifndef SUPERPOINTMATCHER_H
#define SUPERPOINTMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "sophus/sim3.hpp"

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM3
{

class SuperPointMatcher
{
public:

    SuperPointMatcher(float nnratio = 0.8f, bool checkOri = true);

    // Computes the Euclidean distance between two SuperPoint descriptors
    static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th = 3.0f, const bool bFarPoints = false, const float thFarPoints = 50.0f);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const float distThresh);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th, float ratio = 1.0f);

    // Search matches between MapPoints in a KeyFrame and features in a Frame.
    // Brute force constrained to features that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2,
                               std::vector<std::pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0f, const bool bRight = false);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint*> &vpPoints, float th, std::vector<MapPoint *> &vpReplacePoint);

public:

    static const int HISTO_LENGTH = 30;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

} // namespace ORB_SLAM3

#endif // SUPERPOINTMATCHER_H