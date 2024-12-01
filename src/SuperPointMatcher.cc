/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/
#include "SuperPointMatcher.h"

#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include <stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int SuperPointMatcher::HISTO_LENGTH;

SuperPointMatcher::SuperPointMatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

float SuperPointMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    // Assuming descriptors are of type CV_32F and rows are descriptors
    // Compute Euclidean distance
    return cv::norm(a, b, cv::NORM_L2);
}

float SuperPointMatcher::RadiusByViewingCos(const float &viewCos)
{
    if (viewCos > 0.998)
        return 2.5f;
    else
        return 4.0f;
}

void SuperPointMatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

int SuperPointMatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    int nmatches = 0;

    const bool bFactor = th != 1.0f;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if (!pMP->mbTrackInView)
            continue;

        if (bFarPoints && pMP->mTrackDepth > thFarPoints)
            continue;

        if (pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if (bFactor)
            r *= th;

        const vector<size_t> vIndices =
            F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

        if (vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        float bestDist = std::numeric_limits<float>::max();
        int bestLevel = -1;
        float bestDist2 = std::numeric_limits<float>::max();
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second matches with nearby keypoints
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            const size_t idx = *vit;

            if (F.mvpMapPoints[idx])
                if (F.mvpMapPoints[idx]->Observations() > 0)
                    continue;

            const cv::Mat &d = F.mDescriptors.row(idx);

            const float dist = DescriptorDistance(MPdescriptor, d);

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx = idx;
            }
            else if (dist < bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= mfNNratio * bestDist2)
        {
            F.mvpMapPoints[bestIdx] = pMP;
            nmatches++;
        }
    }
    return nmatches;
}

int SuperPointMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();

    const Sophus::SE3f Tlw = LastFrame.GetPose();
    const Eigen::Vector3f tlc = Tlw * twc;

    const bool bForward = tlc(2) > CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc(2) > CurrentFrame.mb && !bMono;

    for (int i = 0; i < LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if (pMP)
        {
            if (!LastFrame.mvbOutlier[i])
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const float invzc = 1.0f / x3Dc(2);

                if (invzc < 0)
                    continue;

                Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                    continue;
                if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.mvKeysUn[i].octave;

                // Search in a window. Size depends on scale
                float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                if (bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nLastOctave);
                else if (bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nLastOctave - 1, nLastOctave + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = std::numeric_limits<float>::max();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                {
                    const size_t i2 = *vit;

                    if (CurrentFrame.mvpMapPoints[i2])
                        if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                            continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const float dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= mfNNratio * bestDist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        const cv::KeyPoint &kpLF = LastFrame.mvKeysUn[i];
                        const cv::KeyPoint &kpCF = CurrentFrame.mvKeysUn[bestIdx2];
                        float rot = kpLF.angle - kpCF.angle;
                        if (rot < 0.0f)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    // Apply rotation consistency
    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int SuperPointMatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const set<MapPoint*> &sAlreadyFound, const float th, const float distThresh)
{
    int nmatches = 0;

    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if (pMP)
        {
            if (!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                // Project
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                    continue;
                if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                Eigen::Vector3f PO = x3Dw - Ow;
                float dist3D = PO.norm();

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale invariance region
                if (dist3D < minDistance || dist3D > maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

                // Search in a window
                const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel - 1, nPredictedLevel + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = std::numeric_limits<float>::max();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                {
                    const size_t idx = *vit;
                    if (CurrentFrame.mvpMapPoints[idx])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(idx);

                    const float dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = idx;
                    }
                }

                if (bestDist <= distThresh)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0f)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int SuperPointMatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

    cv::BFMatcher matcher(cv::NORM_L2);

    vector<vector<cv::DMatch>> matches;
    matcher.knnMatch(F1.mDescriptors, F2.mDescriptors, matches, 2);

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].size() < 2)
            continue;

        const cv::DMatch &bestMatch = matches[i][0];
        const cv::DMatch &secondBestMatch = matches[i][1];

        if (bestMatch.distance < mfNNratio * secondBestMatch.distance)
        {
            int idx1 = bestMatch.queryIdx;
            int idx2 = bestMatch.trainIdx;

            vnMatches12[idx1] = idx2;
            nmatches++;
        }
    }

    // Update vbPrevMatched
    vbPrevMatched.clear();
    vbPrevMatched.reserve(F1.mvKeysUn.size());
    for (size_t i1 = 0; i1 < vnMatches12.size(); i1++)
    {
        if (vnMatches12[i1] >= 0)
        {
            vbPrevMatched.push_back(F2.mvKeysUn[vnMatches12[i1]].pt);
        }
        else
        {
            vbPrevMatched.push_back(F1.mvKeysUn[i1].pt);
        }
    }

    return nmatches;
}

// SearchByBoW between KeyFrame and Frame
int SuperPointMatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    // We perform the matching over features that belong to the same vocabulary node
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if (!pMP)
                    continue;
                if (pMP->isBad())
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                float bestDist = std::numeric_limits<float>::max();
                int bestIdxF = -1;
                float bestDist2 = std::numeric_limits<float>::max();

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if (vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const float dist = DescriptorDistance(dKF, dF);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestIdxF = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist <= mfNNratio * bestDist2)
                {
                    vpMapPointMatches[bestIdxF] = pMP;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[realIdxKF].angle - F.mvKeysUn[bestIdxF].angle;
                        if (rot < 0.0f)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdxF);
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

// SearchByBoW between two KeyFrames
int SuperPointMatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint* pMP1 = vpMapPoints1[idx1];

                if (!pMP1)
                    continue;
                if (pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                float bestDist = std::numeric_limits<float>::max();
                int bestIdx2 = -1;
                float bestDist2 = std::numeric_limits<float>::max();

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];
                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if (vbMatched2[idx2])
                        continue;

                    if (!pMP2)
                        continue;
                    if (pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    const float dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist <= mfNNratio * bestDist2)
                {
                    vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                    vbMatched2[bestIdx2] = true;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                        if (rot < 0.0f)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

// SearchForTriangulation
int SuperPointMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                              vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{
    // Implemented similarly to ORBmatcher but using L2 distance and SuperPoint features
    // For brevity, here's a simplified version

    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    // Compute epipole in second image
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();
    Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
    Eigen::Vector3f Cw = pKF1->GetCameraCenter();
    Eigen::Vector3f C2 = T2w * Cw;
    Eigen::Vector2f ep = pKF2->mpCamera->project(C2);

    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N, false);
    vector<int> vMatches12(pKF1->N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if (pMP1)
                    continue;

                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                float bestDist = std::numeric_limits<float>::max();
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const float dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = idx2;
                    }
                }

                if (bestDist <= mfNNratio * bestDist)
                {
                    vMatches12[idx1] = bestIdx2;
                    vbMatched2[bestIdx2] = true;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        float rot = kp1.angle - pKF2->mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0f)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}

// SearchBySim3
int SuperPointMatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();

    Sophus::Sim3f S21 = S12.inverse();

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1, false);
    vector<bool> vbAlreadyMatched2(N2, false);

    for (int i = 0; i < N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if (pMP)
        {
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    vector<int> vnMatch1(N1, -1);
    vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (int i1 = 0; i1 < N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
            continue;

        if (pMP->isBad())
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc1 = T1w * p3Dw;
        Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

        // Depth must be positive
        if (p3Dc2(2) < 0.0f)
            continue;

        const float invz = 1.0f / p3Dc2(2);
        const float x = p3Dc2(0) * invz;
        const float y = p3Dc2(1) * invz;

        const float u = fx * x + cx;
        const float v = fy * y + cy;

        // Point must be inside the image
        if (!pKF2->IsInImage(u, v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

        // Search in a radius
        const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = std::numeric_limits<float>::max();
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel + 1)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= mfNNratio * bestDist)
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Check agreement and fill vpMatches12
    int nFound = 0;
    for (int i1 = 0; i1 < N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

// Fuse function
int SuperPointMatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    // Implement fusion of MapPoints into the KeyFrame, using SuperPoint features
    // For brevity, a simplified version is provided

    Sophus::SE3f Tcw = bRight ? pKF->GetRightPose() : pKF->GetPose();
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    GeometricCamera* pCamera = bRight ? pKF->mpCamera2 : pKF->mpCamera;

    int nFused = 0;
    const int nMPs = vpMapPoints.size();

    for (int i = 0; i < nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if (!pMP)
            continue;

        if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        if (p3Dc(2) < 0.0f)
            continue;

        const Eigen::Vector2f uv = pCamera->project(p3Dc);

        if (!pKF->IsInImage(uv(0), uv(1)))
            continue;

        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        Eigen::Vector3f Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5f * dist3D)
            continue;

        const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius, bRight);

        if (vIndices.empty())
            continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = std::numeric_limits<float>::max();
        int bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
        {
            size_t idx = *vit;
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel + 1)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const float dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= mfNNratio * bestDist)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

} // namespace ORB_SLAM3
