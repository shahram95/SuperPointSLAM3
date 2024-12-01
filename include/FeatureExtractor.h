#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3 {

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    virtual int operator()(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         std::vector<int>& vLappingArea) = 0;

    virtual std::vector<float> GetScaleFactors() = 0;
    virtual std::vector<float> GetInverseScaleFactors() = 0;
    virtual std::vector<float> GetScaleSigmaSquares() = 0;
    virtual std::vector<float> GetInverseScaleSigmaSquares() = 0;
    
    virtual int GetLevels() = 0;
    virtual float GetScaleFactor() = 0;

    // Access to image pyramid
    virtual const std::vector<cv::Mat>& GetImagePyramid() = 0;

protected:
    // Common utilities that might be needed by derived classes
    virtual void ComputePyramid(cv::Mat image) = 0;
};

} //namespace ORB_SLAM3

#endif //FEATUREEXTRACTOR_H