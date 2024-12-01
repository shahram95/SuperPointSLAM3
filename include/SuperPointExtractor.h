#ifndef SUPERPOINTEXTRACTOR_H
#define SUPERPOINTEXTRACTOR_H

#include <vector>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"

namespace ORB_SLAM3 {

class SuperPointExtractor : public FeatureExtractor {
public:
    SuperPointExtractor(const std::string& model_path, 
                       int nfeatures,
                       float scaleFactor,
                       int nlevels,
                       float confidence_threshold = 0.015f);
    
    ~SuperPointExtractor() {}

    // Implement pure virtual functions from FeatureExtractor
    virtual int operator()(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         std::vector<int>& vLappingArea) override;

    virtual std::vector<float> GetScaleFactors() override { return mvScaleFactors; }
    virtual std::vector<float> GetInverseScaleFactors() override { return mvInvScaleFactors; }
    virtual std::vector<float> GetScaleSigmaSquares() override { return mvLevelSigma2; }
    virtual std::vector<float> GetInverseScaleSigmaSquares() override { return mvInvLevelSigma2; }
    
    virtual int GetLevels() override { return mnLevels; }
    virtual float GetScaleFactor() override { return mfScaleFactor; }
    
    virtual const std::vector<cv::Mat>& GetImagePyramid() override { return mvImagePyramid; }

protected:
    virtual void ComputePyramid(cv::Mat image) override;

private:
    // SuperPoint specific methods
    torch::Tensor PreprocessImage(const cv::Mat& image);
    std::vector<cv::KeyPoint> ExtractKeypoints(const torch::Tensor& scores, float threshold);
    cv::Mat ExtractDescriptors(const torch::Tensor& descriptors, const std::vector<cv::KeyPoint>& keypoints);
    void ApplyANMS(std::vector<cv::KeyPoint>& keypoints, const int num_features);

    // Model and parameters
    torch::jit::Module mModel;
    float mConfidenceThreshold;
    int mnFeatures;
    float mfScaleFactor;
    int mnLevels;

    // Scale-space pyramid
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
    std::vector<cv::Mat> mvImagePyramid;
};

} // namespace ORB_SLAM3

#endif // SUPERPOINTEXTRACTOR_H