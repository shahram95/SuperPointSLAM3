#include "SuperPointExtractor.h"
#include <iostream>

namespace ORB_SLAM3 {

SuperPointExtractor::SuperPointExtractor(const std::string& model_path,
                                       int nfeatures,
                                       float scaleFactor,
                                       int nlevels,
                                       float confidence_threshold)
    : mConfidenceThreshold(confidence_threshold),
      mnFeatures(nfeatures),
      mfScaleFactor(scaleFactor),
      mnLevels(nlevels)
{
    try {
        // Load the TorchScript model
        mModel = torch::jit::load(model_path);
        mModel.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading SuperPoint model: " << e.what() << std::endl;
        throw;
    }

    // Compute scale pyramid parameters
    mvScaleFactors.resize(mnLevels);
    mvLevelSigma2.resize(mnLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for(int i=1; i<mnLevels; i++)
    {
        mvScaleFactors[i] = mvScaleFactors[i-1]*mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i]*mvScaleFactors[i];
    }

    mvInvScaleFactors.resize(mnLevels);
    mvInvLevelSigma2.resize(mnLevels);
    for(int i=0; i<mnLevels; i++)
    {
        mvInvScaleFactors[i] = 1.0f/mvScaleFactors[i];
        mvInvLevelSigma2[i] = 1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(mnLevels);
}

int SuperPointExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::OutputArray _descriptors,
                                  std::vector<int>& vLappingArea)
{
    cv::Mat image = _image.getMat();
    if(image.empty())
        return -1;

    // Create grayscale image if necessary
    cv::Mat grayImage;
    if(image.channels() > 1)
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    else
        grayImage = image.clone();

    ComputePyramid(grayImage);

    // Process base level with SuperPoint
    torch::Tensor inputTensor = PreprocessImage(mvImagePyramid[0]);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);
    
    auto output = mModel.forward(inputs).toTuple();
    
    // Extract semi-dense keypoint scores and dense descriptors
    auto scores = output->elements()[0].toTensor();
    auto dense_desc = output->elements()[1].toTensor();
    
    // Extract initial keypoints based on confidence threshold
    keypoints = ExtractKeypoints(scores, mConfidenceThreshold);
    
    // Apply ANMS to limit number of features
    ApplyANMS(keypoints, mnFeatures);
    
    // Extract descriptors for the selected keypoints
    cv::Mat descriptors = ExtractDescriptors(dense_desc, keypoints);
    descriptors.copyTo(_descriptors);

    return keypoints.size();
}

torch::Tensor SuperPointExtractor::PreprocessImage(const cv::Mat& image) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // Create tensor from image
    auto tensor_img = torch::from_blob(float_img.data, 
                                     {1, 1, float_img.rows, float_img.cols},
                                     torch::kFloat32);
    return tensor_img;
}

std::vector<cv::KeyPoint> SuperPointExtractor::ExtractKeypoints(const torch::Tensor& scores, float threshold) {
    std::vector<cv::KeyPoint> keypoints;
    auto scores_acc = scores.accessor<float,4>();
    
    // Get dimensions
    int height = scores.size(2);
    int width = scores.size(3);
    
    // Extract keypoints based on confidence threshold
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            float score = scores_acc[0][0][h][w];
            if(score > threshold) {
                keypoints.push_back(
                    cv::KeyPoint(static_cast<float>(w), 
                                static_cast<float>(h), 
                                8.0f, -1, score));
            }
        }
    }
    
    return keypoints;
}

cv::Mat SuperPointExtractor::ExtractDescriptors(const torch::Tensor& descriptors,
                                              const std::vector<cv::KeyPoint>& keypoints) {
    const int desc_dim = descriptors.size(1);
    cv::Mat desc(keypoints.size(), desc_dim, CV_32F);
    
    auto desc_acc = descriptors.accessor<float,3>();
    
    for(size_t i = 0; i < keypoints.size(); i++) {
        const cv::KeyPoint& kp = keypoints[i];
        int x = static_cast<int>(std::round(kp.pt.x));
        int y = static_cast<int>(std::round(kp.pt.y));
        
        float* desc_row = desc.ptr<float>(i);
        for(int d = 0; d < desc_dim; d++) {
            desc_row[d] = desc_acc[0][d][y * descriptors.size(3) + x];
        }
    }
    
    return desc;
}

void SuperPointExtractor::ApplyANMS(std::vector<cv::KeyPoint>& keypoints, const int num_features) {
    if(keypoints.size() <= num_features)
        return;

    // Sort keypoints by response (confidence)
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    std::vector<cv::KeyPoint> filtered_kps;
    filtered_kps.reserve(num_features);

    // Initialize radius for each keypoint
    std::vector<float> radii(keypoints.size(), std::numeric_limits<float>::max());

    // Compute radius for each keypoint
    for(size_t i = 0; i < keypoints.size(); i++) {
        for(size_t j = 0; j < i; j++) {
            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float dist = dx*dx + dy*dy;
            radii[i] = std::min(radii[i], dist);
            radii[j] = std::min(radii[j], dist);
        }
    }

    // Create indices array
    std::vector<size_t> indices(keypoints.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by radius
    std::sort(indices.begin(), indices.end(),
              [&radii](size_t i1, size_t i2) {
                  return radii[i1] > radii[i2];
              });

    // Select top num_features keypoints
    std::vector<cv::KeyPoint> new_keypoints;
    new_keypoints.reserve(num_features);
    for(size_t i = 0; i < std::min(num_features, (int)indices.size()); i++) {
        new_keypoints.push_back(keypoints[indices[i]]);
    }

    keypoints = std::move(new_keypoints);
}

void SuperPointExtractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < mnLevels; ++level)
    {
        float scale = mvInvScaleFactors[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::resize(image, mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
    }
}

} // namespace ORB_SLAM3