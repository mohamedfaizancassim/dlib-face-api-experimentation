#pragma once
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <concurrent_vector.h>
#include "DLibDeepNeuralNetworks.h"
class DlibFaceAlgorithms
{
private:
	dlib::frontal_face_detector _face_detector;
	dlib::shape_predictor _shape_predictor;
	anet_type _face_recognizer;
public:
	DlibFaceAlgorithms(const char* shapeModelPath, const char* faceModelPath);
	//Detects faces in the prescribed image.
	std::vector<dlib::rectangle> DetectFaces(cv::Mat& inputFrame);
	//Detects the parts of the face. 
	dlib::full_object_detection DetectFaceParts(cv::Mat& inputFrame, dlib::rectangle face);
	//Crops the Face Image from the detected Faces.
	std::vector <dlib::matrix <rgb_pixel>> GetCropedFaces(cv::Mat& inputFrame, std::vector<dlib::rectangle>& face_detections);
	std::vector<dlib::matrix<float, 0, 1>> GetFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>> &faces_list);
	~DlibFaceAlgorithms();
};

