#include "DlibFaceAlgorithms.h"

DlibFaceAlgorithms::DlibFaceAlgorithms(const char* shapeModelPath, const char* faceModelPath)
{
	//Initialise the models
	this->_face_detector = dlib::get_frontal_face_detector();
	dlib::deserialize(shapeModelPath) >> this->_shape_predictor;
	dlib::deserialize(faceModelPath) >> this->_face_recognizer;
}

std::vector<dlib::rectangle> DlibFaceAlgorithms::DetectFaces(cv::Mat& inputFrame)
{
	dlib::cv_image<bgr_pixel> dlib_frame(inputFrame);
	return this->_face_detector(dlib_frame);
}

dlib::full_object_detection DlibFaceAlgorithms::DetectFaceParts(cv::Mat& inputFrame, dlib::rectangle face_rects)
{
	//Convert to OpenCV Frame
	dlib::cv_image<bgr_pixel> dlib_frame(inputFrame);
	return this->_shape_predictor(dlib_frame, face_rects);
}

std::vector<dlib::matrix<rgb_pixel>> DlibFaceAlgorithms::GetCropedFaces(cv::Mat& inputFrame, std::vector<dlib::rectangle>& face_detections)
{
	//The return vector to store croped faces.
	concurrency::concurrent_vector<dlib::matrix<rgb_pixel >> cropedFaces;

	//Convert opencv frame to Dlib frame
	dlib::cv_image<rgb_pixel> dlib_frame(inputFrame);

	cv::parallel_for_(cv::Range(0, face_detections.size()), [&](const cv::Range& range)
		{
			for (int i = range.start; i < range.end; i++)
			{
				auto shape = this->_shape_predictor(dlib_frame, face_detections[i]);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(dlib_frame, get_face_chip_details(shape, 150, 0.25), face_chip);
				cropedFaces.push_back(face_chip);
			}
		});
	
	std::vector<dlib::matrix<rgb_pixel>> stdVecCropedFaces;
	for (auto cropedFace : cropedFaces)
	{
		stdVecCropedFaces.push_back(cropedFace);
	}


	return stdVecCropedFaces;
}

std::vector<dlib::matrix<float, 0, 1>> DlibFaceAlgorithms::GetFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>>&faces_list)
{
	//Check if there is atleast one face in the faces_list vector
	if (faces_list.size() == 0)
	{
		return std::vector<dlib::matrix<float, 0, 1>>();
	}
	return this->_face_recognizer(faces_list);
}

DlibFaceAlgorithms::~DlibFaceAlgorithms()
{
	
}
