#pragma once
#include"DirectoryFunctions.h"
#include "DlibFaceAlgorithms.h"
#include "dlib/threads.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <concurrent_vector.h>
#include <future>
#include <thread>
#include<algorithm>

using namespace concurrency;

class FaceRecognition
{
public:
	//This stucture stores the respective face matrix of a person,
	// which can be used to identify a person in the image. 
	static struct PersonMatrix
	{
		std::string personName;
		dlib::matrix<float, 0, 1> matrix;
	};
private:
	//Stores known faces, that are obtained form the training folder.
	concurrent_vector<FaceRecognition::PersonMatrix> known_Faces;
	//Initialise Dlib Face Algorithms
	std::string shapeModelPath = "models/shape_predictor_68_face_landmarks.dat";
	std::string faceRecogModelPath = "models/dlib_face_recognition_resnet_model_v1.dat";
	DlibFaceAlgorithms faceAlgo = DlibFaceAlgorithms(shapeModelPath.c_str(), faceRecogModelPath.c_str());
	void Process_TrainingImage(DirectoryFunctions::personImage person);
public:
	//Takes each image from the trainnig dataset, obtains a matrix, adds the matrix to known_faces.
	FaceRecognition(std::vector<DirectoryFunctions::personImage> &personImagesVector);
	std::string GetNameFromMatrix(dlib::matrix<float, 0, 1>& inputMatrix);
};

