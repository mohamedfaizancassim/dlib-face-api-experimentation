#pragma once
//TITLE: Image Optimisations
//Author: Faizan Cassim
/*
* This is a static library full of optimisations that can be done on image matrices to make DNN algorithms more 
* efficient and effective. 
*/
#define IMAGE_WIDTH 480
#define IMAGE_HEIGHT 240
#include<opencv2/opencv.hpp>
#include<dlib/opencv.h>
void Optimize_For_DLIB_Frontal_Face(cv::Mat& inputImg)
{
	// Reduce image size
	cv::resize(inputImg, inputImg, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
	// To Black and White
	//cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2GRAY);
}