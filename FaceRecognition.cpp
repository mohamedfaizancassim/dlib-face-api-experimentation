#include "FaceRecognition.h"



void FaceRecognition::Process_TrainingImage(DirectoryFunctions::personImage person)
{
	
}

FaceRecognition::FaceRecognition(std::vector<DirectoryFunctions::personImage>& personImagesVector)
{
	for (auto person : personImagesVector)
	{
		std::cout << "Person: " << person.person << std::endl;
		std::cout << "Person Image: " << person.image << std::endl;


		cv::Mat image = cv::imread(person.image);
		auto faces = this->faceAlgo.DetectFaces(image);
		if (faces.size() > 0)
		{
			auto cropedFaces = this->faceAlgo.GetCropedFaces(image, faces);
			auto matrices = this->faceAlgo.GetFaceDescriptors(cropedFaces);
			this->known_Faces.push_back(FaceRecognition::PersonMatrix{person.person,matrices[0] });
		}
	}
}

std::string FaceRecognition::GetNameFromMatrix(dlib::matrix<float, 0, 1>& inputMatrix)
{
	concurrent_vector<float> recogEdges;
	cv::parallel_for_(cv::Range(0, this->known_Faces.size()), [&](const cv::Range& range)
		{
			for (int i = range.start; i < range.end; i++)
			{
				auto diff = dlib::length(known_Faces[i].matrix - inputMatrix);
				if (diff< 0.5)
				{
					recogEdges.push_back(diff);
				}
			}
		});

	int closedPersonIdx = std::min_element(recogEdges.begin(), recogEdges.end()) - recogEdges.begin();
	
	return this->known_Faces[closedPersonIdx].personName;
}
