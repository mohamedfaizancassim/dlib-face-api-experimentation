// Dlib_Face_API_Experimentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <vector>
#include <sstream>
#include <future>
#include <Windows.h>
#include "DirectoryFunctions.h"
#include "ImageOptimisations.h"
#include "DLibDeepNeuralNetworks.h"
#include"DlibFaceAlgorithms.h"
#include "FaceRecognition.h"
#include <mmsystem.h>
#define WEBCAM_IDX 1

using namespace std;

//Initialise Video Capture object
cv::VideoCapture vidCap(WEBCAM_IDX);
//Initialise Dlib Image Window
dlib::image_window img_output_window;
//Initialisizng Dlib Face Agorithms
string shapeModelPath = "models/shape_predictor_68_face_landmarks.dat";
string faceRecogModelPath = "models/dlib_face_recognition_resnet_model_v1.dat";
DlibFaceAlgorithms dlibFaceAlgo(shapeModelPath.c_str(), faceRecogModelPath.c_str());



int main()
{
    DirectoryFunctions dir("Known_faces");
    auto filesList = dir.Get_FilesList();
    //FaceRecognition f_recog(filesList);

    bool isMediaPlaying = true;
    

    while (true)
    {
        if (vidCap.isOpened())
        {
            cv::TickMeter tm;
            tm.start();
            cv::Mat frame;
            vidCap >> frame;
            dlib::cv_image<dlib::bgr_pixel> dlib_frame(frame);
            
            img_output_window.set_image(dlib_frame);
           
            auto faces=dlibFaceAlgo.DetectFaces(frame);

            if (faces.size() > 0)
            {
                std::cout << "Faces detected..." << endl;
                keybd_event(VK_SPACE, 0, KEYEVENTF_EXTENDEDKEY, 0);
                PlaySound(TEXT("bell.mp3"), NULL, SND_SYNC); 
                 
               
            }
            else
            {
                std::cout << "No faces detected..." << endl;
                
                
            }
            
            //auto cropedFaces = dlibFaceAlgo.GetCropedFaces(frame,faces);
            
            //auto faceDescriptors = dlibFaceAlgo.GetFaceDescriptors(cropedFaces);

            /*for (int i = 0; i < faceDescriptors.size(); i++)
            {
                string name = f_recog.GetNameFromMatrix(faceDescriptors[i]);
                img_output_window.add_overlay(faces[i], rgb_pixel(0, 255, 0), name);
                cout << "Name: " << name << endl;

            }*/
            /*for (auto descriptor : faceDescriptors)
            {
                cout << "Face Descriptor\r\n---------------------" << endl;
                cout << f_recog.GetNameFromMatrix(descriptor) << endl;
            }*/
            img_output_window.clear_overlay();
            tm.stop();
            cout << "Processing Time (ms): " << tm.getTimeMilli() << endl;

        }
        else
        {
            cout << "Error: Video Capture Initialisation Error." << endl;
            return -1;
        }
    }
   
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
