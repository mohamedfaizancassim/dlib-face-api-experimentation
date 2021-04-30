#pragma once
#include<dlib/dir_nav.h>
#include<vector>
#include<sstream>
#include <iostream>

class DirectoryFunctions
{
public:
	static struct personImage
	{
		std::string person;
		std::string image;
	};
private:
	dlib::directory dir;
	std::vector<dlib::directory> dirs_list;
	std::vector<DirectoryFunctions::personImage> files_list;
public:
	DirectoryFunctions(const char* folderPath);
	
	std::vector<DirectoryFunctions::personImage> Get_FilesList();
	
};

