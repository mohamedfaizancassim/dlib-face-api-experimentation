#include "DirectoryFunctions.h"

DirectoryFunctions::DirectoryFunctions(const char* folderPath)
{
	dir = dlib::directory(folderPath);
	dirs_list = dir.get_dirs();
	//Shift through all name directories
	for (auto _nameDir : dirs_list)
	{
		std::stringstream subDirPath;
		subDirPath << folderPath << "/" << _nameDir.name();
		//Create directory object to shift through images
		dlib::directory _subdir(subDirPath.str());
		//Run though each image file
		for (auto _imagefile : _subdir.get_files())
		{

			files_list.push_back(DirectoryFunctions::personImage{ _nameDir.name(),_imagefile.full_name() });
		}
	}
}

std::vector<DirectoryFunctions::personImage> DirectoryFunctions::Get_FilesList()
{
	return this->files_list;
}
