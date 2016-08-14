#include "generial.h"
#include "improved_trajectory/genDescriptors.h"
#include "GMM/getGMM.h"
#include "GMM/getFV.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


int main(int argc, char** argv)
{
	char **fullvideoname = getFullVideoName();
	iline = Malloc(char,max_iline_len);
	int ** classLabel = readLabelFromFile();
    char *actionName[actionType] = {"","AnswerPhone","DriveCar","Eat","FightPerson","GetOutCar","HandShake","HugPerson","Kiss","Run","SitDown","SitUp","StandUp"};
	
	//second //
	//% call this function (genDescriptors) Need a large amount of disk and good IO. This function generate dense features and save to disk.
    //genDescriptors(st,send,fullvideoname,descriptor_path);
	//% create GMM model. Look at this function see if parameters are okay for you.
    //getGMM(fullvideoname,vocabDir,descriptor_path);
    //generate Fisher Vectors
	//getFV(featDir); // compile the fisher.cpp
	//get video darwin~data.
	//getDarwin(featDir,fullvideoname);
	return 0;
}
