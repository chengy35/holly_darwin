#ifndef _GETFV_
#define _GETFV_
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

void getFV(char * featDir)
{
	char *fvFilePath = new char[50];
	char *command = new char[100];
	strcpy(command,"mkdir ");
	strcpy(fvFilePath,featDir);
	strcat(fvFilePath,"trj");
	strcat(command,fvFilePath);
	//cout<<command<<endl;
	system(command);
	strcpy(command,"mkdir ");
	strcpy(fvFilePath,featDir);
	strcat(fvFilePath,"hog");
	strcat(command,fvFilePath);
	//cout<<command<<endl;
	system(command);
	strcpy(command,"mkdir ");
	strcpy(fvFilePath,featDir);
	strcat(fvFilePath,"hof");
	strcat(command,fvFilePath);
	//cout<<command<<endl;
	system(command);
	strcpy(command,"mkdir ");
	strcpy(fvFilePath,featDir);
	strcat(fvFilePath,"mbh");
	strcat(command,fvFilePath);
	//cout<<command<<endl;
	system(command);

	strcpy(command,"./debug/vlfisher ");
	cout<<command<<endl;
	system(command);
}
#endif
