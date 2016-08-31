#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int compare(const void *a, const void *b)
{
    int *pa = (int*)a;
    int *pb = (int*)b;
    return (*pa )- (*pb);  //从小到大排序
}

int main()
{
    int a[10] = {5, 6, 4, 3, 7, 0 ,8, 9, 2, 1};
    qsort(a, 10, sizeof(int), compare);
    for (int i = 0; i < 10; i++)
        cout << a[i] << " " << endl;
    return 0;
} 