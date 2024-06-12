// Invert Image
//
/*
The goal of this program is to take in images and invert the rgb colors

You can specify the # of threads or leave it blank and it will autofill 256

requires opencv to run, I will inlcude the .dll with the .exe


*/
#include "Header.h"


#include "filesystem";
#include <windows.h>


using namespace cv;
using namespace std;


__global__ void InvertMultiImage(uchar* d_data, int size)
{
    int blockID = blockIdx.x
        + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y;

    // global threadID = (threadIndex.n + threadIndex.n+1 * blockdim.n + threadIndex.n+2 * blockdim.n+1 * blockdim.n + ...) 
    //                  + (blockID *blockDim.n * blockDim.n+1 * blockDim.n+2 * ...)
    int threadID = threadIdx.x
        + threadIdx.y * blockDim.x
        + threadIdx.z * blockDim.x * blockDim.y
        + blockID * (blockDim.x * blockDim.y * blockDim.z);


    d_data[threadID] = 255 - d_data[threadID];

}



__host__ void CallCudaKernel(uchar* d_data, size_t rows, size_t columns, int threadsPerBlock)
{
    int blocksPerGrid = (rows * columns + threadsPerBlock - 1) / threadsPerBlock;
    cout << "Executing kernel\n";
    //InvertMultiImage <<<blocksPerGrid, threadsPerBlock >>> (d_data, rows * columns);

}


__host__ string isFolder(string folder)
{
    struct stat s;
    if (stat(folder.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR)
        {
            return folder;
        }
        else if (s.st_mode & S_IFREG)
        {
            printf("File given, directory expected\n");
            exit(0);
        }
        else
        {
            printf("File given, directory expected\n");
            exit(0);
        }
    }
    else
    {
        printf("Error: directory expected\n");
        printf("Using Default folder: &c\n", "../ExampleFolder");
        return("../ExampleFolder/");
    }

}


int main(void)
{
    int threadsperblock = 256; // default thread count
    /* Read the Image in*/
    string folderName;
    printf("Specify # of threads per block: ");
    cin >> threadsperblock;
    if (threadsperblock == 0) {
        threadsperblock = 256;
    }
    printf("(Make sure all images are the same size) \n");
    printf("Specify folder for image inversion: ");
    cin >> folderName;



    folderName = isFolder(folderName);


    std::vector<string> fileNames;
    glob(folderName, fileNames, false);
    Mat singleImage = imread(fileNames.at(0), IMREAD_COLOR);




    /* Allocate Host Memory*/
    cout << "preparing host memory \n";
    size_t cols = 3 * singleImage.cols;
    size_t totalrows = singleImage.rows * fileNames.size();
    size_t size = totalrows * cols;
    int step = 0;
    uchar* h_data = (uchar*)malloc(size);
    /* Read the Images into host memory */
    /* images are read in by row*/
    for (int i = 0; i < fileNames.size(); i++)
    {
        singleImage = imread(fileNames.at(i), IMREAD_COLOR);
        cout << fileNames.at(i) << "\n";
        cout << singleImage.at<uchar>(0, 0) << "\n";
        step = i * singleImage.rows;
        for (int j = 0; j < singleImage.rows; j++)
            memcpy(h_data + cols * (j + step), &singleImage.at<uchar>(j, 0), cols);

    }


    /* Allocate the Device Memory*/
    uchar* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    /* Call the Kernel */
    cout << "calling cuda kernel \n";
    CallCudaKernel(d_data, totalrows, cols, threadsperblock);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    /* write the images back to the folder*/
    printf("Saving Inverted Images \n");
    string newfilename;
    for (int i = 0; i < fileNames.size(); i++)
    {
        step = i * singleImage.rows;
        for (int j = 0; j < singleImage.rows; j++)
            memcpy(&singleImage.at<uchar>(j, 0), h_data + cols * (j + step), cols);

        newfilename = fileNames.at(i).substr(0, fileNames.at(i).find('.'));
        newfilename = newfilename + "_inverted.jpg";
        imwrite(newfilename, singleImage);
    }
    printf("Image Inversion Complete \n");

}




