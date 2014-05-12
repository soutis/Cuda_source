#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"



PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result,d_img_in;
    int hist[256],*d_hist;
    cudaError_t err;
    

   // h_lut = (int *)malloc(sizeof(int)*256);
   
    //arxikopoiw to hist me mhdenika
    for(int i = 0;i<256;i++){

      hist[i] = 0;
      

    }

    
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    cudaSetDevice(0/*cutGetMaxGflopsDevice()*/);


    // Allocate memory on device
    err = cudaSuccess;
    err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));
    err = cudaMalloc((void**)&d_img_in.img, img_in.w * img_in.h * sizeof(unsigned char));
    
    
  
    // if either memory allocation failed, report an error message
    if( err != cudaSuccess) {
       printf("CUDA Error in allocation memory on device: %s\n", cudaGetErrorString(err));
      
    } 


    //Load  to device memory
    err = cudaMemcpy(d_hist,hist,256 * sizeof(int),cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_img_in.img,img_in.img,img_in.w * img_in.h * sizeof(unsigned char),cudaMemcpyHostToDevice);
    

    if( err != cudaSuccess) {
         printf("CUDA Error in loading memory on device: %s\n", cudaGetErrorString(err));
       
    }  


    // Kernel Invocation
    // Setup the execution configuration
      
     int size = (img_in.w * img_in.h);    
     dim3 dimBlock = 256; //afou ta bins einai 256
     dim3 dimGrid = size/256;
     

    //Launch the device
    cudaDeviceSynchronize();
    histogram<<<dimGrid,dimBlock>>>(d_hist,d_img_in.img,size);
    cudaDeviceSynchronize();
    

    //Read  from the device
    err = cudaMemcpy(hist,d_hist,256 * sizeof(int),cudaMemcpyDeviceToHost);
     

    if( err != cudaSuccess) {
       printf("CUDA Error in reading memory from device: %s\n", cudaGetErrorString(err));
     
    }     
  
   histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    

    //Free memory
    cudaFree(d_hist);
    cudaFree(d_img_in.img);
    

    return result;
}
