#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


__global__ void histogram(int * hist_out, unsigned char * img_in,int size){

     __shared__  int data[256];
  
     data[threadIdx.x] = 0;
     __syncthreads();

     int i = threadIdx.x + blockIdx.x * blockDim.x;
     const int shift = blockDim.x * gridDim.x;
     while (i < size)
     {
              atomicAdd( &data[img_in[i]], 1);
              i += shift;
     }
     __syncthreads();


    atomicAdd( &(hist_out[threadIdx.x]), data[threadIdx.x] );
    
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
   

     
    
    int  i,cdf, min, d;
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    /* Construct the LUT by calculating the CDF */

    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */

    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}
