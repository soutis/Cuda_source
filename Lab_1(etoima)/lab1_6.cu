/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>


#define FILTER_RADIUS 	8
#define FILTER_LENGTH 	(2 * FILTER_RADIUS + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	5e-14
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }   
        h_Dst[y * imageW + x] = sum;  
      }
    }
  }
   
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  
  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        } 
        h_Dst[y * imageW + x] = sum;  
      }
    }
  }

 
    
}

///// Reference row convolution filter in GPU   /////

__global__ void row_Kernel(double *Dst, double *Src, double *Filter,int imageW, int imageH, int filterR) {

     int k;
 
     double sum=0; //value to store the element of the matrix that is computed by the thread

        for( k= -filterR; k <= filterR; k++){
         int d = blockIdx.x * blockDim.x + threadIdx.x + k;

          if(d >= 0 && d < imageW) {
            sum += Src[(blockIdx.y * blockDim.y + threadIdx.y) * imageW + d] * Filter[filterR - k];    
          }	    
          Dst[(blockIdx.y * blockDim.y + threadIdx.y) * imageW + blockIdx.x * blockDim.x + threadIdx.x]=sum;
         }
     
       
   
   
}

///// Reference column convolution filter in GPU  /////


__global__ void column_Kernel(double *Dst, double *Src, double *Filter,int imageW, int imageH, int filterR) {

     int k;

  
   
      double sum=0; //value to store the element of the matrix that is computed by the thread

      for( k= -filterR; k <= filterR; k++){
	int d =blockIdx.y * blockDim.y + threadIdx.y+ k;

        if(d >= 0 && d < imageH) {
           sum += Src[d * imageW + blockIdx.x * blockDim.x + threadIdx.x] * Filter[filterR - k];    
        }	
        Dst[(blockIdx.y * blockDim.y + threadIdx.y) * imageW + blockIdx.x * blockDim.x + threadIdx.x]=sum;  
       }
  
        

}




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *d_Input,     //eikona eisodou sto device
    *d_OutputGPU,  //apotelesma apo to device gpu
    *d_Filter,    //filtro sto device 
    *h_OutputGPU,      //to epistrefomeno apotelesma apo thn gpu sto host
    *d_Buffer;       //Buffer sto device gia endiameso apotelesma apo thn row sth column ston kernel

  

    int imageW;
    int imageH;
    unsigned int i;
    cudaEvent_t start_GPU;  //var gia na metrisw xrono sth gpu
    cudaEvent_t stop_GPU;   //var gia na metrisw xrono sth gpu 
    float elapsed_GPU;  //xronos sth gpu
    timeval t1;     //gia na metrisw to xrono sth cpu
    timeval t2;     //gia na metrisw to xrono sth cpu
    double elapsed_CPU;  // xronos sth cpu


    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

    

    // Allocate memory on device
    cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(double));
    cudaMalloc((void**)&d_OutputGPU, imageW * imageH * sizeof(double));
    cudaMalloc((void**)&d_Filter, ((2*FILTER_RADIUS)+1)*sizeof(double)); 
    cudaMalloc((void**)&d_Buffer, imageW * imageH * sizeof(double));


    // if either memory allocation failed, report an error message
    if(h_Filter == 0 || h_Input == 0 || h_Buffer == 0 || h_OutputCPU == 0 || h_OutputGPU == 0 || d_Input ==0 
    || d_OutputGPU == 0 || d_Filter == 0 || d_Buffer == 0)
    {
     printf("couldn't allocate memory\n");
     return 1;
    }
    

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

   //arxikopoiw me floats wste na exw megaluterh anakriveia sta apotelesmata mou
   srand(time(NULL));
       	
    for (i = 0; i < FILTER_LENGTH; i++)
    {
        h_Filter[i] = (double)(rand() / (double)RAND_MAX);
	
		
    }

    
    for (i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (double)(rand() / (double)RAND_MAX);
	
    }

      
     
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    ///// cpu events gia metrisi xronou  /////
     
     gettimeofday(&t1, NULL);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, FILTER_RADIUS); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, FILTER_RADIUS); // convolution kata sthles

    ///// cpu events gia metrisi xronou  /////
    gettimeofday(&t2, NULL);
    elapsed_CPU = (t2.tv_sec - t1.tv_sec) +  ((t2.tv_usec - t1.tv_usec)/1000000.0);
    printf("CPU elapsed time:%f sec\n",elapsed_CPU);

    

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    
    printf("GPU computation...\n");


    /////  cuda events for gpu time calculation /////
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);
    


  

    //Load h_Input and h_Filter to device memory
     cudaMemcpy(d_Input,h_Input,imageW * imageH * sizeof(double),cudaMemcpyHostToDevice);
     cudaMemcpy(d_Filter,h_Filter,((2*FILTER_RADIUS)+1)*sizeof(double),cudaMemcpyHostToDevice);
     
    // Kernel Invocation
    // Setup the execution configuration
    dim3 dimGrid;      
    dim3 dimBlock;
    dimBlock.x=4;
    dimBlock.y=4;
    dimGrid.x=imageW/dimBlock.x;
    dimGrid.y=imageH/dimBlock.y;
    
    

    //Launch the device
    cudaDeviceSynchronize();
    row_Kernel<<<dimGrid,dimBlock>>>(d_Buffer,d_Input,d_Filter,imageW,imageH,FILTER_RADIUS);
    cudaDeviceSynchronize();
    column_Kernel<<<dimGrid,dimBlock>>>(d_OutputGPU,d_Buffer,d_Filter,imageW,imageH,FILTER_RADIUS);
    cudaDeviceSynchronize();


    // ask CUDA for the last error to occur (if one exists)
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
   
    printf("CUDA Error: %s\n", cudaGetErrorString(error));

    
    return 1;
    } 

    //Read d_OutputGPU from the device
    cudaMemcpy(h_OutputGPU,d_OutputGPU,imageW * imageH * sizeof(double),cudaMemcpyDeviceToHost);


    /////  cuda events for gpu time calculation  /////
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(stop_GPU);
    cudaEventElapsedTime(&elapsed_GPU, start_GPU, stop_GPU);

    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);


  

    printf("GPU elapsed time:%f sec\n ",elapsed_GPU/1000);

    

    printf("1.GPU:%d=%f\n",imageW * imageH-1,h_OutputGPU[imageW * imageH-1]); 
    printf("2.CPU:%d=%f\n",imageW * imageH-1,h_OutputCPU[imageW * imageH-1]);  
      

    
    
    // CPU Vs GPU (comparison) //
    

    for(i = 0; i< imageW * imageH; i++){
       if ( ABS(h_OutputCPU[i] - h_OutputGPU[i]) >= accuracy ) {
          printf("ERROR at element i:%d , accuracy error so i have to terminate sorry \n",i);
          return 1;
       }

    }
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(h_OutputGPU);
    cudaFree(d_Input);
    cudaFree(d_Filter);
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);



     printf("success !!!! \n");

    


    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
     cudaDeviceReset();


    return 0;
}
