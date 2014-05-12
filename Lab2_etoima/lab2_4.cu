/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>


#define FILTER_RADIUS 	16
#define FILTER_LENGTH 	(2 * FILTER_RADIUS + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	5e-4
#define TILE_WIDTH      96
#define TILE_HEIGHT     96



__constant__  float Filter[FILTER_LENGTH];   // gia na apothikeusw tous suntelestes tou filtrou
 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
  
                      
  for (y = filterR; y < imageH-filterR; y++) {
    for (x = filterR; x < imageW-filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        
        sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
           
        h_Dst[y * imageW + x] = sum;  
      }
    }
  }
   
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  
  int x, y, k;
  
  for (y = filterR; y < imageH-filterR; y++) {
    for (x = filterR; x < imageW-filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        
         sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        
        h_Dst[y * imageW + x] = sum;  
      }
    }
  }

 
    
}

///// Reference row convolution filter in GPU   /////

__global__ void row_Kernel(float *Dst, float *Src, float *filter,int imageW, int imageH, int filterR) {

     int k;
    
     float sum=0; //value to store the element of the matrix that is computed by the thread

    if(!((threadIdx.x+blockDim.x*blockIdx.x) < filterR  || (threadIdx.x+blockDim.x*blockIdx.x) >= imageH-filterR)){
		for(k = -filterR; k<=filterR; k++){
		
			int d = threadIdx.x + k+blockDim.x * blockIdx.x;	
			sum += Src[(blockIdx.y * blockDim.y + threadIdx.y)*imageW + d]*filter[filterR-k];	
	                
		}
	}
      Dst[(blockIdx.y * blockDim.y + threadIdx.y)*imageW + threadIdx.x + blockDim.x*blockIdx.x] = sum;
	
       
}

///// Reference column convolution filter in GPU  /////


__global__ void column_Kernel(float *Dst, float *Src, float *filter,int imageW, int imageH, int filterR) {

      int k;
     
      float sum=0; //value to store the element of the matrix that is computed by the thread

   if(!((threadIdx.y+blockDim.y*blockIdx.y) < filterR  || (threadIdx.y+blockDim.y*blockIdx.y) >= imageH-filterR)){
		for(k = -filterR; k<=filterR; k++){

			int d = k+ (blockIdx.y * blockDim.y + threadIdx.y);
		        sum += Src[d*imageW + threadIdx.x+blockDim.x * blockIdx.x]*filter[filterR-k];

		}
	}
	Dst[(blockIdx.y * blockDim.y + threadIdx.y)*imageW + threadIdx.x+blockDim.x*blockIdx.x] = sum;



}


///// Reference tiled row convolution filter in GPU   /////

__global__ void tiled_row_Kernel(float *Dst, float *Src,int imageW, int imageH, int filterR) {

     int k;
     
     float sum=0; //value to store the element of the matrix that is computed by the thread

     
     // allocate 1D tile in __shared__ memory
    __shared__ float data[TILE_HEIGHT * (FILTER_LENGTH - 1 +TILE_WIDTH)];
 
    
    //h global adress autou tou thread
    const int adr =  (blockIdx.y * blockDim.y + threadIdx.y)*imageW + (threadIdx.x +blockDim.x * blockIdx.x);  

    //xrhsimopoiw to shift gia na kinoumaste katallhla afou twra exoume 1D tile
    //kai na grafoume sth swsth thesh
    const int shift = threadIdx.y * (TILE_WIDTH + FILTER_LENGTH -1 );

    // bash eikonas
    const int x0 = threadIdx.x + (blockDim.x * blockIdx.x);

    //periptwseis gia na feroume sth share mem ta swsta kommatia tile apo to Src.elegxoume an h eikona einai mesa sta oria pou theloume

    
    //periptwsh 1
    if(!(x0 < filterR)){
        data[threadIdx.x + shift ] = Src[ adr - filterR];
    }
    else{
        data[threadIdx.x  + shift ] = 0;
    }

  
    //periptwsh 2
    if(!( x0 >= imageH - filterR)){
        data[ threadIdx.x + blockDim.x + shift ] = Src[ adr + filterR];
    }
    else{
       data[ threadIdx.x + blockDim.x + shift ] = 0;
    }

    __syncthreads();   //barrier giati theloume na fortwsoume ola ta dedomena apo thn global protou proxwrhsoume

    

     //convolution

    for ( k = -filterR; k <= filterR; k++){
        sum += data[ threadIdx.x + filterR + k + shift ] * Filter[ filterR - k];
    }


    Dst[ adr ] = sum;


	
       
}

///// Reference tiled_column convolution filter in GPU  /////


__global__ void tiled_column_Kernel(float *Dst, float *Src,int imageW, int imageH, int filterR) {

     int k;
     
     float sum=0; //value to store the element of the matrix that is computed by the thread
     
     
     // allocate 1D tile in __shared__ memory
    __shared__ float data[TILE_WIDTH * (TILE_HEIGHT + FILTER_LENGTH - 1)];


    //h global adress autou tou thread
    const int adr =  (blockIdx.y * blockDim.y + threadIdx.y)*imageW + (threadIdx.x +blockDim.x * blockIdx.x);

   
    //xrhsimopoiw to shift gia na kinoumaste katallhla afou twra exoume 1D tile
    //kai na grafoume sth swsth thesh
    //shift_1 gia thn panw periptwsh(periptwsh 1)
    //shift_2 gia thn katw periptwsh(periptwsh 2)
    const int shift_1 = threadIdx.y * (TILE_WIDTH);
    const int shift_2 = shift_1 + (blockDim.y * TILE_WIDTH);

    //bash eikonas
    const int y0 = threadIdx.y + (blockIdx.y * blockDim.y); 


    //periptwseis gia na feroume sth share mem ta swsta kommatia tile apo to Src.elegxoume an h eikona einai mesa sta oria pou theloume

    
    //periptwsh 1
    if(!(y0 < filterR)){

       data[ threadIdx.x + shift_1 ] = Src[ adr - (imageW * filterR)];

    }
    else {
       data[ threadIdx.x + shift_1 ] = 0;
    }



   //periptwsh 2
   if(!(y0 >= imageH - filterR)){
      data[ threadIdx.x + shift_2 ] = Src[ adr + (imageW * filterR)];
   }
   else{
      data[ threadIdx.x + shift_2 ] = 0;
   }

   __syncthreads();    //barrier giati theloume na fortwsoume ola ta dedomena apo thn global protou proxwrhsoume

  



    //convolution
   
   for (k = -filterR; k <= filterR; k++){
     sum += data[ (threadIdx.y + filterR) * TILE_WIDTH + threadIdx.x +  (k * TILE_WIDTH)] * Filter[ filterR - k ];
   }


   Dst[ adr ] = sum;

  
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *d_Input,     //eikona eisodou sto device
    *d_OutputGPU1,*d_OutputGPU2,  //apotelesma apo to device gpu
    *d_Filter,    //filtro sto device 
    *h_OutputGPU1,*h_OutputGPU2,      //to epistrefomeno apotelesma apo thn gpu sto host
    *d_Buffer;       //Buffer sto device gia endiameso apotelesma apo thn row sth column ston kernel

   

    int imageW,newW;
    int imageH,newH;
    unsigned int i,j;
    cudaEvent_t start_GPU1,start_GPU2;  //var gia na metrisw xrono sth gpu
    cudaEvent_t stop_GPU1,stop_GPU2;   //var gia na metrisw xrono sth gpu 
    float elapsed_GPU1,elapsed_GPU2;  //xronos sth gpu
    float average_1=0;  //mesos xronos gia thn gpu xwris to tile
    float average_2=0;  //mesos xronso gia thn gpu me to tile
    timeval t1;     //gia na metrisw to xrono sth cpu
    timeval t2;     //gia na metrisw to xrono sth cpu
    double elapsed_CPU;  // xronos sth cpu
    cudaError_t err;   // elegxos gia cuda malloc kai cudamemcopy


    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  
    if(argc <= 1){
       printf("I have to terminate because you didnt enter image size.Pls try again \n");
       return 1;
    }
    imageW=atoi(argv[1]);
    printf("You entered image size. Should be a power of two and greater than %d\n", FILTER_LENGTH);
    if( imageW <= FILTER_LENGTH){
       printf("I have to terminate because you enter image smaller than filter\n");
       return 1;
    }
    imageH = imageW;

    
    newH = imageH + FILTER_LENGTH -1;
    newW = imageW + FILTER_LENGTH -1;
  
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(newW * newH * sizeof(float));
    h_Buffer    = (float *)malloc(newW * newH * sizeof(float));
    h_OutputCPU = (float *)malloc(newW * newH * sizeof(float));
    h_OutputGPU1 = (float *)malloc(newW * newH * sizeof(float));
    h_OutputGPU2 = (float *)malloc(newW * newH * sizeof(float));



    // if either memory allocation failed, report an error message
    if(h_Filter == 0 || h_Input == 0 || h_Buffer == 0 || h_OutputCPU == 0 || h_OutputGPU1 == 0 || h_OutputGPU2 == 0 )
    {
       printf("couldn't allocate memory\n");
       return 1;
    }
    
    cudaSetDevice(0/*cutGetMaxGflopsDevice()*/);

    // Allocate memory on device
    err = cudaSuccess;
    err = cudaMalloc((void**)&d_Input, newW * newH * sizeof(float));
    err = cudaMalloc((void**)&d_OutputGPU1, newW * newH * sizeof(float));
    err = cudaMalloc((void**)&d_OutputGPU2, newW * newH * sizeof(float));
    err = cudaMalloc((void**)&d_Filter, ((2*FILTER_RADIUS)+1)*sizeof(float)); 
    err = cudaMalloc((void**)&d_Buffer, newW * newH * sizeof(float)); 
    
    // if either memory allocation failed, report an error message

    if( err != cudaSuccess) {
       printf("CUDA Error in allocation memory on device: %s\n", cudaGetErrorString(err));
       return 1;
    }  
    


    

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

   //arxikopoiw me floats wste na exw megaluterh anakriveia sta apotelesmata mou
    srand(time(NULL));
       	
    for (i = 0; i < FILTER_LENGTH; i++)
    {
        h_Filter[i] = (float)(rand() / (float)RAND_MAX);
	
     }

    
    for (i = 0; i < newW; i++){	
	for(j=0; j < newH; j++){
		if(i<FILTER_RADIUS || j<FILTER_RADIUS  || i >= (imageW+FILTER_RADIUS) || j>=(imageH+FILTER_RADIUS)){		
		       
			 
			h_Input[j*newW+i] = 0;
		
		}
		else{
		      h_Input[j*newH+i] = (float)(rand()/(float)RAND_MAX);			
		}
		
        	h_Buffer[j*newW+i]=0;
		h_OutputCPU[j*newH+i]=0;
	}
	 	       
    }

      
     
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    ///// cpu events gia metrisi xronou  /////
     
    gettimeofday(&t1, NULL);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, newW, newH, FILTER_RADIUS); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, newW, newH, FILTER_RADIUS); // convolution kata sthles

    ///// cpu events gia metrisi xronou  /////
    gettimeofday(&t2, NULL);
    elapsed_CPU = (t2.tv_sec - t1.tv_sec) +  ((t2.tv_usec - t1.tv_usec)/1000000.0);
    printf("CPU elapsed time:%f sec\n",elapsed_CPU);

    

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    
    printf("GPU computation...\n");

    
    //Load h_Input and h_Filter to device memory
     err = cudaMemcpy(d_Input,h_Input,newW * newH * sizeof(float),cudaMemcpyHostToDevice);
     err = cudaMemcpy(d_Filter,h_Filter,((2*FILTER_RADIUS)+1)*sizeof(float),cudaMemcpyHostToDevice);  
     err = cudaMemcpyToSymbol(Filter,h_Filter,((2*FILTER_RADIUS)+1)*sizeof(float),0,cudaMemcpyHostToDevice);

     if( err != cudaSuccess) {
         printf("CUDA Error in loading memory on device: %s\n", cudaGetErrorString(err));
         return 1;
    }  
     
    // Kernel Invocation
    // Setup the execution configuration
          
     dim3 dimBlock;
     dimBlock.x=16;
     dimBlock.y=16;
     dim3 dimGrid(newW/dimBlock.x,newH/dimBlock.y);

     dim3 dimBlock_tiled;
     dimBlock_tiled.x=32;
     dimBlock_tiled.y=32;
     dim3 dimGrid_tiled(newW/dimBlock_tiled.x,newH/dimBlock_tiled.y);
     
     
    /////  cuda events for gpu time calculation /////
    cudaEventCreate(&start_GPU1);
    cudaEventCreate(&stop_GPU1);
    

    //Launch the device
    cudaFuncSetCacheConfig(row_Kernel,cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(column_Kernel,cudaFuncCachePreferL1);

    for(int i = 0; i < 10; ++ i){
       cudaEventRecord(start_GPU1, 0);
       cudaDeviceSynchronize();
       row_Kernel<<<dimGrid,dimBlock>>>(d_Buffer,d_Input,d_Filter,newW,newH,FILTER_RADIUS);
       cudaDeviceSynchronize();
       column_Kernel<<<dimGrid,dimBlock>>>(d_OutputGPU1,d_Buffer,d_Filter,newW,newH,FILTER_RADIUS);
       cudaDeviceSynchronize();


     /////  cuda events for gpu time calculation  /////
      cudaEventRecord(stop_GPU1, 0);
      cudaEventSynchronize(stop_GPU1);
      cudaEventElapsedTime(&elapsed_GPU1, start_GPU1, stop_GPU1);
  
      average_1 += elapsed_GPU1;
    }

    average_1 /= 10; 
    
    cudaEventDestroy(start_GPU1);
    cudaEventDestroy(stop_GPU1);



    
    
    /////  cuda events for tiled gpu time calculation /////
    cudaEventCreate(&start_GPU2);
    cudaEventCreate(&stop_GPU2);
    

    //Launch the tiled device
    cudaFuncSetCacheConfig(tiled_row_Kernel,cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(tiled_column_Kernel,cudaFuncCachePreferShared);
    
    for(int i = 0; i < 10; ++i ){
       cudaEventRecord(start_GPU2, 0);
       cudaDeviceSynchronize();
       tiled_row_Kernel<<<dimGrid_tiled,dimBlock_tiled>>>(d_Buffer,d_Input,newW,newH,FILTER_RADIUS);
       cudaDeviceSynchronize();
       tiled_column_Kernel<<<dimGrid_tiled,dimBlock_tiled>>>(d_OutputGPU2,d_Buffer,newW,newH,FILTER_RADIUS);
       cudaDeviceSynchronize();


   /////  cuda events for tiled gpu time calculation  /////
       cudaEventRecord(stop_GPU2, 0);
       cudaEventSynchronize(stop_GPU2);
       cudaEventElapsedTime(&elapsed_GPU2, start_GPU2, stop_GPU2);
     
       average_2 += elapsed_GPU2 ;

    }

    average_2 /= 10;    

    cudaEventDestroy(start_GPU2);
    cudaEventDestroy(stop_GPU2);


    // ask CUDA for the last error to occur (if one exists)
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
       printf("CUDA Error: %s\n", cudaGetErrorString(error));
       return 1;
    } 

    //Read d_OutputGPU1 and d_OutputGPU2 from the device
    err = cudaMemcpy(h_OutputGPU1,d_OutputGPU1,newW * newH * sizeof(float),cudaMemcpyDeviceToHost);
    err = cudaMemcpy(h_OutputGPU2,d_OutputGPU2,newW * newH * sizeof(float),cudaMemcpyDeviceToHost);

    if( err != cudaSuccess) {
       printf("CUDA Error in reading memory from device: %s\n", cudaGetErrorString(err));
       return 1;
    }  


    printf("GPU elapsed time:%f sec \n ",average_1/1000);
    printf("--tiled--GPU elapsed time:%f sec \n ",average_2/1000);

    

    printf("1.GPU:%d=%f\n",imageW * imageH-1,h_OutputGPU1[imageW * imageH-1]); 
    printf("2.--tiled--GPU:%d=%f\n",imageW * imageH-1,h_OutputGPU2[imageW * imageH-1]); 
    printf("3.CPU:%d=%f\n",imageW * imageH-1,h_OutputCPU[imageW * imageH-1]);  
      

    
    
    // CPU Vs GPU (comparison) //
    

    for(i = 0; i< newW * newH; i++){
       if ( ABS(h_OutputCPU[i] - h_OutputGPU1[i]) >= accuracy) {
          printf("ERROR at element i:%d , accuracy error so i have to terminate sorry \n",i);
          return 1;
       }
       

    }
     
    
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(h_OutputGPU1);
    free(h_OutputGPU2);
    cudaFree(d_Input);
    cudaFree(d_Filter);
    cudaFree(d_OutputGPU1);
    cudaFree(d_OutputGPU2);
    cudaFree(d_Buffer);



     printf("success !!!! \n");

    


    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
     cudaDeviceReset();


    return 0;
}
