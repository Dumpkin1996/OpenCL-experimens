#define __CL_ENABLE_EXCEPTIONS    
#include "util.hpp" 

#include <CL/cl.hpp>
#include <vector>
#include <string>

#include <iostream>
#include <fstream>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

int LENGTH = 1000000;

const char * KernelSource = "\n" \
"__kernel void vadd(                                                    \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";  

int main(void)
{
    util::Timer timer;

    for (int j = 0; j < 100; j++) {
    	
    	int err;

    	size_t global;                  // global domain size

	    float* h_a = (float*) calloc(LENGTH, sizeof(float));   
	    float* h_b = (float*) calloc(LENGTH, sizeof(float));  
	    float* h_c = (float*) calloc(LENGTH, sizeof(float));   

	    cl_device_id     device_id;     // compute device id
	    cl_context       context;       // compute context
	    cl_command_queue commands;      // compute command queue
	    cl_program       program;       // compute program
	    cl_kernel        ko_vadd;       // compute kernel

	    cl_mem d_a;                     // device memory used for the input  a vector
	    cl_mem d_b;                     // device memory used for the input  b vector
	    cl_mem d_c;                     // device memory used for the output c vector

	    // Fill vectors a and b with random float values
	    int count = LENGTH;
	    for(int i = 0; i < count; i++){
	        h_a[i] = rand() / (float)RAND_MAX;
	        h_b[i] = rand() / (float)RAND_MAX;
	    }

	    // Set up platform and GPU device
	    cl_uint numPlatforms;

	    // Find number of platforms
	    err = clGetPlatformIDs(0, NULL, &numPlatforms);

	    // Get all platforms
	    cl_platform_id Platform[numPlatforms];
	    err = clGetPlatformIDs(numPlatforms, Platform, NULL);

	    // Secure a GPU
	    for (int i = 0; i < numPlatforms; i++) {
	        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
	        if (err == CL_SUCCESS) {
	            break;
	        }
	    }

	    // Create a compute context
	    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);;

	    // Create a command queue
	    commands = clCreateCommandQueue(context, device_id, 0, &err);

	    // Create the compute program from the source buffer
	    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);

	    // Build the program
	    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	    // Create the compute kernel from the program
	    ko_vadd = clCreateKernel(program, "vadd", &err);

	    // Create the input (a, b) and output (c) arrays in device memory
	    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
	    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
	    d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);

	    // Write a and b vectors into compute device memory
	    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
	    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);


	    // Set the arguments to our compute kernel
	    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
	    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
	    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
	    err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);

	    // Execute the kernel over the entire range of our 1d input data set
	    // letting the OpenCL runtime choose the work-group size
	    global = count;
	    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);

	    err = clFinish(commands);

	    // Read back the results from the compute device
	    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL );  

	    // cleanup then shutdown
	    clReleaseMemObject(d_a);
	    clReleaseMemObject(d_b);
	    clReleaseMemObject(d_c);
	    clReleaseProgram(program);
	    clReleaseKernel(ko_vadd);
	    clReleaseCommandQueue(commands);
	    clReleaseContext(context);
	    free(h_a);
	    free(h_b);
	    free(h_c);
	
	}

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    printf("\nThe program ran in %lf seconds\n", rtime);

    return 0;
}
