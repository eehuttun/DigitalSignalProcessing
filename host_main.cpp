#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>

#include <CL/cl.h>
#include <vector>

using namespace std;

/* OpenCL macros */

#define MAX_ERROR_VALUE 64
#define PREFERRED_PLATFORM "NVIDIA" // Change the platform you want to use here. I.e. Intel
#define PREFERRED_DEVICE CL_DEVICE_TYPE_CPU
#define CL_KERNEL_SOURCE_FILE "kernels/depth_estimator_simple.cl"

/* Simple function for rounding up global work sizes */

int roundUp2(int groupSize, int globalSize) {
	int r = globalSize % groupSize;

	if(r == 0)
		return globalSize;
	else
		return globalSize + groupSize - r;
}

/* A function to check for error code as per cl_int returned by OpenCl
Parameter errCheck Error value as cl_int
Parameter msg User provided error message
Return True if Error found, False otherwise */

int cl_errCheck(const cl_int errCheck, const char * msg, bool exitOnError)
{
	char *cl_error[MAX_ERROR_VALUE] = {
    "CL_SUCCESS",                         // 0
    "CL_DEVICE_NOT_FOUND",                //-1
    "CL_DEVICE_NOT_AVAILABLE",            //-2
    "CL_COMPILER_NOT_AVAILABLE",          //-3
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
    "CL_OUT_OF_RESOURCES",                //-5
    "CL_OUT_OF_HOST_MEMORY",              //-6
    "CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
    "CL_MEM_COPY_OVERLAP",                //-8
    "CL_IMAGE_FORMAT_MISMATCH",           //-9
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
    "CL_BUILD_PROGRAM_FAILURE",           //-11
    "CL_MAP_FAILURE",                     //-12
    "",                                   //-13
    "",                                   //-14
    "",                                   //-15
    "",                                   //-16
    "",                                   //-17
    "",                                   //-18
    "",                                   //-19
    "",                                   //-20
    "",                                   //-21
    "",                                   //-22
    "",                                   //-23
    "",                                   //-24
    "",                                   //-25
    "",                                   //-26
    "",                                   //-27
    "",                                   //-28
    "",                                   //-29
    "CL_INVALID_VALUE",                   //-30
    "CL_INVALID_DEVICE_TYPE",             //-31
    "CL_INVALID_PLATFORM",                //-32
    "CL_INVALID_DEVICE",                  //-33
    "CL_INVALID_CONTEXT",                 //-34
    "CL_INVALID_QUEUE_PROPERTIES",        //-35
    "CL_INVALID_COMMAND_QUEUE",           //-36
    "CL_INVALID_HOST_PTR",                //-37
    "CL_INVALID_MEM_OBJECT",              //-38
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
    "CL_INVALID_IMAGE_SIZE",              //-40
    "CL_INVALID_SAMPLER",                 //-41
    "CL_INVALID_BINARY",                  //-42
    "CL_INVALID_BUILD_OPTIONS",           //-43
    "CL_INVALID_PROGRAM",                 //-44
    "CL_INVALID_PROGRAM_EXECUTABLE",      //-45
    "CL_INVALID_KERNEL_NAME",             //-46
    "CL_INVALID_KERNEL_DEFINITION",       //-47
    "CL_INVALID_KERNEL",                  //-48
    "CL_INVALID_ARG_INDEX",               //-49
    "CL_INVALID_ARG_VALUE",               //-50
    "CL_INVALID_ARG_SIZE",                //-51
    "CL_INVALID_KERNEL_ARGS",             //-52
    "CL_INVALID_WORK_DIMENSION ",         //-53
    "CL_INVALID_WORK_GROUP_SIZE",         //-54
    "CL_INVALID_WORK_ITEM_SIZE",          //-55
    "CL_INVALID_GLOBAL_OFFSET",           //-56
    "CL_INVALID_EVENT_WAIT_LIST",         //-57
    "CL_INVALID_EVENT",                   //-58
    "CL_INVALID_OPERATION",               //-59
    "CL_INVALID_GL_OBJECT",               //-60
    "CL_INVALID_BUFFER_SIZE",             //-61
    "CL_INVALID_MIP_LEVEL",               //-62
    "CL_INVALID_GLOBAL_WORK_SIZE"};       //-63


    if(errCheck != CL_SUCCESS) {
        printf("OpenCL Error: %d %s %s\n", errCheck, (char *)(cl_error[-errCheck]), msg);

        if(exitOnError) {
            exit(-1);
        }
        return true;
    }
    return false;
}

/* Input and output image declaration */

char * defLeftInputImageName = "./images/view1.pgm";
char * defRightInputImageName = "./images/view5.pgm";
char * defOutputImageName = "./images/depth_map.pgm";

/* Portable graymap (PGM) functions */
#ifdef _WIN32 | _WIN64
	extern "C" {
	int ReadPGMHeader(char * filename, int * imwidth, int* imheight);
	int WritePGM(char *filename, unsigned char *source, unsigned int width, unsigned int height);
	int ReadPGMData(char *filename, unsigned char * target, int width, int height, int dataoffset);
	int AsciiToInt(unsigned char *asc);
	}
#else
	int ReadPGMHeader(char * filename, int * imwidth, int* imheight);
	int WritePGM(char *filename, unsigned char *source, unsigned int width, unsigned int height);
	int ReadPGMData(char *filename, unsigned char * target, int width, int height, int dataoffset);
	int AsciiToInt(char *asc);
#endif


/* OpenCL objects declaration */
struct OpenCLObjects
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel[2];
};
OpenCLObjects openCLObjects;

//load Kernel from stream
std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram (const std::string& source,
	cl_context context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	cl_errCheck(error,"Program object creation failed", true);

	return program;
}


/////////////////////////////////// Main function  //////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int imWidthL, imHeightL, imWidthR, imHeightR, imSize, width, height,size;
	int channels=1;

	int win_size = 15;              //window dimension for SSD block matching e.g. 15 creates 15x15 window
	int max_disparity = 90;
	int l_width = 32;               //local work group width
	int l_height = 16;              //local work group height

	int headerlengthL, headerlengthR;
	unsigned char *imgL = NULL, *imgR = NULL, *resultL = NULL;
	char *imNameL = NULL, *imNameR = NULL;

	cl_ulong time_start = 0, time_end = 0;
    double total_time = 0.0;

	if(argc < 3)
	{
		imNameL = defLeftInputImageName;
		imNameR = defRightInputImageName;
	}
	else
	{
		imNameL = argv[1];
		imNameR = argv[2];
	}

	/* Image information check*/
	headerlengthL = ReadPGMHeader(imNameL, &imWidthL, &imHeightL);
	headerlengthR = ReadPGMHeader(imNameR, &imWidthR, &imHeightR);

	if (imWidthL != imWidthR || imHeightL != imHeightR)
	{
		fprintf(stderr, "Images are not the same size!\n");
		exit(1);
	}

	width = imWidthL;
	height = imHeightL;
	imSize = width * height;

	imgL = (unsigned char*)malloc(sizeof(unsigned char)*imSize);
	imgR = (unsigned char*)malloc(sizeof(unsigned char)*imSize);
	resultL = (unsigned char*)malloc(sizeof(unsigned char)*imSize); // result depth map

	if (imgL == NULL || imgR == NULL)
	{
		fprintf(stderr, "Memory allocation error!\n");
		exit(1);
	}

	ReadPGMData(imNameL, imgL, width, height, headerlengthL);
	ReadPGMData(imNameR, imgR, width, height, headerlengthR);


///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
OPENCLinitialization (Platform, Device, program, etc.)
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////

	unsigned int idP=-1;
	cl_uint num_platforms = 0, numDevices = 0, i = 0;
	cl_int errCheck = 0;
	size_t platform_name_length = 0,workitem_size[3], workgroup_size, address_bits;

	/* Step 1 */
	/*	Query for all available OpenCL platforms and devices on the system.
		Select a platform that has the required PREFERRED_PLATFORM substring using strstr-function.
	*/

	// Get total number of the available platforms.
	errCheck = clGetPlatformIDs(0, 0, &num_platforms);
	cl_errCheck(errCheck,"Platform inquiry",true);
	printf("Number of available platforms: %u \n", num_platforms);

	// Get IDs for all platforms.
	vector<cl_platform_id> platforms(num_platforms);
	errCheck = clGetPlatformIDs(num_platforms, &platforms[0], 0);
	cl_errCheck(errCheck,"clGetPlatformIds",true);

	// Get the size of the platform name in bytes.
	errCheck = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
			);
	cl_errCheck(errCheck,"clGetPlatformInfo",true);

	for (i=0; i<num_platforms; i++)
	{

		// Get the actual name for the i-th platform.
		vector<char> platform_name(platform_name_length);
		errCheck = clGetPlatformInfo(
					platforms[i],
					CL_PLATFORM_NAME,
					platform_name_length,
					&platform_name[i],
					0
					);
		cl_errCheck(errCheck,"clGetPlatformIInfo Names",true);

		//Print out the platform id and name
		string platforName = &platform_name[i];
		printf("\n[%u] %s \n", i, (platforName).c_str());

		// Check if the platform is the preferred platform
		if(strstr(&platform_name[i], PREFERRED_PLATFORM))
		{
			openCLObjects.platform = platforms[i];
			idP = i;
		}

/////////////// This is where the exercise work begins //////////////////////////////////////////////////////////
		/* STEP 2 */
		/* All the platforms are now queried and selected. Next you need to query all the available devices for the platforms.
			Of course you can only query the devices for the selected platform and then select a suitable device.
			Depending on your approach place the device query inside or outside of the above for loop. The reason for scanning all
			the devices in each platform is just to show you what device options you might have.
		*/

	}

	if (idP == -1) {
		printf("Preferred platform not found. Exiting...");
		exit(1);
	}
	else
		printf("\nPlaform ID [%u] selected \n", idP);

    //find the devices and print how many devices found + device information
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platforms [idP], CL_DEVICE_TYPE_ALL, sizeof(openCLObjects.device),
        &openCLObjects.device, &deviceIdCount);
    cout << "Found " << deviceIdCount << " device(s)" << endl;
    char deviceName[1024];
    cl_ulong local_mem_size;
    clGetDeviceInfo(openCLObjects.device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(openCLObjects.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    cout << "Name: " << deviceName << endl;
    cout << "Local memory size: " << local_mem_size << endl;

	/* STEP 3 */
	/* You have now selected a platform and a device to use either a CPU or a GPU in our case. In this exercise we simply use one device at a
	time. Of course you are free to implement a multidevice setup if you want. Now you need to create context for the selected device.
	*/
    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties> (platforms [idP]),
        0, 0
    };

    //initialize variable for error
    cl_int error = CL_SUCCESS;

    //create context
    openCLObjects.context = clCreateContext (
        contextProperties, deviceIdCount,
        &openCLObjects.device, nullptr,
        nullptr, &error);

	/* STEP 4. */
	/* Query for the OpenCL device that was used for context creation using clGetContextInfo. This step is just to check
		that no errors occured during context creation step. Error handling is very important in OpenCL since the bug might be in the host or
		kernel code. Use the errCheck-function on every step to indicate the location of the possible bug.
	*/

    int max_work_group;

	//initialize a test variable that is only used for querying the context
	cl_device_id contextsDevice;
	cl_errCheck(clGetContextInfo(openCLObjects.context, CL_CONTEXT_DEVICES, sizeof(contextsDevice), &contextsDevice, NULL),
                "context creation failed",
                true);
	cout << "Context created successfully";
	clGetDeviceInfo(contextsDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    cout << " for: " << deviceName << endl;
    clGetDeviceInfo(contextsDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group), &max_work_group, NULL);
    cout << "max workgroup size in bytes: " << max_work_group << endl;

	/* STEP 5. */
	/*	Create OpenCL program from the kernel file source code
		First read the source kernel code in from the .cl file as an array of char's.
		Then "submit" the source code of the kernel to OpenCL and create a program object with it.
	*/

    //create kernel program object
    openCLObjects.program = CreateProgram (LoadKernel ("ssd_block_match.cl"),
        openCLObjects.context);
    cout << "Program object created" << endl;

	/* STEP 6. */
	/* Build the program. The program is now created but not built. Next you need to build it
	*/

    cl_errCheck(clBuildProgram (openCLObjects.program, deviceIdCount, &openCLObjects.device, nullptr, nullptr, nullptr),
                "Program building failed",
                true);
    cout << "Program built" << endl;

    /* STEP 7. */
	/*	Extract the kernel/kernels from the built program. The program consists of one or more kernels. Each kernel needs to be enqueued for
		execution from the host code. Creating a kernel via clCreateKernel is similar to obtaining an entry point of a specific function
		in an OpenCL program.
	*/

    openCLObjects.kernel[0] = clCreateKernel (openCLObjects.program, "SSD_depth_estimator", &error);
    cl_errCheck(error,"Kernel creation failed", true);
    cout << "Kernel created" << endl;

	/* STEP 8. */
	/*	Now that you have created the kernel/kernels, you also need to enqueue them for execution to the selected device. The command queu can
	be either in-order or out-of-order type depending on your approach. In-order command queue is a good starting point.
	*/

	openCLObjects.queue = clCreateCommandQueue (openCLObjects.context, openCLObjects.device,
    CL_QUEUE_PROFILING_ENABLE, &error);
    cl_errCheck(error,"Command queue creation failed",true);
    cout << "Created command queue" << endl;

	/* STEP 9. */
	/* Allocate device memory. You need to at least allocate memory on the device for the input and output images.
	Remember the error handling for the memory objects also.
	*/

    size_t mem_size = sizeof(unsigned char)*imSize;

    cl_mem leftBuffer = clCreateBuffer (openCLObjects.context, CL_MEM_READ_ONLY,
                                        mem_size,
                                        NULL, &error);
    cl_errCheck(error, "Memory buffer 1 creation failed", true);
    cout << "Created buffer for left input image" << endl;

    cl_mem rightBuffer = clCreateBuffer (openCLObjects.context, CL_MEM_READ_ONLY,
                                         mem_size,
                                         NULL , &error);
    cl_errCheck(error, "Memory buffer 2 creation failed", true);
    cout << "Created buffer for right input image" << endl;

    cl_mem leftRightResultBuffer = clCreateBuffer (openCLObjects.context, CL_MEM_READ_WRITE,
                                                   mem_size,
                                                   NULL , &error);
    cl_errCheck(error, "Memory buffer 3 creation failed", true);
    cout << "Created buffer for left-right result image" << endl;

    cl_mem leftRightCopyBuffer = clCreateBuffer (openCLObjects.context, CL_MEM_READ_WRITE,
                                                   mem_size,
                                                   NULL , &error);
    cl_errCheck(error, "Memory buffer 4 creation failed", true);
    cout << "Created buffer for second left-right result image" << endl;

    cl_mem rightLeftBuffer = clCreateBuffer (openCLObjects.context, CL_MEM_READ_WRITE,
                                                   mem_size,
                                                   NULL , &error);
    cl_errCheck(error, "Memory buffer 5 creation failed", true);
    cout << "Created buffer for right-left result image" << endl;

	/* STEP 10. */
	/* Enqueue the memory objects for writing to device using clEnqueueWriteBuffer/clEnqueueWriteImage.*/

    clEnqueueWriteBuffer(openCLObjects.queue,
                         leftBuffer,
                         CL_TRUE,
                         0,
                         sizeof(unsigned char)*imSize,
                         imgL,
                         NULL,
                         nullptr,
                         NULL);

    clEnqueueWriteBuffer(openCLObjects.queue,
                         rightBuffer,
                         CL_TRUE,
                         0,
                         sizeof(unsigned char)*imSize,
                         imgR,
                         NULL,
                         nullptr,
                         NULL);
    cout << "Buffers enqueued" << endl;

	/* STEP 11. */
	/*	Set the kernel arguments. Input images/buffers, output image/buffer, etc. Also set the global and local workgroup sizes if you have not
	done it yet. Remember that the global work group size needs to be a multiple of the local workgroup size.
	You can use the RoundUp2-function to make sure that this condition is met.
	*/

    //round up image width and height for work group sizes
    int pow2_width = roundUp2(l_width, width);
    int pow2_height = roundUp2(l_height, height);

	size_t globalWorkSize [2] = { pow2_width, pow2_height };
	size_t localWorkSize [2]  = { l_width, l_height };

	cout << "Global work size: " << globalWorkSize[0] << "x" << globalWorkSize[1] << endl;
	cout << "Local work size: " << localWorkSize[0] << "x" << localWorkSize[1] << endl;

	win_size = (win_size-1)/2;          //correct the window size to be the "radius" of the box

    cout << "Local memory needed: " << sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size)*2 << endl;

    i = 0;
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftBuffer);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &rightBuffer);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftRightResultBuffer);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftRightCopyBuffer);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &rightLeftBuffer);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size), NULL);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size), NULL);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &width);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &height);
	clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &win_size);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &max_disparity);


	/* STEP 12. */
	/*	Queue the kernel for execution.
		If you have more than one kernel, repeat step 11. and 12. for all of them.
	*/

    cl_event event[2];

	clEnqueueNDRangeKernel (openCLObjects.queue,
                            openCLObjects.kernel[0],
                            2,
                            nullptr,
                            globalWorkSize,
                            localWorkSize,
                            0,
                            nullptr,
                            &event[0]);
    cout << "Execution started" << endl;

    clFinish(openCLObjects.queue);

	/* STEP 13. */
	/* Read the output buffer back to the host using clEnqueueReadBuffer/clEnqueueReadImage */

	clEnqueueReadBuffer (openCLObjects.queue, leftRightResultBuffer, CL_TRUE, 0,
        sizeof(unsigned char)*imSize,
        resultL,
        0, nullptr, &event[1]);
    cout << "Result read back to host" << endl;

    clWaitForEvents(2, event);

    /*get the time taken to execute the kernel*/
    cl_errCheck(clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL),
                "start time failed", false);
    cl_errCheck(clGetEventProfilingInfo(event[1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL),
                "end time failed", false);
    total_time = (double)(time_end - time_start)/1000000000.0;

	/* Write the result image to a pgm file */
	WritePGM( defOutputImageName, resultL, width, height);

	/* STEP 14 */
	/* Perform "cleanup". Release all OpenCL objects i.e. kernels, program, memory */

    cout << endl << "Performing cleanup:" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseMemObject (leftRightResultBuffer);
    clReleaseMemObject (leftRightCopyBuffer);
    clReleaseMemObject (rightLeftBuffer);
    cout << "Result image buffers released" << endl;

    clReleaseMemObject (rightBuffer);
    cout << "Right input image buffer released" << endl;

    clReleaseMemObject (leftBuffer);
    cout << "Left input image buffer released" << endl;

    clReleaseCommandQueue (openCLObjects.queue);
    cout << "Command queue released" << endl;

    clReleaseContext (openCLObjects.context);
    cout << "Context released" << endl;

    clReleaseKernel (openCLObjects.kernel[0]);
    cout << "Kernel released" << endl;
    clReleaseProgram (openCLObjects.program);
    cout << "Program released" << endl;

    cout << "The OpenCL-implementation took " << total_time << " seconds to execute\n" << endl;


	return 0;
}



