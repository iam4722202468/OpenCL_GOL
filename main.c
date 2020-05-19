#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <curses.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>
 
#define PROGRAM_FILE "kernel.cl"
#define GRID_SIZE 24

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
  cl_platform_id platform;
  cl_device_id dev;
  int err;

  /* Identify a platform */
  err = clGetPlatformIDs(1, &platform, NULL);
  if(err < 0) {
    perror("Couldn't identify a platform");
    exit(1);
  } 

  /* Access a device */
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if(err == CL_DEVICE_NOT_FOUND) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  }

  if(err < 0) {
    perror("Couldn't access any devices");
    exit(1);   
  }

  return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  int err;

  /* Read program file and place content into buffer */
  program_handle = fopen(filename, "r");
  if(program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(1);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  /* Create program from file */
  program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
  if(err < 0) {
    perror("Couldn't create the program");
    exit(1);
  }
  free(program_buffer);

  /* Build program */
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err < 0) {
    /* Find size of log and print to std output */
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    program_log = (char*) malloc(log_size + 1);
    program_log[log_size] = '\0';

    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);

    printf("%s\n", program_log);

    free(program_log);
    exit(1);
  }

  return program;
}

void setVal(int *arr, int size, int x, int y, int val) {
  arr[size*y+x] = val;
}

int main(int argc, char* argv[])
{
  srand(time(NULL));
 
  // OpenCL init
  cl_device_id device_id;
  cl_context context;
  cl_command_queue queue;
  cl_program program;

  cl_kernel kernel;
  cl_int err;

  cl_mem grid;
  cl_mem gridSwp;

  // Valid factors of 24 to be used for concurrency
  int kernelCount = 2;
  int multiples[] = {24, 12, 8, 4, 2, 1};

  // Tick counter
  int tick = 0;
  int tickFinal = 1000;

  // Grid store
  // Store in 1d and expand to 2d
  int *initGrid;

  int output = true;

  int argPtr;
  if (argc > 1) {
    argPtr = 1;
    while (argPtr < argc) {
      if (strcmp(argv[argPtr], "-o") == 0) {
        output = false;
        argPtr += 1;
      } else if (strcmp(argv[argPtr], "-n") == 0) {
        sscanf(argv[argPtr + 1], "%d", &kernelCount);
        kernelCount -= 1;
        argPtr += 2;
      }
    }
  }

  if (kernelCount < 0 || kernelCount >= 6)
    printf("Kernel count must be on [1,6]");

  // Ncurses init
  WINDOW* ncursesWin;
  int ch;

  if (output) {
    if ((ncursesWin = initscr()) == NULL) {
      fprintf(stderr, "Error initialising ncurses.\n");
      exit(EXIT_FAILURE);
    }

    nodelay(ncursesWin, TRUE);
    noecho();
  }



  // Size in bytes of grid
  size_t gridSizeBytes = sizeof(int) * GRID_SIZE * GRID_SIZE;

  // Allocate memory for initial grid
  initGrid = (int*)malloc(gridSizeBytes);

  // Randomly set grid memory
  for (int j = 0; j < GRID_SIZE; ++j)
    for (int i = 0; i < GRID_SIZE; ++i)
      initGrid[i*GRID_SIZE+j] = rand()%5 == 0;

  setVal(initGrid, GRID_SIZE, 4,4, 1);
  setVal(initGrid, GRID_SIZE, 4,5, 1);
  setVal(initGrid, GRID_SIZE, 4,6, 1);
  setVal(initGrid, GRID_SIZE, 5,6, 1);
  setVal(initGrid, GRID_SIZE, 5,7, 1);

  device_id = create_device();

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    printf("Error: Failed to create a compute context\n");
    return EXIT_FAILURE;
  }

  // Create a command queue
  queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
  if (!queue) {
    printf("Error: Failed to create a command commands\n");
    return EXIT_FAILURE;
  }

  program = build_program(context, device_id, PROGRAM_FILE);

  // Create the GOL kernel
  kernel = clCreateKernel(program, "GOL", &err);
  if (!kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create GOL kernel \n");
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  grid = clCreateBuffer(context, CL_MEM_READ_WRITE, gridSizeBytes, NULL, NULL);
  gridSwp = clCreateBuffer(context, CL_MEM_READ_WRITE, gridSizeBytes, NULL, NULL);

  if (!grid || !gridSwp) {
    printf("Error: Failed to allocate device memory\n");
    return EXIT_FAILURE;
  }

  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(queue, grid, CL_TRUE, 0, gridSizeBytes, initGrid, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array\n");
    return EXIT_FAILURE;
  }

  // Set the arguments to GOL  kernel
  int gridSize = GRID_SIZE;
  err  = clSetKernelArg(kernel, 0, sizeof(int), &gridSize);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &grid);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &gridSwp);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments\n");
    return EXIT_FAILURE;
  }

  // Here we are setting up the worker groups. Worker groups will be run concurrently
  // Work groups will be assigned to free cores. This is like running multiple kernels
  size_t workerSizes[2] = {multiples[kernelCount], multiples[kernelCount]};
  size_t workGroupSizes[2] = {24, 24};

  // Main game loop
  while (tick <= tickFinal || output) {
      // Create two dimensional work item / work groups and use this for the xy coordinate to update
      // Queue up kernel to run with workers and worker group
      err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, workGroupSizes, workerSizes, 0, NULL, NULL);

      // On every tick switch boards
      if(tick%2 == 1) {
          err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &grid);
          err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &gridSwp);
      } else {
          err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gridSwp);
          err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &grid);
      }

      if (output) {
        // Wait for queue to finish before reading it
        clFinish(queue);

        // Read the results from kernel
        if (tick%2 == 1)
          err =  clEnqueueReadBuffer(queue, gridSwp, CL_TRUE, 0, gridSizeBytes, initGrid, 0, NULL, NULL );
        else
          err =  clEnqueueReadBuffer(queue, grid, CL_TRUE, 0, gridSizeBytes, initGrid, 0, NULL, NULL );

        if (err != CL_SUCCESS) {
          printf("Error: Failed to read output array\n");
          return EXIT_FAILURE;;
        }

        // Check if q was pressed
        ch = getch();
        if (ch == 'q')
          break;

        // Draw with ncurses
        for (int j = 0; j < GRID_SIZE; j++) {
          for (int i = 0; i < GRID_SIZE; i++) {
            if (initGrid[i * GRID_SIZE+j]) 
              mvaddstr(i, j, "+");
            else
              mvaddstr(i, j, ".");
          }
        }

        refresh();
        sleep(1);
      }

      tick++;
  }

  if (err != CL_SUCCESS) {
     printf("Error: Failed to launch kernels%d\n",err);
     return EXIT_FAILURE;
  }

  // Free grid memory
  free(initGrid);

  if (output) {
    // Free ncurses memory
    delwin(ncursesWin);
    endwin();
    refresh();
  }

  return 0;
}
