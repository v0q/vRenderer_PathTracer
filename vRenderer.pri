vrenderer_cl {
  message("Using OpenCL");
  DEFINES += __VRENDERER_OPENCL__
  macx: LIBS += -framework OpenCL
  linux: LIBS += -L/usr/lib64/nvidia -lOpenCL
  SOURCES += $$PWD/src/vRendererCL.cpp
  HEADERS += $$PWD/include/vRendererCL.h

  CL_SOURCES += $$PWD/cl/src/PathTracer.cl \
                $$PWD/cl/src/CL_UVRender.cl
  CL_HEADERS += $$PWD/cl/include/PathTracer.h

  INCLUDEPATH += $$PWD/cl/include
  OTHER_FILES += $$CL_SOURCES $$CL_HEADERS

  OTHER_FILES += $$PWD/src/vRendererCuda.cpp
  OTHER_FILES += $$PWD/include/vRendererCuda.h
}
vrenderer_cuda {
  message("Using Cuda");
  DEFINES += __VRENDERER_CUDA__
  SOURCES += $$PWD/src/vRendererCuda.cpp
  HEADERS += $$PWD/include/vRendererCuda.h

  OTHER_FILES += $$PWD/src/vRendererCL.cpp
  OTHER_FILES += $$PWD/include/vRendererCL.h
  # ------------
  # Cuda related
  # ------------

  # Project specific
  CUDA_SOURCES += $$PWD/cuda/src/PathTracer.cu
  CUDA_HEADERS += $$PWD/cuda/include/PathTracer.cuh \
                  $$PWD/cuda/include/MathHelpers.cuh
  CUDA_OBJECTS_DIR = $$PWD/cuda/obj

  INCLUDEPATH += $$PWD/cuda/include

  # Cuda specific
  CUDA_PATH = "/usr"
  NVCC_CXXFLAGS += -ccbin g++
  NVCC = $(CUDA_PATH)/bin/nvcc

  # Extra NVCC options
  NVCC_OPTIONS = --use_fast_math

  # System type
  OS_SIZE = 64
  # Compute capabilities that you want the project to be compiled for
  SMS = 50 52

  # Generate gencode flags from the cc list
  for(sm, SMS) {
    GENCODE_FLAGS += -gencode arch=compute_$$sm,code=sm_$$sm
  }

  # Specify the location to your cuda headers here
  INCLUDEPATH += /usr/include/cuda

  # Compiler instruction, add -I in front of each include path
  CUDA_INCLUDES = $$join(INCLUDEPATH, ' -I', '-I', '')

  QMAKE_LIBDIR += $$CUDA_PATH/lib/
  LIBS += -lcudart

  OTHER_FILES += $$CUDA_SOURCES $$CUDA_HEADERS


  # As cuda needs to be compiled by a separate compiler, we'll add instructions for qmake to use the separate
  # compiler to compile cuda files and finally compile the object files together
  cuda.input = CUDA_SOURCES
  cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
  cuda.commands = $$NVCC $$NVCC_CXXFLAGS -m$$OS_SIZE $$GENCODE_FLAGS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} $$NVCC_OPTIONS $$CUDA_INCLUDES
  cuda.dependency_type = TYPE_C

  # Add the generated compiler instructions to qmake
  QMAKE_EXTRA_COMPILERS += cuda
}
