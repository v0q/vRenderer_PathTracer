# This specifies the exe name
TARGET = vRenderer
# where to put the .o files
OBJECTS_DIR = obj
# core Qt Libs to use add more here if needed.
QT += gui opengl core

# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
  cache()
  DEFINES +=QT5BUILD
}
# where to put moc auto generated files
MOC_DIR=moc
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle
# Auto include all .cpp files in the project src directory (can specifiy individually if required)
SOURCES+= $$PWD/src/main.cpp \
          $$PWD/src/NGLScene.cpp \
          $$PWD/src/NGLSceneMouseControls.cpp
# same for the .h files
HEADERS+= $$PWD/include/NGLScene.h \
          $$PWD/include/WindowParams.h
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include
# where our exe is going to live (root of project)
DESTDIR=./
# add the glsl shader files
OTHER_FILES += README.md \
               shaders/screenQuad.vert \
               shaders/screenQuad.frag
# were are going to default to a console app
CONFIG += console
# note each command you add needs a ; as it will be run as a single line
# first check if we are shadow building or not easiest way is to check out against current
#!equals(PWD, $${OUT_PWD}){
#	copydata.commands = echo "creating destination dirs" ;
#	# now make a dir
#	copydata.commands += mkdir -p $$OUT_PWD/shaders ;
#	copydata.commands += echo "copying files" ;
#	# then copy the files
#	copydata.commands += $(COPY_DIR) $$PWD/shaders/* $$OUT_PWD/shaders/ ;
#	# now make sure the first target is built before copy
#	first.depends = $(first) copydata
#	export(first.depends)
#	export(copydata.commands)
#	# now add it as an extra target
#	QMAKE_EXTRA_TARGETS += first copydata
#}
NGLPATH=$$(NGLDIR)
isEmpty(NGLPATH){ # note brace must be here
	message("including $HOME/NGL")
	include($(HOME)/NGL/UseNGL.pri)
}
else{ # note brace must be here
	message("Using custom NGL location")
	include($(NGLDIR)/UseNGL.pri)
}

# ------------
# Cuda related
# ------------

# Project specific
CUDA_SOURCES += $$PWD/cuda/src/PathTracer.cu
                #$$PWD/cuda/src/TutorialPathTracer.cu
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
SMS = 52

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
