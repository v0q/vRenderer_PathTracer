# vRenderer - Path Tracer (AGSDT Assignment)
Cuda and OpenCL implementations of a simple path tracer. OpenGL used for visualising the output.
Supports 3D models, diffuse, normal and specular maps, HDRI (EXR) environment maps, MERL 100 BRDFS.
https://github.com/NCCA/agsdt2016-v0q

## Installation instructions:

# Dependencies:
- Cuda or OpenCL
- OpenGL 4.1
- OpenEXR
- Assimp
- Qt 5.7

The file vRenderer.pri contains compile instructions for both Cuda and OpenCL versions of the code.
The one being used is controlled by config flags in the .pro-file. By default Cuda is used under Linux and OpenCL under MacOS.

To compile a specific implementation, use CONFIG += vrenderer_cl or vrenderer_cuda before including .pri-file at the end of the .pro-file.

The code has been ran and successfully compiled using the University machines and my personal Macbook Pro.

The easiest way to build the program is by opening the project file in QtCreator, but by using qmake the project can also be built from the command line.
To build the project from command line, navigate to the root of the project and follow these steps:

1. qmake
2. make -j4
3. ./vRenderer

This assumes that the used setup is similar to the one at the University.

# Instructions:

Keyboard:
- Return - Toggle between the colour and depth buffers (depth buffer not implemented for OpenCL)
- Escape - Exit the program

Mouse:
- Left click and move - Rotate the camera
- Right click and move - Move forward and backward

UI Elements:
- Scene tree is not currently in use
- To load a mesh, click on the Load Mesh button and navigate to a wanted 3D model (currently opens .obj .ply and .fbx files)
- Use radio buttons to swap between HDRI environment and Cornell box.
- Only HDRI maps that are in .exr format can be used at this time.
- Use diffuse, normal and specular buttons to load corresponding texture maps.
- Fresnel coef and Fresnel pow sliders control the fresnel reflections (estimation) and can be seen on the second small sphere or on a model with a specular map.
- Camera FOV slider controls the field of view of the camera
- Enabling FXAA will switch to FXAA shader that does anti-aliasing on the final image. The effect can be controlled to an extent using the sliders below.
- Enable example sphere tick box renders a bigger sphere instead of a triangle mesh (will hide the triangle mesh if one's loaded) and is especially suitable for viewing MERL 100 measured BRDFs.
- BRDFs can be used by ticking the "Use Merl BRDF"-checkbox and loading a BRDF binary (binaries can be downloaded from http://people.csail.mit.edu/wojciech/BRDFDatabase/brdfs/). BRDFs will show up on either normal loaded meshes or the example sphere. While the BRDFs are enabled, it will override diffuse and specular textures.

# Limitations/bugs
- The current implementation only supports a single mesh at a time.
- OpenCL version has a bug with calculating normals using the normal map. It is being estimated (wrongly) to provide some visual feedback.
- A model too simple might not load, as the current implementation expects the SBVH tree to contain at least two child nodes (e.g. at least 3 nodes)
- As the implementation is not doing Bidirectional Path Tracing, fireflies might occur especially when using highly reflective materials. Enabling FXAA helps to reduce overly visible fireflies.

## External assets
# 3D models
With this submission, I've included three different 3d models:
- models/adam_head.obj and models/adam_mask.obj - are models from the Unity3D's Tech demo "Adam" (Available online at https://unity3d.com/pages/adam) and are used only for educational purposes.
- models/dragon_vrip_res2.obj - is a 3D scan from the Stanford University Computer Graphics Laboratory and is available online at http://graphics.stanford.edu/data/3Dscanrep/
# Textures
All of the textures inside the textures folder are also from the Unity3D's Tech demo "Adam"
# HDRI maps
Provided with the submission, I've included two different HDRI maps:
- hdr/Arches_E_PineTree_3k.exr - is from HDR Labs (Christian Bloch), available at http://www.hdrlabs.com/sibl/archive.html
- hdr/cathedral02.exr - is from a "26 Free HDRis" package by Joost Vanhoutte, available at https://gumroad.com/l/hdris2

