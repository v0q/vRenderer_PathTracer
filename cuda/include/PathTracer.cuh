#pragma once

void cu_runRenderKernel(cudaSurfaceObject_t _texture, float3 *_colorArr, float3 *_cam, float3 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
