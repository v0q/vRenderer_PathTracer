#pragma once

void cu_ModifyTexture(cudaSurfaceObject_t _texture, float3 *_colorArr, float3 *_cam, float3 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_fillFloat3(float3 *d_ptr, float3 _val, unsigned int _size);
