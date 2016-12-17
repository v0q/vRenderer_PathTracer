struct Ray {
  float3 m_origin;
  float3 m_dir;
  Ray(float3 _o, float3 _d) : m_origin(_o), m_dir(_d) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
  float m_r;       // radius
  float3 m_pos;
  float3 m_emission;
  float3 m_col;
  Refl_t m_refl;

  float intersect(const Ray &_r) const
  { // returns distance, 0 if nohit
    float3 op = m_pos - _r.m_origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t;
    float eps = 1e-4;
    float b = dot(op, _r.m_dir);
    float det = b*b - dot(op, op) + m_r*m_r;
    if(det < 0)
      return 0;
    else
      det = sqrtf(det);
    return (t = b-det) > eps ? t : ((t = b+det) > eps ? t : 0.0);
  }
};

__constant Sphere spheres[] = {			//Scene: radius, position, emission, color, material
  { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.075f, 0.f, 0.f }, { 0.75f, 0.0f, 0.0f }, DIFF }, //Left
  { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.f, 0.075f, 0.f }, { 0.0f, 0.75f, 0.0f }, DIFF }, //Right
  { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back
  { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt
  { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm
  { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top
  { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
  { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
  { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

inline bool intersectScene(const Ray &_r, float &_t, int &_id)
{
  float n = sizeof(spheres)/sizeof(Sphere);
  float d;
  float inf = _t = 1e20;
  for(int i = int(n); i--;) {
    if((d = spheres[i].intersect(_r)) && d < _t)
    {
      _t = d;
      _id = i;
    }
  }
  return _t < inf;
}

static unsigned int hash(unsigned int *seed0, unsigned int *seed1) {
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

  return *seed0**seed1;
}

float3 radiance(Ray &_r, unsigned int *s0, unsigned int *s1)
{
  float3 color = make_float3(0.0, 0.0, 0.0);
  float3 mask = make_float3(1.0, 1.0, 1.0);

  for(unsigned int bounces = 0; bounces < 4; bounces++)
  {
    float t;																	// distance to intersection
    int id = 0;																// id of intersected object
    if(!intersectScene(_r, t, id)) {
      return make_float3(0.0, 0.0, 0.0);			// if miss, return black
    }

    unsigned int seed = hash(s0, s1);
    thrust::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> uniformDist(0, 1);

    const Sphere &obj = spheres[id];  // hitobject
    float3 x = _r.m_origin + _r.m_dir*t;						// hitpoint
    float3 n = normalize(x - obj.m_pos);						// normal, unsigned int *_s0, unsigned int *_s1
    float3 nl = dot(n, _r.m_dir) < 0 ? n : n * -1;	// front facing normal

    color += mask * obj.m_emission;
    if(obj.m_refl == DIFF) {
      float r1 = 2 * M_PI * uniformDist(rng);
      float r2 = uniformDist(rng);
      float r2s = sqrtf(r2);
      float3 w = nl;
      float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
      float3 v = cross(w, u);

      float3 d = normalize(( u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2)));

      _r.m_origin = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
      _r.m_dir = d;

      mask *= obj.m_col;
      mask *= dot(d, nl);
      mask *= 2;
    } else if(obj.m_refl == SPEC) {
      _r.m_origin = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
      _r.m_dir = _r.m_dir - n*2*dot(n, _r.m_dir);
    } else if(obj.m_refl == REFR) {
      bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
      float nc = 1.0f;  // Index of Refraction air
      float nt = 1.33f;  // Index of Refraction glass/water
      float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
      float ddn = dot(_r.m_dir, nl);
      float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

      if(cos2t < 0.0f) {
        _r.m_dir = _r.m_dir - 2.0f * n * dot(n, _r.m_dir);
        _r.m_origin = x + nl * 0.01f;
      }	else {
        // compute direction of transmission ray
        float3 tdir = normalize(_r.m_dir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

        float R0 = (nt - nc)*(nt - nc) / ((nt + nc)*(nt + nc));
        float c = 1.f - (into ? -ddn : dot(tdir, n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = .25f + .5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.f - P);

        // randomly choose reflection or transmission ray
        if(uniformDist(rng) < 0.25f) {
          mask *= RP;
          _r.m_dir = _r.m_dir - 2.0f * n * dot(n, _r.m_dir);
          _r.m_origin = x + nl * 0.02f;
        }	else {
          mask *= TP;
          _r.m_dir = tdir; //r = Ray(x, tdir);
          _r.m_origin = x + nl * 0.05f; // epsilon must be small to avoid artefacts
        }
      }
    }
  }
  return color;
}

__kernel void render(__global float3 *_colors, float3 *_cam, float3 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
  const int work_id = get_global_id(0);
  unsigned int x = work_id%_w;
  unsigned int y = work_id/_w;

  if(x < _w && y < _h) {
    unsigned int s1 = x * _frame;
    unsigned int s2 = y * _time;

    Ray camera(*_cam, *_dir);

    float3 cx = make_float3(_w * .5135 / _h, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, camera.m_dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)

    unsigned int samps = 8;
    for(unsigned int s = 0; s < samps; s++) {  // samples per pixel
      // compute primary ray direction
      float3 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
      // create primary ray, add incoming radiance to pixelcolor
      Ray newcam(camera.m_origin + d * 40, normalize(d));
      _colors[ind] += radiance(newcam, &s1, &s2);
    }

    float coef = 1.f/(samps*_frame);
    unsigned char r = (unsigned char)(powf(clamp(_colors[ind].x*coef, 0.0, 1.0), invGamma) * 255);
    unsigned char g = (unsigned char)(powf(clamp(_colors[ind].y*coef, 0.0, 1.0), invGamma) * 255);
    unsigned char b = (unsigned char)(powf(clamp(_colors[ind].z*coef, 0.0, 1.0), invGamma) * 255);

    uchar4 data = make_uchar4(r, g, b, 0xff);
    surf2Dwrite(data, _tex, x*sizeof(uchar4), y);
  }
}
