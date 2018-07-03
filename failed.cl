/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef PATH_TRACING_ESTIMATOR_UBERV2_CL
#define PATH_TRACING_ESTIMATOR_UBERV2_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef COMMON_CL
#define COMMON_CL

#define PI 3.14159265358979323846f
#define KERNEL __kernel
#define GLOBAL __global

#ifndef APPLE
#define INLINE __attribute__((always_inline))
#endif

#define HIT_MARKER 1
#define MISS_MARKER -1
#define INVALID_IDX -1

#define CRAZY_LOW_THROUGHPUT 0.0f
#define CRAZY_HIGH_RADIANCE 10.f
#define CRAZY_HIGH_DISTANCE 1000000.f
#define CRAZY_LOW_DISTANCE 0.001f
#define CRAZY_HIGH_DISTANCE_IN_VOLUME 1000000.f
#define REASONABLE_RADIANCE(x) (clamp((x), 0.f, CRAZY_HIGH_RADIANCE))
#define NON_BLACK(x) (length(x) > 0.f)

#define MULTISCATTER

#define RANDOM 1
#define SOBOL 2
#define CMJ 3

#define SAMPLER CMJ

#define CMJ_DIM 16

#define BDPT_MAX_SUBPATH_LEN 3

#ifdef BAIKAL_ATOMIC_RESOLVE
#define ADD_FLOAT3(x,y) atomic_add_float3((x),(y))
#define ADD_FLOAT4(x,y) atomic_add_float4((x),(y))
#else
#define ADD_FLOAT3(x,y) add_float3((x),(y))
#define ADD_FLOAT4(x,y) add_float4((x),(y))
#endif

#define VISIBILITY_MASK_PRIMARY (0x1)
#define VISIBILITY_MASK_SHADOW (0x1 << 15)
#define VISIBILITY_MASK_ALL (0xffffffffu)
#define VISIBILITY_MASK_NONE (0x0u)
#define VISIBILITY_MASK_BOUNCE(i) (VISIBILITY_MASK_PRIMARY << (i))
#define VISIBILITY_MASK_BOUNCE_SHADOW(i) (VISIBILITY_MASK_SHADOW << (i))

#endif // COMMON_CL
/**********************************************************************
 Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ********************************************************************/
#ifndef RAY_CL
#define RAY_CL


// Ray descriptor
typedef struct
{
    // xyz - origin, w - max range
    float4 o;
    // xyz - direction, w - time
    float4 d;
    // x - ray mask, y - activity flag
    int2 extra;
    // Padding
    float2 padding;
} ray;

// Set ray activity flag
INLINE void Ray_SetInactive(GLOBAL ray* r)
{
    r->extra.y = 0;
}

INLINE bool Ray_IsActive(GLOBAL ray* r)
{
    return r->extra.y != 0;
}

// Set extra data for ray
INLINE void Ray_SetExtra(GLOBAL ray* r, float2 extra)
{
    r->padding = extra;
}

// Set mask
INLINE void Ray_SetMask(GLOBAL ray* r, int mask)
{
    r->extra.x = mask;
}

INLINE int Ray_GetMask(GLOBAL ray* r)
{
    return r->extra.x;
}

// Get extra data for ray
INLINE float2 Ray_GetExtra(GLOBAL ray const* r)
{
    return r->padding;
}

// Initialize ray structure
INLINE void Ray_Init(GLOBAL ray* r, float3 o, float3 d, float maxt, float time, int mask)
{
    r->o.xyz = o;
    r->d.xyz = d;
    r->o.w = maxt;
    r->d.w = time;
    r->extra.x = mask;
    r->extra.y = 0xFFFFFFFF;
}

#endif
/**********************************************************************
 Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ********************************************************************/
#ifndef ISECT_CL
#define ISECT_CL

/// Intersection data returned by RadeonRays
typedef struct _Intersection
{
    // id of a shape
    int shapeid;
    // Primitive index
    int primid;
    // Padding elements
    int padding0;
    int padding1;
        
    // uv - hit barycentrics, w - ray distance
    float4 uvwt;
} Intersection;

float Intersection_GetDistance(__global Intersection const* isect)
{
    return isect->uvwt.w;
}

float2 Intersection_GetBarycentrics(__global Intersection const* isect)
{
    return isect->uvwt.xy;
}

#endif
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef UTILS_CL
#define UTILS_CL

#define PI 3.14159265358979323846f
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef PAYLOAD_CL
#define PAYLOAD_CL

#define TEXTURED_INPUT(x) union { struct { float4 value; } float_value; struct { int value[4]; } int_value; } x
#define TEXTURED_INPUT_HAS_TEXTURE(x) ((x).int_value.value[3] != -1)
#define TEXTURED_INPUT_GET_COLOR(x) ((x).float_value.value.xyz)

// Matrix
typedef struct
{
    float4 m0;
    float4 m1;
    float4 m2;
    float4 m3;
} matrix4x4;

// Camera
typedef struct
{
    // Coordinate frame
    float3 forward;
    float3 right;
    float3 up;
    // Position
    float3 p;

    // Image plane width & height in current units
    float2 dim;
    // Near and far Z
    float2 zcap;
    // Focal lenght
    float focal_length;
    // Camera aspect_ratio ratio
    float aspect_ratio;
    float focus_distance;
    float aperture;
} Camera;

enum UberMaterialLayers
{
    kEmissionLayer = 0x1,
    kTransparencyLayer = 0x2,
    kCoatingLayer = 0x4,
    kReflectionLayer = 0x8,
    kDiffuseLayer = 0x10,
    kRefractionLayer = 0x20,
    kSSSLayer = 0x40,
    kShadingNormalLayer = 0x80
};

typedef struct
{
    int offset;
    int layers;
    int flags;
    int padding;
} Material;

// Shape description
typedef struct
{
    // Shape starting index
    int startidx;
    // Start vertex
    int startvtx;
    // Number of primitives in the shape
    int volume_idx;
    // unique shape id
    int id;
    // Linear motion vector
    float3 linearvelocity;
    // Angular velocity
    float4 angularvelocity;
    // Transform in row major format
    matrix4x4 transform;
    Material material;
} Shape;

typedef struct
{
    int group_id;
    int padding[3];
} ShapeAdditionalData;

typedef enum
{
    kFloat3 = 0,
    kFloat = 1,
    kInt = 2
} InputMapDataType;

// Input data for input maps
typedef struct _InputMapData
{
    union
    {
        struct
        {
            float3 value;
        } float_value;
        struct
        {
            int idx;
            int placeholder[2];
            int type; //We can use it since float3 is actually float4
        } int_values;
    };
} InputMapData;

enum Bxdf
{
    kZero,
    kUberV2
};

enum LightType
{
    kPoint = 0x1,
    kDirectional,
    kSpot,
    kArea,
    kIbl
};

typedef struct
{
    union
    {
        // Area light
        struct
        {
            int id;
            int shapeidx;
            int primidx;
            int padding0;
        };

        // IBL
        struct
        {
            int tex;
            int tex_reflection;
            int tex_refraction;
            int tex_transparency;
        };

        // Spot
        struct
        {
            float ia;
            float oa;
            float f;
            int padding1;
        };
    };

    float3 p;
    float3 d;
    float3 intensity;
    int type;
    float multiplier;
    int tex_background;
    bool ibl_mirror_x;
} Light;

typedef enum
    {
        kEmpty,
        kHomogeneous,
        kHeterogeneous
    } VolumeType;


typedef struct _Volume
{
    VolumeType type;
    float g;

    // Id of volume data if present
    int data;
    int extra;

    // Absorbtion
    TEXTURED_INPUT(sigma_a);
    // Scattering
    TEXTURED_INPUT(sigma_s);
    // Emission
    TEXTURED_INPUT(sigma_e);
} Volume;

/// Supported formats
enum TextureFormat
{
    UNKNOWN,
    RGBA8,
    RGBA16,
    RGBA32
};

/// Texture description
typedef
struct _Texture
{
    // Width, height and depth
    int w;
    int h;
    int d;
    // Offset in texture data array
    int dataoffset;
    // Format
    int fmt;
    int extra;
} Texture;

// Hit data
typedef struct _DifferentialGeometry
{
    // World space position
    float3 p;
    // Shading normal
    float3 n;
    // Geo normal
    float3 ng;
    // UVs
    float2 uv;
    // Derivatives
    float3 dpdu;
    float3 dpdv;

    matrix4x4 world_to_tangent;
    matrix4x4 tangent_to_world;

    // Material
    Material mat;
    float  area;
    int transfer_mode;
    int padding[2];
} DifferentialGeometry;










#endif // PAYLOAD_CL


#ifndef APPLE
/// These functions are defined on OSX already
float4 make_float4(float x, float y, float z, float w)
{
    float4 res;
    res.x = x;
    res.y = y;
    res.z = z;
    res.w = w;
    return res;
}

float3 make_float3(float x, float y, float z)
{
    float3 res;
    res.x = x;
    res.y = y;
    res.z = z;
    return res;
}

float2 make_float2(float x, float y)
{
    float2 res;
    res.x = x;
    res.y = y;
    return res;
}

int2 make_int2(int x, int y)
{
    int2 res;
    res.x = x;
    res.y = y;
    return res;
}
#endif

matrix4x4 matrix_from_cols(float4 c0, float4 c1, float4 c2, float4 c3)
{
    matrix4x4 m;
    m.m0 = make_float4(c0.x, c1.x, c2.x, c3.x);
    m.m1 = make_float4(c0.y, c1.y, c2.y, c3.y);
    m.m2 = make_float4(c0.z, c1.z, c2.z, c3.z);
    m.m3 = make_float4(c0.w, c1.w, c2.w, c3.w);
    return m;
}

matrix4x4 matrix_from_rows(float4 c0, float4 c1, float4 c2, float4 c3)
{
    matrix4x4 m;
    m.m0 = c0;
    m.m1 = c1;
    m.m2 = c2;
    m.m3 = c3;
    return m;
}

matrix4x4 matrix_from_rows3(float3 c0, float3 c1, float3 c2)
{
    matrix4x4 m;
    m.m0.xyz = c0; m.m0.w = 0;
    m.m1.xyz = c1; m.m1.w = 0;
    m.m2.xyz = c2; m.m2.w = 0;
    m.m3 = make_float4(0.f, 0.f, 0.f, 1.f);
    return m;
}

matrix4x4 matrix_from_cols3(float3 c0, float3 c1, float3 c2)
{
    matrix4x4 m;
    m.m0 = make_float4(c0.x, c1.x, c2.x, 0.f);
    m.m1 = make_float4(c0.y, c1.y, c2.y, 0.f);
    m.m2 = make_float4(c0.z, c1.z, c2.z, 0.f);
    m.m3 = make_float4(0.f, 0.f, 0.f, 1.f);
    return m;
}

matrix4x4 matrix_transpose(matrix4x4 m)
{
    return matrix_from_cols(m.m0, m.m1, m.m2, m.m3);
}

float4 matrix_mul_vector4(matrix4x4 m, float4 v)
{
    float4 res;
    res.x = dot(m.m0, v);
    res.y = dot(m.m1, v);
    res.z = dot(m.m2, v);
    res.w = dot(m.m3, v);
    return res;
}

float3 matrix_mul_vector3(matrix4x4 m, float3 v)
{
    float3 res;
    res.x = dot(m.m0.xyz, v);
    res.y = dot(m.m1.xyz, v);
    res.z = dot(m.m2.xyz, v);
    return res;
}

float3 matrix_mul_point3(matrix4x4 m, float3 v)
{
    float3 res;
    res.x = dot(m.m0.xyz, v) + m.m0.w;
    res.y = dot(m.m1.xyz, v) + m.m1.w;
    res.z = dot(m.m2.xyz, v) + m.m2.w;
    return res;
}

/// Linearly interpolate between two values
float4 lerp(float4 a, float4 b, float w)
{
    return a + w*(b-a);
}

/// Linearly interpolate between two values
float3 lerp3(float3 a, float3 b, float w)
{
	return a + w*(b - a);
}

/// Translate cartesian coordinates to spherical system
void CartesianToSpherical ( float3 cart, float* r, float* phi, float* theta )
{
    float temp = atan2(cart.x, cart.z);
    *r = sqrt(cart.x*cart.x + cart.y*cart.y + cart.z*cart.z);
    // Account for discontinuity
    *phi = (float)((temp >= 0)?temp:(temp + 2*PI));
    *theta = acos(cart.y/ *r);
}

/// Get vector orthogonal to a given one
float3 GetOrthoVector(float3 n)
{
    float3 p;

    if (fabs(n.z) > 0.f) {
        float k = sqrt(n.y*n.y + n.z*n.z);
        p.x = 0; p.y = -n.z/k; p.z = n.y/k;
    }
    else {
        float k = sqrt(n.x*n.x + n.y*n.y);
        p.x = n.y/k; p.y = -n.x/k; p.z = 0;
    }

    return normalize(p);
}

float luminance(float3 v)
{
    // Luminance
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

uint upper_power_of_two(uint v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

INLINE
void atomic_add_float(volatile __global float* addr, float value)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + value;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
            expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

void atomic_add_float3(volatile __global float3* ptr, float3 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomic_add_float(p, value.x);
    atomic_add_float(p + 1, value.y);
    atomic_add_float(p + 2, value.z);
}

void atomic_add_float4(volatile __global float4* ptr, float4 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomic_add_float(p, value.x);
    atomic_add_float(p + 1, value.y);
    atomic_add_float(p + 2, value.z);
    atomic_add_float(p + 3, value.w);
}

void add_float3(__global float3* ptr, float3 value)
{
    *ptr += value;
}

void add_float4(__global float4* ptr, float4 value)
{
    *ptr += value;
}


#endif // UTILS_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef TEXTURE_CL
#define TEXTURE_CL




/// To simplify a bit
#define TEXTURE_ARG_LIST __global Texture const* textures, __global char const* texturedata
#define TEXTURE_ARG_LIST_IDX(x) int x, __global Texture const* textures, __global char const* texturedata
#define TEXTURE_ARGS textures, texturedata
#define TEXTURE_ARGS_IDX(x) x, textures, texturedata

/// Sample 2D texture
inline
float4 Texture_Sample2D(float2 uv, TEXTURE_ARG_LIST_IDX(texidx))
{
    // Get width and height
    int width = textures[texidx].w;
    int height = textures[texidx].h;

    // Find the origin of the data in the pool
    __global char const* mydata = texturedata + textures[texidx].dataoffset;

    // Handle UV wrap
    // TODO: need UV mode support
    uv -= floor(uv);

    // Reverse Y:
    // it is needed as textures are loaded with Y axis going top to down
    // and our axis goes from down to top
    uv.y = 1.f - uv.y;

    // Calculate integer coordinates
    int x0 = clamp((int)floor(uv.x * width), 0, width - 1);
    int y0 = clamp((int)floor(uv.y * height), 0, height - 1);

    // Calculate samples for linear filtering
    int x1 = clamp(x0 + 1, 0,  width - 1);
    int y1 = clamp(y0 + 1, 0, height - 1);

    // Calculate weights for linear filtering
    float wx = uv.x * width - floor(uv.x * width);
    float wy = uv.y * height - floor(uv.y * height);

    switch (textures[texidx].fmt)
    {
        case RGBA32:
        {
            __global float4 const* mydataf = (__global float4 const*)mydata;

            // Get 4 values for linear filtering
            float4 val00 = *(mydataf + width * y0 + x0);
            float4 val01 = *(mydataf + width * y0 + x1);
            float4 val10 = *(mydataf + width * y1 + x0);
            float4 val11 = *(mydataf + width * y1 + x1);

            // Filter and return the result
            return lerp(lerp(val00, val01, wx), lerp(val10, val11, wx), wy);
        }

        case RGBA16:
        {
            __global half const* mydatah = (__global half const*)mydata;

            // Get 4 values
            float4 val00 = vload_half4(width * y0 + x0, mydatah);
            float4 val01 = vload_half4(width * y0 + x1, mydatah);
            float4 val10 = vload_half4(width * y1 + x0, mydatah);
            float4 val11 = vload_half4(width * y1 + x1, mydatah);

            // Filter and return the result
            return lerp(lerp(val00, val01, wx), lerp(val10, val11, wx), wy);
        }

        case RGBA8:
        {
            __global uchar4 const* mydatac = (__global uchar4 const*)mydata;

            // Get 4 values and convert to float
            uchar4 valu00 = *(mydatac + width * y0 + x0);
            uchar4 valu01 = *(mydatac + width * y0 + x1);
            uchar4 valu10 = *(mydatac + width * y1 + x0);
            uchar4 valu11 = *(mydatac + width * y1 + x1);

            float4 val00 = make_float4((float)valu00.x / 255.f, (float)valu00.y / 255.f, (float)valu00.z / 255.f, (float)valu00.w / 255.f);
            float4 val01 = make_float4((float)valu01.x / 255.f, (float)valu01.y / 255.f, (float)valu01.z / 255.f, (float)valu01.w / 255.f);
            float4 val10 = make_float4((float)valu10.x / 255.f, (float)valu10.y / 255.f, (float)valu10.z / 255.f, (float)valu10.w / 255.f);
            float4 val11 = make_float4((float)valu11.x / 255.f, (float)valu11.y / 255.f, (float)valu11.z / 255.f, (float)valu11.w / 255.f);

            // Filter and return the result
            return lerp(lerp(val00, val01, wx), lerp(val10, val11, wx), wy);
        }

        default:
        {
            return make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

/// Sample lattitue-longitude environment map using 3d vector
inline
float3 Texture_SampleEnvMap(float3 d, TEXTURE_ARG_LIST_IDX(texidx), bool mirror_x)
{
    // Transform to spherical coords
    float r, phi, theta;
    CartesianToSpherical(d, &r, &phi, &theta);

    // Map to [0,1]x[0,1] range and reverse Y axis
    float2 uv;
    uv.x = (mirror_x) ? (1.f - phi / (2 * PI)) : phi / (2 * PI);
    uv.y = 1.f - theta / PI;

    // Sample the texture
    return Texture_Sample2D(uv, TEXTURE_ARGS_IDX(texidx)).xyz;
}

/// Get data from parameter value or texture
inline
float3 Texture_GetValue3f(
                // Value
                float3 v,
                // Texture coordinate
                float2 uv,
                // Texture args
                TEXTURE_ARG_LIST_IDX(texidx)
                )
{
    // If texture present sample from texture
    if (texidx != -1)
    {
        // Sample texture
        return native_powr(Texture_Sample2D(uv, TEXTURE_ARGS_IDX(texidx)).xyz, 2.2f);
    }

    // Return fixed color otherwise
    return v;
}

/// Get data from parameter value or texture
inline
float4 Texture_GetValue4f(
                // Value
                float4 v,
                // Texture coordinate
                float2 uv,
                // Texture args
                TEXTURE_ARG_LIST_IDX(texidx)
                )
{
    // If texture present sample from texture
    if (texidx != -1)
    {
        // Sample texture
        return native_powr(Texture_Sample2D(uv, TEXTURE_ARGS_IDX(texidx)), 2.2f);
    }

    // Return fixed color otherwise
    return v;
}

/// Get data from parameter value or texture
inline
float Texture_GetValue1f(
                        // Value
                        float v,
                        // Texture coordinate
                        float2 uv,
                        // Texture args
                        TEXTURE_ARG_LIST_IDX(texidx)
                        )
{
    // If texture present sample from texture
    if (texidx != -1)
    {
        // Sample texture
        return Texture_Sample2D(uv, TEXTURE_ARGS_IDX(texidx)).x;
    }

    // Return fixed color otherwise
    return v;
}

inline float3 TextureData_SampleNormalFromBump_uchar4(__global uchar4 const* mydatac, int width, int height, int t0, int s0)
{
	int t0minus = clamp(t0 - 1, 0, height - 1);
	int t0plus = clamp(t0 + 1, 0, height - 1);
	int s0minus = clamp(s0 - 1, 0, width - 1);
	int s0plus = clamp(s0 + 1, 0, width - 1);

	const uchar utex00 = (*(mydatac + width * t0minus + s0minus)).x;
	const uchar utex10 = (*(mydatac + width * t0minus + (s0))).x;
	const uchar utex20 = (*(mydatac + width * t0minus + s0plus)).x;

	const uchar utex01 = (*(mydatac + width * (t0)+s0minus)).x;
	const uchar utex21 = (*(mydatac + width * (t0)+(s0 + 1))).x;

	const uchar utex02 = (*(mydatac + width * t0plus + s0minus)).x;
	const uchar utex12 = (*(mydatac + width * t0plus + (s0))).x;
	const uchar utex22 = (*(mydatac + width * t0plus + s0plus)).x;

	const float tex00 = (float)utex00 / 255.f;
	const float tex10 = (float)utex10 / 255.f;
	const float tex20 = (float)utex20 / 255.f;

	const float tex01 = (float)utex01 / 255.f;
	const float tex21 = (float)utex21 / 255.f;

	const float tex02 = (float)utex02 / 255.f;
	const float tex12 = (float)utex12 / 255.f;
	const float tex22 = (float)utex22 / 255.f;

	const float Gx = tex00 - tex20 + 2.0f * tex01 - 2.0f * tex21 + tex02 - tex22;
	const float Gy = tex00 + 2.0f * tex10 + tex20 - tex02 - 2.0f * tex12 - tex22;
	const float3 n = make_float3(Gx, Gy, 1.f);

	return n;
}

inline float3 TextureData_SampleNormalFromBump_half4(__global half const* mydatah, int width, int height, int t0, int s0)
{
	int t0minus = clamp(t0 - 1, 0, height - 1);
	int t0plus = clamp(t0 + 1, 0, height - 1);
	int s0minus = clamp(s0 - 1, 0, width - 1);
	int s0plus = clamp(s0 + 1, 0, width - 1);

	const float tex00 = vload_half4(width * t0minus + s0minus, mydatah).x;
	const float tex10 = vload_half4(width * t0minus + (s0), mydatah).x;
	const float tex20 = vload_half4(width * t0minus + s0plus, mydatah).x;

	const float tex01 = vload_half4(width * (t0)+s0minus, mydatah).x;
	const float tex21 = vload_half4(width * (t0)+s0plus, mydatah).x;

	const float tex02 = vload_half4(width * t0plus + s0minus, mydatah).x;
	const float tex12 = vload_half4(width * t0plus + (s0), mydatah).x;
	const float tex22 = vload_half4(width * t0plus + s0plus, mydatah).x;

	const float Gx = tex00 - tex20 + 2.0f * tex01 - 2.0f * tex21 + tex02 - tex22;
	const float Gy = tex00 + 2.0f * tex10 + tex20 - tex02 - 2.0f * tex12 - tex22;
	const float3 n = make_float3(Gx, Gy, 1.f);

	return n;
}

inline float3 TextureData_SampleNormalFromBump_float4(__global float4 const* mydataf, int width, int height, int t0, int s0)
{
	int t0minus = clamp(t0 - 1, 0, height - 1);
	int t0plus = clamp(t0 + 1, 0, height - 1);
	int s0minus = clamp(s0 - 1, 0, width - 1);
	int s0plus = clamp(s0 + 1, 0, width - 1);

	const float tex00 = (*(mydataf + width * t0minus + s0minus)).x;
	const float tex10 = (*(mydataf + width * t0minus + (s0))).x;
	const float tex20 = (*(mydataf + width * t0minus + s0plus)).x;

	const float tex01 = (*(mydataf + width * (t0)+s0minus)).x;
	const float tex21 = (*(mydataf + width * (t0)+s0plus)).x;

	const float tex02 = (*(mydataf + width * t0plus + s0minus)).x;
	const float tex12 = (*(mydataf + width * t0plus + (s0))).x;
	const float tex22 = (*(mydataf + width * t0plus + s0plus)).x;

	const float Gx = tex00 - tex20 + 2.0f * tex01 - 2.0f * tex21 + tex02 - tex22;
	const float Gy = tex00 + 2.0f * tex10 + tex20 - tex02 - 2.0f * tex12 - tex22;
	const float3 n = make_float3(Gx, Gy, 1.f);

	return n;
}

/// Sample 2D texture
inline
float3 Texture_SampleBump(float2 uv, TEXTURE_ARG_LIST_IDX(texidx))
{
    // Get width and height
    int width = textures[texidx].w;
    int height = textures[texidx].h;

    // Find the origin of the data in the pool
    __global char const* mydata = texturedata + textures[texidx].dataoffset;

    // Handle UV wrap
    // TODO: need UV mode support
    uv -= floor(uv);

    // Reverse Y:
    // it is needed as textures are loaded with Y axis going top to down
    // and our axis goes from down to top
    uv.y = 1.f - uv.y;

    // Calculate integer coordinates
    int s0 = clamp((int)floor(uv.x * width), 0, width - 1);
    int t0 = clamp((int)floor(uv.y * height), 0, height - 1);

	int s1 = clamp(s0 + 1, 0, width - 1);
	int t1 = clamp(t0 + 1, 0, height - 1);

	// Calculate weights for linear filtering
	float wx = uv.x * width - floor(uv.x * width);
	float wy = uv.y * height - floor(uv.y * height);

    switch (textures[texidx].fmt)
    {
    case RGBA32:
    {
        __global float3 const* mydataf = (__global float3 const*)mydata;

		float3 n00 = TextureData_SampleNormalFromBump_float4(mydataf, width, height, t0, s0);
		float3 n01 = TextureData_SampleNormalFromBump_float4(mydataf, width, height, t0, s1);
		float3 n10 = TextureData_SampleNormalFromBump_float4(mydataf, width, height, t1, s0);
		float3 n11 = TextureData_SampleNormalFromBump_float4(mydataf, width, height, t1, s1);

		float3 n = lerp3(lerp3(n00, n01, wx), lerp3(n10, n11, wx), wy);

		return 0.5f * normalize(n) + make_float3(0.5f, 0.5f, 0.5f);
    }

    case RGBA16:
    {
        __global half const* mydatah = (__global half const*)mydata;

		float3 n00 = TextureData_SampleNormalFromBump_half4(mydatah, width, height, t0, s0);
		float3 n01 = TextureData_SampleNormalFromBump_half4(mydatah, width, height, t0, s1);
		float3 n10 = TextureData_SampleNormalFromBump_half4(mydatah, width, height, t1, s0);
		float3 n11 = TextureData_SampleNormalFromBump_half4(mydatah, width, height, t1, s1);

		float3 n = lerp3(lerp3(n00, n01, wx), lerp3(n10, n11, wx), wy);

		return 0.5f * normalize(n) + make_float3(0.5f, 0.5f, 0.5f);
    }

    case RGBA8:
    {
        __global uchar4 const* mydatac = (__global uchar4 const*)mydata;

		float3 n00 = TextureData_SampleNormalFromBump_uchar4(mydatac, width, height, t0, s0);
		float3 n01 = TextureData_SampleNormalFromBump_uchar4(mydatac, width, height, t0, s1);
		float3 n10 = TextureData_SampleNormalFromBump_uchar4(mydatac, width, height, t1, s0);
		float3 n11 = TextureData_SampleNormalFromBump_uchar4(mydatac, width, height, t1, s1);

		float3 n = lerp3(lerp3(n00, n01, wx), lerp3(n10, n11, wx), wy);

		return 0.5f * normalize(n) + make_float3(0.5f, 0.5f, 0.5f);
    }

    default:
    {
        return make_float3(0.f, 0.f, 0.f);
    }
    }
}



#endif // TEXTURE_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef SAMPLING_CL
#define SAMPLING_CL


#define SAMPLE_DIMS_PER_BOUNCE 300
#define SAMPLE_DIM_CAMERA_OFFSET 1
#define SAMPLE_DIM_SURFACE_OFFSET 5
#define SAMPLE_DIM_VOLUME_APPLY_OFFSET 101
#define SAMPLE_DIM_VOLUME_EVALUATE_OFFSET 201
#define SAMPLE_DIM_IMG_PLANE_EVALUATE_OFFSET 401

typedef struct
{
    uint seq;
    uint s0;
    uint s1;
    uint s2;
} SobolSampler;

typedef struct _Sampler
{
    uint index;
    uint dimension;
    uint scramble;
    uint padding;
} Sampler;

#if SAMPLER == SOBOL
#define SAMPLER_ARG_LIST __global uint const* sobol_mat
#define SAMPLER_ARGS sobol_mat
#elif SAMPLER == RANDOM
#define SAMPLER_ARG_LIST int unused
#define SAMPLER_ARGS 0
#elif SAMPLER == CMJ
#define SAMPLER_ARG_LIST int unused
#define SAMPLER_ARGS 0
#endif

/**
    Sobol sampler
**/
#define MATSIZE 52

// The code is taken from: http://gruenschloss.org/sobol/kuo-2d-proj-single-precision.zip
// 
float SobolSampler_Sample1D(Sampler* sampler, __global uint const* mat)
{
    uint result = sampler->scramble;
    uint index = sampler->index;
    for (uint i = sampler->dimension * MATSIZE; index;  index >>= 1, ++i)
    {
        if (index & 1)
            result ^= mat[i];
    }

    return result * (1.f / (1UL << 32));
}

/**
    Random sampler
**/

/// Hash function
uint WangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

/// Return random unsigned
uint UniformSampler_SampleUint(Sampler* sampler)
{
    sampler->index = WangHash(1664525U * sampler->index + 1013904223U);
    return sampler->index;
}

/// Return random float
float UniformSampler_Sample1D(Sampler* sampler)
{
    return ((float)UniformSampler_SampleUint(sampler)) / 0xffffffffU;
}


/**
    Correllated multi-jittered 
**/

uint permute(uint i, uint l, uint p)
{
    unsigned w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;

    do
    {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

float randfloat(uint i, uint p)
{
    i ^= p;
    i ^= i >> 17;
    i ^= i >> 10;
    i *= 0xb36534e5;
    i ^= i >> 12;
    i ^= i >> 21;
    i *= 0x93fc4795;
    i ^= 0xdf6e307f;
    i ^= i >> 17;
    i *= 1 | p >> 18;
    return i * (1.0f / 4294967808.0f);
}

float2 cmj(int s, int n, int p)
{
    int sx = permute(s % n, n, p * 0xa511e9b3);
    int sy = permute(s / n, n, p * 0x63d83595);
    float jx = randfloat(s, p * 0xa399d265);
    float jy = randfloat(s, p * 0x711ad6a5);

    return make_float2((s % n + (sy + jx) / n) / n,
        (s / n + (sx + jy) / n) / n);
}

float2 CmjSampler_Sample2D(Sampler* sampler)
{
    int idx = permute(sampler->index, CMJ_DIM * CMJ_DIM, 0xa399d265 * sampler->dimension * sampler->scramble);
    return cmj(idx, CMJ_DIM, sampler->dimension * sampler->scramble);
}

#if SAMPLER == SOBOL
void Sampler_Init(Sampler* sampler, uint index, uint start_dimension, uint scramble)
{
    sampler->index = index;
    sampler->scramble = scramble;
    sampler->dimension = start_dimension;
}
#elif SAMPLER == RANDOM
void Sampler_Init(Sampler* sampler, uint seed)
{
    sampler->index = seed;
    sampler->scramble = 0;
    sampler->dimension = 0;
}
#elif SAMPLER == CMJ
void Sampler_Init(Sampler* sampler, uint index, uint dimension, uint scramble)
{
    sampler->index = index;
    sampler->scramble = scramble;
    sampler->dimension = dimension;
}
#endif


float2 Sampler_Sample2D(Sampler* sampler, SAMPLER_ARG_LIST)
{
#if SAMPLER == SOBOL
    float2 sample;
    sample.x = SobolSampler_Sample1D(sampler, SAMPLER_ARGS);
    ++(sampler->dimension);
    sample.y = SobolSampler_Sample1D(sampler, SAMPLER_ARGS);
    ++(sampler->dimension);
    return sample;
#elif SAMPLER == RANDOM
    float2 sample;
    sample.x = UniformSampler_Sample1D(sampler);
    sample.y = UniformSampler_Sample1D(sampler);
    return sample;
#elif SAMPLER == CMJ
    float2 sample;
    sample = CmjSampler_Sample2D(sampler);
    ++(sampler->dimension);
    return sample;
#endif
}

float Sampler_Sample1D(Sampler* sampler, SAMPLER_ARG_LIST)
{
#if SAMPLER == SOBOL
    float sample = SobolSampler_Sample1D(sampler, SAMPLER_ARGS);
    ++(sampler->dimension);
    return sample;
#elif SAMPLER == RANDOM
    return UniformSampler_Sample1D(sampler);
#elif SAMPLER == CMJ
    float2 sample;
    sample = CmjSampler_Sample2D(sampler);
    ++(sampler->dimension);
    return sample.x;
#endif
}

/// Sample hemisphere with cos weight
float3 Sample_MapToHemisphere(
                        // Sample
                        float2 sample,
                        // Hemisphere normal
                        float3 n,
                        // Cos power
                        float e
                        )
{
    // Construct basis
    float3 u = GetOrthoVector(n);
    float3 v = cross(u, n);
    u = cross(n, v);
    
    // Calculate 2D sample
    float r1 = sample.x;
    float r2 = sample.y;
    
    // Transform to spherical coordinates
    float sinpsi = sin(2*PI*r1);
    float cospsi = cos(2*PI*r1);
    float costheta = pow(1.f - r2, 1.f/(e + 1.f));
    float sintheta = sqrt(1.f - costheta * costheta);
    
    // Return the result
    return normalize(u * sintheta * cospsi + v * sintheta * sinpsi + n * costheta);
}

float2 Sample_MapToDisk(
    // Sample
    float2 sample
    )
{
    float r = native_sqrt(sample.x); 
    float theta = 2 * PI * sample.y;
    return make_float2(r * native_cos(theta), r * native_sin(theta));
}

float2 Sample_MapToDiskConcentric(
    // Sample
    float2 sample
    )
{
    float2 offset = 2.f * sample - make_float2(1.f, 1.f);

    if (offset.x == 0 && offset.y == 0) return 0.f;

    float theta, r;

    if (fabs(offset.x) > fabs(offset.y)) 
    {
        r = offset.x;
        theta = PI / 4.f * (offset.y / offset.x);
    }
    else 
    {
        r = offset.y;
        theta = PI / 2.f * ( 1.f - 0.5f * (offset.x / offset.y));
    }
    
    return make_float2(r * native_cos(theta), r * native_sin(theta));
}

/// Sample hemisphere with cos weight
float3 Sample_MapToSphere(
                        // Sample
                        float2 sample
                        )
{
    float z = 1.f - 2.f * sample.x;
    float r = native_sqrt(max(0.f, 1.f - z*z));
    float phi = 2.f * PI * sample.y;
    float x = cos(phi);
    float y = sin(phi);
    
    // Return the result
    return make_float3(x,y,z);
}

float2 Sample_MapToPolygon(int n, float2 sample, float sample1)
{
    float theta = 2.f * PI / n;
    int edge = clamp((int)(sample1 * n), 0, n - 1);
    float t = native_sqrt(sample.x);
    float u = 1.f - t;
    float v = t * sample.y;
    float2 v1 = make_float2(native_cos(theta * edge), native_sin(theta * edge));
    float2 v2 = make_float2(native_cos(theta * (edge + 1)), native_sin(theta * (edge + 1)));
    return u*v1 + v*v2;;
}

/// Power heuristic for multiple importance sampling
float PowerHeuristic(int nf, float fpdf, int ng, float gpdf)
{
    float f = nf * fpdf;
    float g = ng * gpdf;
    return (f*f) / (f*f + g*g);
}

/// Balance heuristic for multiple importance sampling
float BalanceHeuristic(int nf, float fpdf, int ng, float gpdf)
{
    float f = nf * fpdf;
    float g = ng * gpdf;
    return (f) / (f + g);
}

int lower_bound(GLOBAL float const* values, int n, float value)
{
    int count = n;
    int b = 0;
    int it = 0;
    int step = 0;

    while (count > 0)
    {
        it = b;
        step = count / 2;
        it += step;
        if (values[it] < value)
        {
            b = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }

    return b;
}
/// Sample 1D distribution
float Distribution1D_Sample(float s, GLOBAL int const* data, float* pdf)
{
    int num_segments = data[0];

    GLOBAL float const* cdf_data = (GLOBAL float const*)&data[1];
    GLOBAL float const* pdf_data = cdf_data + num_segments + 1;

    int segment_idx = max(lower_bound(cdf_data, num_segments + 1, s), 1);

    // Find lerp coefficient
    float du = (s - cdf_data[segment_idx - 1]) / (cdf_data[segment_idx] - cdf_data[segment_idx - 1]);

    // Calc pdf
    *pdf = pdf_data[segment_idx - 1];

    return (segment_idx - 1 + du) / num_segments;;
}

/// Sample 1D distribution
int Distribution1D_SampleDiscrete(float s, GLOBAL int const* data, float* pdf)
{
    int num_segments = data[0];

    GLOBAL float const* cdf_data = (GLOBAL float const*)&data[1];
    GLOBAL float const* pdf_data = cdf_data + num_segments + 1;

    int segment_idx = max(lower_bound(cdf_data, num_segments + 1, s), 1);

    // Find lerp coefficient
    float du = (s - cdf_data[segment_idx - 1]) / (cdf_data[segment_idx] - cdf_data[segment_idx - 1]);

    // Calc pdf
    *pdf = pdf_data[segment_idx - 1] / num_segments;

    return segment_idx - 1;
}

/// PDF of  1D distribution
float Distribution1D_GetPdf(float s, GLOBAL int const* data)
{
    int num_segments = data[0];
    GLOBAL float const* cdf_data = (GLOBAL float const*)&data[1];
    GLOBAL float const* pdf_data = cdf_data + num_segments + 1;

    int segment_idx = max(lower_bound(cdf_data, num_segments + 1, s), 1);

    // Calc pdf
    return pdf_data[segment_idx - 1];
}

/// PDF of  1D distribution
float Distribution1D_GetPdfDiscreet(int d, GLOBAL int const* data)
{
    int num_segments = data[0];
    GLOBAL float const* cdf_data = (GLOBAL float const*)&data[1];
    GLOBAL float const* pdf_data = cdf_data + num_segments + 1;

    // Calc pdf
    return pdf_data[d] / num_segments;
}



#endif // SAMPLING_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef BXDF_CL
#define BXDF_CL
/**********************************************************************
 * Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ********************************************************************/
#ifndef BXDF_FLAGS_CL
#define BXDF_FLAGS_CL

#define DENOM_EPS 1e-8f
#define ROUGHNESS_EPS 0.0001f

enum BxdfFlags
{
    kBxdfFlagsSingular = (1 << 0),
    kBxdfFlagsBrdf = (1 << 1),
    kBxdfFlagsEmissive = (1 << 2),
    kBxdfFlagsTransparency = (1 << 3),
    kBxdfFlagsDiffuse = (1 << 4),

    //Used to mask value from bxdf_flags
    kBxdfFlagsAll = (kBxdfFlagsSingular | kBxdfFlagsBrdf | kBxdfFlagsEmissive | kBxdfFlagsTransparency | kBxdfFlagsDiffuse)
};

enum BxdfUberV2SampledComponent
{
    kBxdfUberV2SampleTransparency = 0,
    kBxdfUberV2SampleCoating = 1,
    kBxdfUberV2SampleReflection = 2,
    kBxdfUberV2SampleRefraction = 3,
    kBxdfUberV2SampleDiffuse = 4
};

/// Returns BxDF flags. Flags stored in first byte of bxdf_flags
int Bxdf_GetFlags(DifferentialGeometry const* dg)
{
    return (dg->mat.flags & kBxdfFlagsAll);
}

/// Sets BxDF flags. Flags stored in first byte of bxdf_flags
void Bxdf_SetFlags(DifferentialGeometry *dg, int flags)
{
    dg->mat.flags &= 0xffffff00; //Reset flags
    dg->mat.flags |= flags; //Set new flags
}

/// Return BxDF sampled component. Sampled component stored in second byte of bxdf_flags
int Bxdf_UberV2_GetSampledComponent(DifferentialGeometry const* dg)
{
    return (dg->mat.flags >> 8) & 0xff;
}

/// Sets BxDF sampled component. Sampled component stored in second byte of bxdf_flags
void Bxdf_UberV2_SetSampledComponent(DifferentialGeometry *dg, int sampledComponent)
{
    dg->mat.flags &= 0xffff00ff; //Reset sampled component
    dg->mat.flags |= (sampledComponent << 8); //Set new component
}

#endif


/// Schlick's approximation of Fresnel equtions
float SchlickFresnel(float eta, float ndotw)
{
    const float f = ((1.f - eta) / (1.f + eta)) * ((1.f - eta) / (1.f + eta));
    const float m = 1.f - fabs(ndotw);
    const float m2 = m*m;
    return f + (1.f - f) * m2 * m2 * m;
}

/// Full Fresnel equations
float FresnelDielectric(float etai, float etat, float ndotwi, float ndotwt)
{
    // Parallel and perpendicular polarization
    float rparl = ((etat * ndotwi) - (etai * ndotwt)) / ((etat * ndotwi) + (etai * ndotwt));
    float rperp = ((etai * ndotwi) - (etat * ndotwt)) / ((etai * ndotwi) + (etat * ndotwt));
    return (rparl*rparl + rperp*rperp) * 0.5f;
}
#ifndef BXDF_UBERV2_CL
#define BXDF_UBERV2_CL
#ifndef INPUTMAPS_CL
#define INPUTMAPS_CL

float4 ReadInputMap1(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[0].float_value.value, 0.0f))
	);
}
float4 ReadInputMap2(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[3].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[4].float_value.value, 0.0f))
	);
}
float4 ReadInputMap886(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[5].float_value.value, 0.0f))
	);
}
float4 ReadInputMap888(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[6].float_value.value, 0.0f))
	);
}
float4 ReadInputMap889(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[7].float_value.value, 0.0f))
	);
}
float4 ReadInputMap890(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[8].float_value.value, 0.0f))
	);
}
float4 ReadInputMap894(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[10].int_values.idx))
	, 
		((float4)(input_map_values[9].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[11].float_value.value, 0.0f))
	);
}
float4 ReadInputMap902(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[12].float_value.value, 0.0f))
	);
}
float4 ReadInputMap903(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[13].float_value.value, 0.0f))
	);
}
float4 ReadInputMap904(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[14].float_value.value, 0.0f))
	);
}
float4 ReadInputMap908(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[16].int_values.idx))
	, 
		((float4)(input_map_values[15].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap914(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[17].float_value.value, 0.0f))
	);
}
float4 ReadInputMap916(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[18].float_value.value, 0.0f))
	);
}
float4 ReadInputMap917(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[19].float_value.value, 0.0f))
	);
}
float4 ReadInputMap918(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[20].float_value.value, 0.0f))
	);
}
float4 ReadInputMap922(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[22].int_values.idx))
	, 
		((float4)(input_map_values[21].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap928(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[23].float_value.value, 0.0f))
	);
}
float4 ReadInputMap930(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[24].float_value.value, 0.0f))
	);
}
float4 ReadInputMap931(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[25].float_value.value, 0.0f))
	);
}
float4 ReadInputMap932(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[26].float_value.value, 0.0f))
	);
}
float4 ReadInputMap936(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[28].int_values.idx))
	, 
		((float4)(input_map_values[27].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap942(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[29].float_value.value, 0.0f))
	);
}
float4 ReadInputMap944(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[30].float_value.value, 0.0f))
	);
}
float4 ReadInputMap945(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[31].float_value.value, 0.0f))
	);
}
float4 ReadInputMap946(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[32].float_value.value, 0.0f))
	);
}
float4 ReadInputMap950(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[34].int_values.idx))
	, 
		((float4)(input_map_values[33].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap956(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[35].float_value.value, 0.0f))
	);
}
float4 ReadInputMap958(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[36].float_value.value, 0.0f))
	);
}
float4 ReadInputMap959(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[37].float_value.value, 0.0f))
	);
}
float4 ReadInputMap960(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[38].float_value.value, 0.0f))
	);
}
float4 ReadInputMap964(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[40].int_values.idx))
	, 
		((float4)(input_map_values[39].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap970(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[41].float_value.value, 0.0f))
	);
}
float4 ReadInputMap972(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[42].float_value.value, 0.0f))
	);
}
float4 ReadInputMap973(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[43].float_value.value, 0.0f))
	);
}
float4 ReadInputMap974(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[44].float_value.value, 0.0f))
	);
}
float4 ReadInputMap978(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[46].int_values.idx))
	, 
		((float4)(input_map_values[45].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap984(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[47].float_value.value, 0.0f))
	);
}
float4 ReadInputMap986(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[48].float_value.value, 0.0f))
	);
}
float4 ReadInputMap987(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[49].float_value.value, 0.0f))
	);
}
float4 ReadInputMap988(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[50].float_value.value, 0.0f))
	);
}
float4 ReadInputMap992(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[52].int_values.idx))
	, 
		((float4)(input_map_values[51].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap998(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[53].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1000(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[54].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1001(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[55].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1002(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[56].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[58].int_values.idx))
	, 
		((float4)(input_map_values[57].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap1012(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[59].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1014(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[60].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1015(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[61].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1016(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[62].float_value.value, 0.0f))
	);
}
float4 ReadInputMap1020(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[64].int_values.idx))
	, 
		((float4)(input_map_values[63].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3211(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[65].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3213(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[66].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3214(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[67].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3215(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[68].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3219(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[70].int_values.idx))
	, 
		((float4)(input_map_values[69].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3225(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[71].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3227(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[72].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3228(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[73].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3229(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[74].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3233(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[76].int_values.idx))
	, 
		((float4)(input_map_values[75].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3271(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[77].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3273(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[78].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3274(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[79].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3275(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[80].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3279(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[82].int_values.idx))
	, 
		((float4)(input_map_values[81].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3285(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[83].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3287(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[84].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3288(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[85].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3289(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[86].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3293(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[88].int_values.idx))
	, 
		((float4)(input_map_values[87].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3560(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[89].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3562(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[90].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3563(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[91].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3564(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[92].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3568(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[94].int_values.idx))
	, 
		((float4)(input_map_values[93].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3574(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[95].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3576(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[96].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3577(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[97].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[98].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3582(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[100].int_values.idx))
	, 
		((float4)(input_map_values[99].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3608(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[102].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3609(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[103].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3610(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[104].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3619(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[106].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3620(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[107].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3622(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[108].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3623(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[109].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3624(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[110].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3625(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[111].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3626(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[112].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3633(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[113].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3634(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[114].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3636(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[115].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3637(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[116].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3638(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[117].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3642(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[119].int_values.idx))
	, 
		((float4)(input_map_values[105].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3646(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[121].int_values.idx))
	, 
		((float4)(input_map_values[101].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3652(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[123].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3653(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[124].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3655(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[125].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3656(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[126].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3663(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[127].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3666(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[128].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3667(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[129].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3668(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[130].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3676(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[132].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3677(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[133].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3679(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[134].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3680(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[135].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3684(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[136].int_values.idx))
	, 
		((float4)(input_map_values[122].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3689(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[137].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3692(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[138].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3693(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[139].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3694(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[140].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3698(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[141].int_values.idx))
	, 
		((float4)(input_map_values[118].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3702(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[143].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[143].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[142].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[144].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[144].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[144].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap3715(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[145].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3716(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[146].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3718(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[147].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3719(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[148].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3720(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[149].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3723(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[150].int_values.idx))
	, 
		((float4)(input_map_values[131].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3732(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[151].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3733(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[152].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3735(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[153].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3736(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[154].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3737(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[155].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3743(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[156].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3744(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[157].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3746(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[158].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3747(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[159].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3748(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[160].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3754(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[161].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3755(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[162].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3757(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[163].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3758(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[164].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3759(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[165].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3771(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[166].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3772(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[167].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3774(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[168].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3775(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[169].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3776(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[170].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3782(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[171].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3783(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[172].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3785(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[173].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3786(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[174].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3787(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[175].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3793(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[176].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3794(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[177].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3796(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[178].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3797(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[179].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3798(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[180].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3804(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[181].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3805(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[182].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3807(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[183].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3808(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[184].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3809(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[185].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3815(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[186].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3816(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[187].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3818(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[188].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3819(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[189].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3820(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[190].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3826(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[191].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3827(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[192].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3829(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[193].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3830(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[194].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3831(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[195].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3838(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[196].int_values.idx))
	, 
		((float4)(input_map_values[120].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3844(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[198].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3845(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[199].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3847(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[200].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3848(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[201].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3855(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[203].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3858(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[204].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3859(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[205].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3860(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[206].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3879(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[209].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3880(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[210].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3882(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[211].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3883(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[212].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3887(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[213].int_values.idx))
	, 
		((float4)(input_map_values[202].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3893(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[215].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3899(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[216].int_values.idx))
	, 
		((float4)(input_map_values[207].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3905(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[217].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3906(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[218].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3908(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[219].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3909(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[220].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3910(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[221].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3911(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[222].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3912(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[223].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3913(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[224].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3916(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[225].int_values.idx))
	, 
		((float4)(input_map_values[197].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3925(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[226].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3928(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[227].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3929(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[228].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3930(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[229].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3931(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[230].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3941(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[231].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3942(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[232].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3943(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[233].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3944(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[234].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3953(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[235].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3954(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[236].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3956(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[237].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3957(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[238].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3958(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[239].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3964(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[240].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3965(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[241].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3967(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[242].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3968(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[243].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3969(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[244].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3975(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[246].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3976(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[247].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3978(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[248].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3979(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[249].float_value.value, 0.0f))
	);
}
float4 ReadInputMap3983(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[250].int_values.idx))
	, 
		((float4)(input_map_values[208].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap3989(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[252].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4001(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[254].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4004(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[255].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4005(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[256].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[257].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4010(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[258].int_values.idx))
	, 
		((float4)(input_map_values[214].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4011(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[259].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4012(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[260].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4013(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[261].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4019(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[262].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4022(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[263].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4023(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[264].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4024(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[265].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4025(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[266].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4026(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[267].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4027(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[268].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4028(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[269].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4034(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[271].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4040(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[272].int_values.idx))
	, 
		((float4)(input_map_values[245].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4043(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[273].int_values.idx))
	, 
		((float4)(input_map_values[253].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4049(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[275].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4050(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[276].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4052(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[277].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4053(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[278].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4057(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[279].int_values.idx))
	, 
		((float4)(input_map_values[251].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4058(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[280].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4059(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[281].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4068(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[283].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4071(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[284].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4072(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[285].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4073(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[286].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4077(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[287].int_values.idx))
	, 
		((float4)(input_map_values[270].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4078(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[288].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4079(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[289].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4080(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[290].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4083(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[291].int_values.idx))
	, 
		((float4)(input_map_values[251].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4089(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[292].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4090(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[293].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4092(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[294].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4093(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[295].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4094(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[296].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4095(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[297].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4096(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[298].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4097(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[299].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4100(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[300].int_values.idx))
	, 
		((float4)(input_map_values[274].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4106(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[302].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4107(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[303].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4109(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[304].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4110(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[305].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4111(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[306].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4112(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[307].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4113(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[308].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4117(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[309].int_values.idx))
	, 
		((float4)(input_map_values[282].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4123(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[311].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4126(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[312].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4127(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[313].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4128(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[314].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4132(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[315].int_values.idx))
	, 
		((float4)(input_map_values[301].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4135(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[316].int_values.idx))
	, 
		((float4)(input_map_values[310].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4143(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[318].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4144(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[319].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4146(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[320].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4147(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[321].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4159(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[323].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4165(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[324].int_values.idx))
	, 
		((float4)(input_map_values[317].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4171(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[326].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4172(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[327].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4174(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[328].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4175(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[329].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4179(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[330].int_values.idx))
	, 
		((float4)(input_map_values[322].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4180(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[331].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4181(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[332].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4185(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[333].int_values.idx))
	, 
		((float4)(input_map_values[325].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4193(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[335].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4194(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[336].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4199(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[337].int_values.idx))
	, 
		((float4)(input_map_values[322].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4205(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[339].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4219(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[340].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4222(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[341].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4223(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[342].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4224(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[343].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4230(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[344].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4233(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[345].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4234(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[346].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4235(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[347].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4236(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[348].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4242(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[349].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4245(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[350].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4246(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[351].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4247(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[352].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4248(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[353].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4254(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[354].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4257(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[355].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4258(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[356].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4259(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[357].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4260(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[358].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4266(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[359].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4269(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[360].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4270(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[361].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4271(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[362].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4279(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[364].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4280(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[365].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4282(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[366].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4283(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[367].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4287(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[368].int_values.idx))
	, 
		((float4)(input_map_values[338].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4288(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[369].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4289(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[370].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4293(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[371].int_values.idx))
	, 
		((float4)(input_map_values[334].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4294(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[372].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4298(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[373].int_values.idx))
	, 
		((float4)(input_map_values[363].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4304(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[374].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4305(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[375].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4307(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[376].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4308(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[377].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4309(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[378].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4312(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[379].int_values.idx))
	, 
		((float4)(input_map_values[334].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4318(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[381].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4319(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[382].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4324(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[383].int_values.idx))
	, 
		((float4)(input_map_values[338].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4330(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[385].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4341(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[387].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4342(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[388].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4344(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[389].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4345(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[390].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4349(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[391].int_values.idx))
	, 
		((float4)(input_map_values[384].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4350(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[392].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4351(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[393].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4355(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[394].int_values.idx))
	, 
		((float4)(input_map_values[386].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4361(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[396].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4362(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[397].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4364(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[398].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4365(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[399].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4369(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[400].int_values.idx))
	, 
		((float4)(input_map_values[380].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4370(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[401].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4374(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[402].int_values.idx))
	, 
		((float4)(input_map_values[384].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4380(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[404].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4386(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[405].int_values.idx))
	, 
		((float4)(input_map_values[380].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4389(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[406].int_values.idx))
	, 
		((float4)(input_map_values[395].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4397(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[407].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[408].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4400(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[409].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4401(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[410].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4402(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[411].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4408(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[412].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4409(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[413].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4411(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[414].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4412(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[415].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4413(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[416].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4425(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[417].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4426(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[418].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4428(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[419].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4429(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[420].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4430(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[421].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4436(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[422].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4437(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[423].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4439(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[424].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4440(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[425].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4441(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[426].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4447(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[427].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4448(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[428].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4450(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[429].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4451(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[430].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4452(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[431].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4458(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[432].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4459(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[433].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4461(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[434].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4462(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[435].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4463(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[436].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4469(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[437].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4470(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[438].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4472(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[439].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4473(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[440].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4474(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[441].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4480(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[442].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4481(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[443].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4483(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[444].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4484(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[445].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4485(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[446].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4492(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[447].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4493(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[448].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4495(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[449].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4496(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[450].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4497(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[451].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4504(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[452].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4505(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[453].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4507(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[454].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4508(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[455].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4509(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[456].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4521(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[457].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4522(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[458].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4524(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[459].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4525(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[460].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4526(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[461].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4527(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[462].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4528(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[463].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4529(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[464].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4535(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[465].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4536(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[466].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4538(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[467].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4539(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[468].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4540(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[469].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4541(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[470].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4542(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[471].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4543(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[472].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4549(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[473].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4550(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[474].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4552(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[475].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4553(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[476].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4554(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[477].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4555(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[478].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4556(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[479].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4557(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[480].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4563(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[481].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4564(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[482].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4566(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[483].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4567(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[484].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4568(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[485].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4569(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[486].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4570(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[487].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4571(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[488].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4577(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[489].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[490].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4580(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[491].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4581(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[492].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4582(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[493].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4583(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[494].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4584(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[495].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4585(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[496].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4591(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[497].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4592(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[498].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4594(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[499].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4595(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[500].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4596(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[501].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4597(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[502].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4598(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[503].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4599(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[504].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4602(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[505].int_values.idx))
	, 
		((float4)(input_map_values[403].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4603(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[506].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4604(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[507].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4617(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[508].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4618(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[509].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4620(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[510].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4621(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[511].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4622(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[512].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4632(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[514].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4635(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[515].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4636(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[516].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4637(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[517].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4641(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[518].int_values.idx))
	, 
		((float4)(input_map_values[403].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4647(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[520].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4656(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[521].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4657(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[522].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4659(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[523].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4660(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[524].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4661(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[525].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4664(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[526].int_values.idx))
	, 
		((float4)(input_map_values[513].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4670(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[528].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4673(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[529].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4674(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[530].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4675(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[531].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4682(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[532].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4683(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[533].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4685(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[534].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4686(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[535].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4687(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[536].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4690(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[537].int_values.idx))
	, 
		((float4)(input_map_values[519].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4691(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[538].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4692(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[539].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4696(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[540].int_values.idx))
	, 
		((float4)(input_map_values[527].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4702(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[542].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4703(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[543].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4705(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[544].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4706(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[545].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4707(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[546].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4708(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[547].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4709(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[548].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4716(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[549].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4717(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[550].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4719(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[551].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4720(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[552].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4721(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[553].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4724(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[554].int_values.idx))
	, 
		((float4)(input_map_values[519].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4727(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[555].int_values.idx))
	, 
		((float4)(input_map_values[541].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4733(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[557].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4734(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[558].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4736(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[559].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4737(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[560].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4738(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[561].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4739(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[562].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4740(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[563].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4749(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[565].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4761(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[567].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4764(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[568].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4765(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[569].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4766(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[570].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4773(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[571].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4774(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[572].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4776(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[573].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4777(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[574].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4778(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[575].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4781(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[576].int_values.idx))
	, 
		((float4)(input_map_values[556].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4784(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[577].int_values.idx))
	, 
		((float4)(input_map_values[566].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4790(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[579].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4793(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[580].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4794(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[581].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4795(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[582].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4799(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[583].int_values.idx))
	, 
		((float4)(input_map_values[564].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4800(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[584].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4801(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[585].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4808(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[586].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4809(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[587].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4811(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[588].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4812(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[589].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4813(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[590].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4821(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[592].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4822(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[593].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4824(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[594].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4825(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[595].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4829(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[596].int_values.idx))
	, 
		((float4)(input_map_values[578].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4835(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[598].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4838(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[599].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4839(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[600].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4840(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[601].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4844(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[602].int_values.idx))
	, 
		((float4)(input_map_values[564].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4850(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[604].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4851(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[605].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4862(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[607].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4863(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[608].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4865(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[609].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4866(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[610].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4870(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[611].int_values.idx))
	, 
		((float4)(input_map_values[591].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4882(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[613].int_values.idx))
	, 
		((float4)(input_map_values[603].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4883(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[614].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4887(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[615].int_values.idx))
	, 
		((float4)(input_map_values[597].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4890(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[616].int_values.idx))
	, 
		((float4)(input_map_values[606].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4896(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[618].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4897(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[619].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4899(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[620].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[621].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4904(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[622].int_values.idx))
	, 
		((float4)(input_map_values[612].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4907(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[623].int_values.idx))
	, 
		((float4)(input_map_values[603].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4919(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[625].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4922(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[626].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4923(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[627].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4924(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[628].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4928(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[629].int_values.idx))
	, 
		((float4)(input_map_values[617].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4934(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[631].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4935(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[632].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4937(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[633].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4938(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[634].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4942(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[635].int_values.idx))
	, 
		((float4)(input_map_values[630].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4945(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[636].int_values.idx))
	, 
		((float4)(input_map_values[624].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4951(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[638].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4954(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[639].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4955(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[640].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4956(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[641].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4965(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[643].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4968(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[644].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4969(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[645].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4970(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[646].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4974(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[647].int_values.idx))
	, 
		((float4)(input_map_values[637].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap4980(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[649].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4981(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[650].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4983(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[651].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4984(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[652].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4997(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[654].float_value.value, 0.0f))
	);
}
float4 ReadInputMap4998(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[655].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5000(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[656].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5001(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[657].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5005(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[658].int_values.idx))
	, 
		((float4)(input_map_values[642].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5011(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[660].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5012(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[661].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5014(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[662].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5015(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[663].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5019(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[664].int_values.idx))
	, 
		((float4)(input_map_values[648].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5025(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[666].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5028(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[667].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5029(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[668].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5030(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[669].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5034(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[670].int_values.idx))
	, 
		((float4)(input_map_values[653].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5040(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[672].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5041(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[673].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5043(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[674].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5044(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[675].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5048(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[676].int_values.idx))
	, 
		((float4)(input_map_values[659].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5059(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[679].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5060(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[680].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5062(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[681].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5063(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[682].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5067(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[683].int_values.idx))
	, 
		((float4)(input_map_values[665].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5073(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[685].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5076(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[686].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5077(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[687].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5078(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[688].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5082(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[689].int_values.idx))
	, 
		((float4)(input_map_values[671].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5088(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[691].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5089(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[692].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5091(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[693].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5092(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[694].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5093(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[695].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5094(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[696].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5095(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[697].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5102(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[698].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5105(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[699].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5106(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[700].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5107(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[701].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5111(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[702].int_values.idx))
	, 
		((float4)(input_map_values[678].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5117(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[704].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5118(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[705].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5120(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[706].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5121(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[707].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5125(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[708].int_values.idx))
	, 
		((float4)(input_map_values[684].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5131(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[710].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5134(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[711].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5135(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[712].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5136(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[713].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5140(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[714].int_values.idx))
	, 
		((float4)(input_map_values[690].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5146(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[716].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5147(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[717].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5149(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[718].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5150(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[719].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5154(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[720].int_values.idx))
	, 
		((float4)(input_map_values[677].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5158(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[722].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[722].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[721].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[723].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[723].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[723].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap5170(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[725].int_values.idx))
	, 
		((float4)(input_map_values[709].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5173(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[726].int_values.idx))
	, 
		((float4)(input_map_values[703].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5180(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[728].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5183(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[729].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5184(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[730].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5185(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[731].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5193(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[733].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5194(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[734].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5196(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[735].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5197(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[736].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5201(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[737].int_values.idx))
	, 
		((float4)(input_map_values[715].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5207(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[739].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5208(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[740].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5210(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[741].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5211(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[742].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5215(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[743].int_values.idx))
	, 
		((float4)(input_map_values[724].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5218(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[744].int_values.idx))
	, 
		((float4)(input_map_values[732].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5221(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[745].int_values.idx))
	, 
		((float4)(input_map_values[727].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5224(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[746].int_values.idx))
	, 
		((float4)(input_map_values[738].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5247(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[749].int_values.idx))
	, 
		((float4)(input_map_values[748].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5259(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[751].int_values.idx))
	, 
		((float4)(input_map_values[747].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5270(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[753].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5276(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[754].int_values.idx))
	, 
		((float4)(input_map_values[750].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5279(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[755].int_values.idx))
	, 
		((float4)(input_map_values[752].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5280(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[756].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5281(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[757].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5285(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[758].int_values.idx))
	, 
		((float4)(input_map_values[752].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5291(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[760].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5294(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[761].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5295(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[762].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5296(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[763].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5300(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[764].int_values.idx))
	, 
		((float4)(input_map_values[759].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5306(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[766].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5309(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[767].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5310(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[768].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5311(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[769].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5330(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[771].int_values.idx))
	, 
		((float4)(input_map_values[765].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5336(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[773].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5342(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[774].int_values.idx))
	, 
		((float4)(input_map_values[770].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5348(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[776].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5349(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[777].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5351(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[778].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5352(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[779].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5358(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[781].int_values.idx))
	, 
		((float4)(input_map_values[772].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5359(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[782].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5360(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[783].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5361(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[784].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5362(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[785].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5363(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[786].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5367(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[787].int_values.idx))
	, 
		((float4)(input_map_values[775].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5379(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[789].int_values.idx))
	, 
		((float4)(input_map_values[772].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5385(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[791].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5386(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[792].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5388(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[793].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5389(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[794].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5390(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[795].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5391(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[796].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5392(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[797].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5405(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[798].int_values.idx))
	, 
		((float4)(input_map_values[788].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5417(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[800].int_values.idx))
	, 
		((float4)(input_map_values[790].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5420(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[801].int_values.idx))
	, 
		((float4)(input_map_values[780].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5424(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[803].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[803].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[802].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[804].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[804].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[804].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap5427(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[805].int_values.idx))
	, 
		((float4)(input_map_values[799].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5439(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[807].int_values.idx))
	, 
		((float4)(input_map_values[806].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5451(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[809].int_values.idx))
	, 
		((float4)(input_map_values[808].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5461(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[811].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5464(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[812].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5465(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[813].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5466(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[814].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5478(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[816].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5479(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[817].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5481(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[818].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5482(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[819].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5510(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[822].int_values.idx))
	, 
		((float4)(input_map_values[815].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5516(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[824].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5517(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[825].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5519(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[826].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5520(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[827].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5524(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[828].int_values.idx))
	, 
		((float4)(input_map_values[820].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5530(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[830].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5531(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[831].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5536(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[832].int_values.idx))
	, 
		((float4)(input_map_values[810].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5542(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[834].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5548(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[835].int_values.idx))
	, 
		((float4)(input_map_values[823].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5554(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[837].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5555(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[838].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5557(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[839].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5558(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[840].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5562(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[841].int_values.idx))
	, 
		((float4)(input_map_values[821].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5568(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[843].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5569(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[844].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5571(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[845].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5572(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[846].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5576(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[847].int_values.idx))
	, 
		((float4)(input_map_values[833].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5577(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[848].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[849].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5582(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[850].int_values.idx))
	, 
		((float4)(input_map_values[829].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5583(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[851].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5587(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[852].int_values.idx))
	, 
		((float4)(input_map_values[836].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5590(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[853].int_values.idx))
	, 
		((float4)(input_map_values[842].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5597(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[855].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5598(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[856].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5600(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[857].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5601(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[858].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5605(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[859].int_values.idx))
	, 
		((float4)(input_map_values[829].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5608(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[860].int_values.idx))
	, 
		((float4)(input_map_values[833].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5620(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[862].int_values.idx))
	, 
		((float4)(input_map_values[854].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5627(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[864].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5628(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[865].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5633(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[866].int_values.idx))
	, 
		((float4)(input_map_values[861].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5639(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[868].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5642(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[869].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5643(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[870].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5644(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[871].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5648(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[872].int_values.idx))
	, 
		((float4)(input_map_values[863].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5649(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[873].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5653(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[874].int_values.idx))
	, 
		((float4)(input_map_values[867].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5658(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[876].int_values.idx))
	, 
		((float4)(input_map_values[863].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5673(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[878].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5676(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[879].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5677(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[880].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5678(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[881].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5689(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[883].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5690(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[884].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5692(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[885].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5693(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[886].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5694(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[887].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5695(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[888].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5696(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[889].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5700(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[890].int_values.idx))
	, 
		((float4)(input_map_values[877].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5706(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[892].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5707(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[893].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5709(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[894].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5710(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[895].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5714(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[896].int_values.idx))
	, 
		((float4)(input_map_values[875].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5717(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[897].int_values.idx))
	, 
		((float4)(input_map_values[882].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5723(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[899].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5724(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[900].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5726(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[901].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5727(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[902].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5737(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[904].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5738(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[905].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5740(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[906].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5741(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[907].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5745(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[908].int_values.idx))
	, 
		((float4)(input_map_values[891].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5751(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[910].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5754(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[911].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5755(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[912].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5756(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[913].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5762(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[915].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5765(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[916].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5766(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[917].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5767(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[918].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5771(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[919].int_values.idx))
	, 
		((float4)(input_map_values[898].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5775(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[921].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5778(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[922].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5779(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[923].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5780(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[924].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5784(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[925].int_values.idx))
	, 
		((float4)(input_map_values[903].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5790(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[927].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5791(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[928].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5793(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[929].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5794(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[930].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5798(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[931].int_values.idx))
	, 
		((float4)(input_map_values[909].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5807(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[933].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5808(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[934].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5810(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[935].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5811(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[936].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5815(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[937].int_values.idx))
	, 
		((float4)(input_map_values[914].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5821(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[939].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5822(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[940].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5824(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[941].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5825(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[942].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5829(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[943].int_values.idx))
	, 
		((float4)(input_map_values[926].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5841(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[945].int_values.idx))
	, 
		((float4)(input_map_values[920].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5859(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[947].int_values.idx))
	, 
		((float4)(input_map_values[938].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5862(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[948].int_values.idx))
	, 
		((float4)(input_map_values[932].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5868(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[950].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5869(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[951].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5871(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[952].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5872(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[953].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5876(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[954].int_values.idx))
	, 
		((float4)(input_map_values[944].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5879(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[955].int_values.idx))
	, 
		((float4)(input_map_values[946].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5885(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[957].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5886(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[958].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5888(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[959].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5889(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[960].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5893(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[961].int_values.idx))
	, 
		((float4)(input_map_values[949].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5899(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[963].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[964].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5902(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[965].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5903(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[966].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5916(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[969].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5917(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[970].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5919(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[971].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5920(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[972].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5921(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[973].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5922(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[974].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5923(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[975].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5927(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[976].int_values.idx))
	, 
		((float4)(input_map_values[962].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5930(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[977].int_values.idx))
	, 
		((float4)(input_map_values[956].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5936(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[979].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5939(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[980].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5940(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[981].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5941(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[982].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5949(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[984].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5950(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[985].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5952(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[986].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5953(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[987].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5957(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[988].int_values.idx))
	, 
		((float4)(input_map_values[968].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5963(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[990].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5964(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[991].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5966(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[992].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5967(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[993].float_value.value, 0.0f))
	);
}
float4 ReadInputMap5980(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[994].int_values.idx))
	, 
		((float4)(input_map_values[983].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5983(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[995].int_values.idx))
	, 
		((float4)(input_map_values[978].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap5995(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[997].int_values.idx))
	, 
		((float4)(input_map_values[989].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6001(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[999].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6002(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1000].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6004(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1001].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6005(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1002].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6009(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1003].int_values.idx))
	, 
		((float4)(input_map_values[967].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6013(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1005].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1005].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1004].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1006].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1006].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1006].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6016(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1007].int_values.idx))
	, 
		((float4)(input_map_values[996].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6022(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1009].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6025(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1010].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6026(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1011].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6027(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1012].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6028(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1013].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6029(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1014].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6030(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1015].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6034(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1016].int_values.idx))
	, 
		((float4)(input_map_values[998].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6040(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1018].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6041(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1019].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6043(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1020].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6044(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1021].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6048(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1022].int_values.idx))
	, 
		((float4)(input_map_values[1017].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6051(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1023].int_values.idx))
	, 
		((float4)(input_map_values[1008].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6069(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1026].int_values.idx))
	, 
		((float4)(input_map_values[1024].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6075(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1027].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6078(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1028].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6079(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1029].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6080(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1030].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6084(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1031].int_values.idx))
	, 
		((float4)(input_map_values[1025].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6088(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1033].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1033].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1032].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1034].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1034].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1034].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6102(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1037].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6103(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1038].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6111(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1040].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6114(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1041].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6115(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1042].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6116(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1043].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6123(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1044].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6126(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1045].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6127(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1046].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6128(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1047].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6138(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1048].int_values.idx))
	, 
		((float4)(input_map_values[1039].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6144(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1050].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6147(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1051].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6148(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1052].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6149(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1053].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6153(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1054].int_values.idx))
	, 
		((float4)(input_map_values[1036].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6154(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1055].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6161(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1057].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6164(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1058].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6165(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1059].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6166(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1060].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6170(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1061].int_values.idx))
	, 
		((float4)(input_map_values[1035].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6174(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1063].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1063].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1062].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1064].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1064].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1064].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6178(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1066].int_values.idx))
	, 
		((float4)(input_map_values[1056].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6184(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1068].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6185(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1069].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6187(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1070].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6188(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1071].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6192(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1072].int_values.idx))
	, 
		((float4)(input_map_values[1036].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6204(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1074].int_values.idx))
	, 
		((float4)(input_map_values[1049].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6210(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1076].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6213(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1077].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6214(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1078].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6215(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1079].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6222(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1080].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6225(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1081].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6226(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1082].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6227(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1083].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6231(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1084].int_values.idx))
	, 
		((float4)(input_map_values[1067].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6237(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1086].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6238(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1087].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6240(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1088].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6241(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1089].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6245(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1090].int_values.idx))
	, 
		((float4)(input_map_values[1075].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6251(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1092].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6254(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1093].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6255(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1094].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6256(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1095].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6260(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1096].int_values.idx))
	, 
		((float4)(input_map_values[1073].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6272(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1098].int_values.idx))
	, 
		((float4)(input_map_values[1065].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6276(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1100].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1100].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1099].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1101].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1101].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1101].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6280(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1103].int_values.idx))
	, 
		((float4)(input_map_values[1085].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6286(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1105].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6287(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1106].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6289(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1107].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6290(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1108].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6294(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1109].int_values.idx))
	, 
		((float4)(input_map_values[1091].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6297(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1110].int_values.idx))
	, 
		((float4)(input_map_values[1097].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6303(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1111].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6306(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1112].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6307(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1113].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6308(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1114].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6312(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1115].int_values.idx))
	, 
		((float4)(input_map_values[1104].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6318(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1117].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6321(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1118].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6322(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1119].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6323(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1120].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6333(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1122].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6334(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1123].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6336(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1124].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6337(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1125].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6341(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1126].int_values.idx))
	, 
		((float4)(input_map_values[1102].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6345(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1128].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1128].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1127].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1129].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1129].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1129].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6349(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1131].int_values.idx))
	, 
		((float4)(input_map_values[1121].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6353(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1133].int_values.idx))
	, 
		((float4)(input_map_values[1116].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6359(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1135].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6360(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1136].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6368(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1137].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6371(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1138].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6372(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1139].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6373(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1140].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6386(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1141].int_values.idx))
	, 
		((float4)(input_map_values[1134].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6387(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1142].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6391(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1143].int_values.idx))
	, 
		((float4)(input_map_values[1130].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6395(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1145].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1145].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1144].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1146].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1146].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1146].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1147].int_values.idx))
	, 
		((float4)(input_map_values[1134].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6401(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1148].int_values.idx))
	, 
		((float4)(input_map_values[1132].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6405(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1150].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1150].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1149].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1151].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1151].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1151].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6430(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1155].int_values.idx))
	, 
		((float4)(input_map_values[1153].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6436(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1156].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6437(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1157].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6439(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1158].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6440(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1159].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6448(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1161].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6451(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1162].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6452(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1163].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6453(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1164].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6457(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1165].int_values.idx))
	, 
		((float4)(input_map_values[1152].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6461(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1167].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1167].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1166].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1168].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1168].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1168].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6464(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1169].int_values.idx))
	, 
		((float4)(input_map_values[1154].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6468(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1171].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1171].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1170].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1172].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1172].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1172].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6471(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1173].int_values.idx))
	, 
		((float4)(input_map_values[1160].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6477(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1175].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6488(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1177].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6489(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1178].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6491(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1179].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6492(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1180].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6496(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1181].int_values.idx))
	, 
		((float4)(input_map_values[1174].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6497(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1182].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6498(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1183].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6502(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1184].int_values.idx))
	, 
		((float4)(input_map_values[1176].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6508(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1186].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6509(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1187].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6511(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1188].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6512(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1189].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6516(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1190].int_values.idx))
	, 
		((float4)(input_map_values[1174].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6519(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1192].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6522(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1193].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6523(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1194].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6524(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1195].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6528(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1196].int_values.idx))
	, 
		((float4)(input_map_values[1185].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6531(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1197].int_values.idx))
	, 
		((float4)(input_map_values[1191].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6537(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1199].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6543(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1200].int_values.idx))
	, 
		((float4)(input_map_values[1198].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6544(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1201].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6545(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1202].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6549(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1203].int_values.idx))
	, 
		((float4)(input_map_values[1198].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6558(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1205].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6561(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1206].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6562(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1207].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6563(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1208].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6567(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1209].int_values.idx))
	, 
		((float4)(input_map_values[1204].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6573(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1211].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6576(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1212].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6577(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1213].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1214].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6582(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1215].int_values.idx))
	, 
		((float4)(input_map_values[1210].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6588(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1217].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6591(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1218].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6592(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1219].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6593(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1220].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6608(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1222].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6609(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1223].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6611(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1224].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6612(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1225].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6619(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1226].int_values.idx))
	, 
		((float4)(input_map_values[1216].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6631(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1228].int_values.idx))
	, 
		((float4)(input_map_values[1221].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6637(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1230].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6638(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1231].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6640(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1232].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6641(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1233].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6649(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1234].int_values.idx))
	, 
		((float4)(input_map_values[1227].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6668(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1236].int_values.idx))
	, 
		((float4)(input_map_values[1229].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6674(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1238].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6675(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1239].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6677(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1240].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6678(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1241].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6685(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1243].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6686(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1244].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6688(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1245].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6689(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1246].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6699(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1247].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6703(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1249].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1249].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1248].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1250].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1250].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1250].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6707(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1251].int_values.idx))
	, 
		((float4)(input_map_values[1235].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6719(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1253].int_values.idx))
	, 
		((float4)(input_map_values[1237].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6725(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1255].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6726(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1256].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6728(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1257].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6729(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1258].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6733(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1259].int_values.idx))
	, 
		((float4)(input_map_values[1242].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6739(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1261].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6740(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1262].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6742(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1263].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6743(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1264].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6753(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1265].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6757(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1267].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1267].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1266].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1268].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1268].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1268].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6761(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1269].int_values.idx))
	, 
		((float4)(input_map_values[1252].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6769(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1271].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6770(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1272].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6772(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1273].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6773(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1274].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6783(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1275].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6787(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1277].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1277].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1276].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1278].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1278].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1278].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6791(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1279].int_values.idx))
	, 
		((float4)(input_map_values[1260].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6797(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1281].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6798(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1282].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6800(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1283].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6801(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1284].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6805(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1285].int_values.idx))
	, 
		((float4)(input_map_values[1254].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6811(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1287].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6812(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1288].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6814(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1289].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6815(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1290].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6819(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1291].int_values.idx))
	, 
		((float4)(input_map_values[1270].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6831(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1293].int_values.idx))
	, 
		((float4)(input_map_values[1280].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6837(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1295].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6838(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1296].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6840(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1297].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6841(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1298].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6845(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1299].int_values.idx))
	, 
		((float4)(input_map_values[1286].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6851(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1301].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6852(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1302].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6854(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1303].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6855(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1304].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6865(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1305].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6869(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1307].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1307].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1306].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1308].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1308].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1308].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6872(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1309].int_values.idx))
	, 
		((float4)(input_map_values[1292].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6875(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1310].int_values.idx))
	, 
		((float4)(input_map_values[1294].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6879(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1312].int_values.idx))
	, 
		((float4)(input_map_values[1300].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6885(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1313].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6888(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1314].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6889(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1315].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6890(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1316].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6896(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1318].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6897(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1319].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6899(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1320].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1321].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6913(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1322].int_values.idx))
	, 
		((float4)(input_map_values[1317].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6921(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1324].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6922(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1325].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6924(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1326].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6925(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1327].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6949(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1330].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6950(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1331].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6955(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1332].int_values.idx))
	, 
		((float4)(input_map_values[1311].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6959(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1334].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1334].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1333].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1335].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1335].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1335].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap6963(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1337].int_values.idx))
	, 
		((float4)(input_map_values[1328].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6969(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1339].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6970(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1340].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6972(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1341].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6973(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1342].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6977(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1343].int_values.idx))
	, 
		((float4)(input_map_values[1323].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6983(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1345].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6984(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1346].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6989(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1347].int_values.idx))
	, 
		((float4)(input_map_values[1329].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6990(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1348].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6994(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1349].int_values.idx))
	, 
		((float4)(input_map_values[1344].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap6995(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1350].float_value.value, 0.0f))
	);
}
float4 ReadInputMap6999(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1351].int_values.idx))
	, 
		((float4)(input_map_values[1338].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7005(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1353].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1354].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7008(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1355].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7009(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1356].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7016(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1357].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7017(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1358].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7019(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1359].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7020(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1360].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7024(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1361].int_values.idx))
	, 
		((float4)(input_map_values[1329].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7030(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1363].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7033(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1364].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7034(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1365].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7035(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1366].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7039(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1367].int_values.idx))
	, 
		((float4)(input_map_values[1344].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7042(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1368].int_values.idx))
	, 
		((float4)(input_map_values[1352].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7048(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1370].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7049(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1371].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7051(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1372].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7052(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1373].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7056(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1374].int_values.idx))
	, 
		((float4)(input_map_values[1336].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7060(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1376].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1376].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1375].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1377].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1377].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1377].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap7068(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1379].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7069(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1380].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7071(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1381].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7072(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1382].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7076(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1383].int_values.idx))
	, 
		((float4)(input_map_values[1362].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7088(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1385].int_values.idx))
	, 
		((float4)(input_map_values[1378].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7100(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1387].int_values.idx))
	, 
		((float4)(input_map_values[1369].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7103(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1388].int_values.idx))
	, 
		((float4)(input_map_values[1384].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7109(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1390].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7110(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1391].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7112(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1392].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7113(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1393].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7117(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1394].int_values.idx))
	, 
		((float4)(input_map_values[1386].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7120(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1395].int_values.idx))
	, 
		((float4)(input_map_values[1389].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7129(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1397].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7130(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1398].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7132(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1399].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7133(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1400].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7145(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1402].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7146(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1403].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7147(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1404].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7151(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1405].int_values.idx))
	, 
		((float4)(input_map_values[1396].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7171(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1408].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7174(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1409].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7175(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1410].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7176(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1411].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7180(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1412].int_values.idx))
	, 
		((float4)(input_map_values[1401].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7199(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1415].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7200(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1416].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7202(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1417].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7203(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1418].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7207(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1419].int_values.idx))
	, 
		((float4)(input_map_values[1406].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7213(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1421].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7214(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1422].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7216(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1423].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7217(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1424].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7221(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1425].int_values.idx))
	, 
		((float4)(input_map_values[1407].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7227(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1427].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7230(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1428].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7231(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1429].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7232(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1430].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7236(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1431].int_values.idx))
	, 
		((float4)(input_map_values[1413].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7245(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1433].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7246(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1434].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7248(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1435].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7249(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1436].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7253(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1437].int_values.idx))
	, 
		((float4)(input_map_values[1414].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7259(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1439].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7260(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1440].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7262(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1441].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7263(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1442].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7267(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1443].int_values.idx))
	, 
		((float4)(input_map_values[1420].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7270(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1444].int_values.idx))
	, 
		((float4)(input_map_values[1432].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7276(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1446].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7277(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1447].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7279(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1448].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7280(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1449].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7284(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1450].int_values.idx))
	, 
		((float4)(input_map_values[1426].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7290(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1452].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7293(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1453].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7294(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1454].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7295(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1455].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7299(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1456].int_values.idx))
	, 
		((float4)(input_map_values[1438].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7309(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1459].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7312(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1460].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7313(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1461].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7314(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1462].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7318(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1463].int_values.idx))
	, 
		((float4)(input_map_values[1445].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7324(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1465].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7325(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1466].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7327(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1467].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7328(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1468].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7332(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1469].int_values.idx))
	, 
		((float4)(input_map_values[1451].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7338(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1471].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7339(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1472].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7346(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1473].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7347(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1474].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7349(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1475].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7350(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1476].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7354(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1477].int_values.idx))
	, 
		((float4)(input_map_values[1464].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7357(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1478].int_values.idx))
	, 
		((float4)(input_map_values[1458].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7360(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1480].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7363(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1481].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7364(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1482].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7365(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1483].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7369(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1484].int_values.idx))
	, 
		((float4)(input_map_values[1470].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7370(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1485].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7382(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1487].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7385(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1488].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7386(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1489].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7387(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1490].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7391(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1491].int_values.idx))
	, 
		((float4)(input_map_values[1457].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7395(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1493].int_values.idx))
	, 
		((float4)(input_map_values[1479].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1495].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7401(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1496].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7402(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1497].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7403(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1498].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7407(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1499].int_values.idx))
	, 
		((float4)(input_map_values[1470].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7413(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1501].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7414(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1502].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7419(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1503].int_values.idx))
	, 
		((float4)(input_map_values[1486].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7425(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1505].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7428(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1506].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7429(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1507].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7430(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1508].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7443(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1509].int_values.idx))
	, 
		((float4)(input_map_values[1500].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7444(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1510].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7448(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1511].int_values.idx))
	, 
		((float4)(input_map_values[1494].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7451(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1513].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7454(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1514].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7455(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1515].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7456(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1516].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7460(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1517].int_values.idx))
	, 
		((float4)(input_map_values[1492].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7464(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1519].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1519].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1518].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1520].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1520].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1520].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap7467(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1521].int_values.idx))
	, 
		((float4)(input_map_values[1504].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7473(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1523].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7476(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1524].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7477(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1525].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7478(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1526].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7486(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1528].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7487(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1529].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7492(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1530].int_values.idx))
	, 
		((float4)(input_map_values[1512].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7495(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1532].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7498(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1533].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7499(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1534].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7500(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1535].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7504(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1536].int_values.idx))
	, 
		((float4)(input_map_values[1500].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7512(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1537].int_values.idx))
	, 
		((float4)(input_map_values[1522].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7524(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1539].int_values.idx))
	, 
		((float4)(input_map_values[1527].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7525(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1540].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7532(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1542].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7535(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1543].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7536(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1544].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7537(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1545].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7541(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1546].int_values.idx))
	, 
		((float4)(input_map_values[1531].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7544(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1548].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7547(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1549].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7548(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1550].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7549(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1551].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7553(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1552].int_values.idx))
	, 
		((float4)(input_map_values[1538].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7559(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1554].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7562(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1555].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7563(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1556].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7564(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1557].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7568(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1558].int_values.idx))
	, 
		((float4)(input_map_values[1541].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7574(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1559].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7577(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1560].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1561].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7579(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1562].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7580(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1563].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7586(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1565].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7589(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1566].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7590(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1567].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7591(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1568].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7595(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1569].int_values.idx))
	, 
		((float4)(input_map_values[1527].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7598(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1570].int_values.idx))
	, 
		((float4)(input_map_values[1547].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7610(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1572].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7613(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1573].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7614(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1574].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7615(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1575].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7628(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1577].int_values.idx))
	, 
		((float4)(input_map_values[1553].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7631(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1578].int_values.idx))
	, 
		((float4)(input_map_values[1564].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7637(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1579].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7640(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1580].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7641(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1581].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7642(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1582].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7643(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1583].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7649(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1584].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7652(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1585].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7653(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1586].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7654(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1587].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7655(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1588].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7658(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1589].int_values.idx))
	, 
		((float4)(input_map_values[1576].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7670(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1591].int_values.idx))
	, 
		((float4)(input_map_values[1571].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7676(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1593].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7677(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1594].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7679(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1595].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7680(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1596].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7687(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1597].int_values.idx))
	, 
		((float4)(input_map_values[1590].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7693(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1599].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7694(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1600].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7696(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1601].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7697(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1602].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7701(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1603].int_values.idx))
	, 
		((float4)(input_map_values[1592].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7707(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1605].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7713(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1606].int_values.idx))
	, 
		((float4)(input_map_values[1604].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7714(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1607].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7715(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1608].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7719(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1609].int_values.idx))
	, 
		((float4)(input_map_values[1598].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7725(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1611].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7726(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1612].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7728(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1613].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7729(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1614].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7733(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1615].int_values.idx))
	, 
		((float4)(input_map_values[1610].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7736(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1616].int_values.idx))
	, 
		((float4)(input_map_values[1604].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7739(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1618].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7742(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1619].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7743(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1620].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7744(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1621].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7755(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1623].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7758(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1624].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7759(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1625].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7760(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1626].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7761(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1627].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7762(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1628].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7763(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1629].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7772(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1631].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7773(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1632].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7778(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1633].int_values.idx))
	, 
		((float4)(input_map_values[1617].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7784(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1635].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7785(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1636].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7787(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1637].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7788(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1638].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7792(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1639].int_values.idx))
	, 
		((float4)(input_map_values[1630].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7793(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1640].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7797(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1641].int_values.idx))
	, 
		((float4)(input_map_values[1622].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7803(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1642].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7806(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1643].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7807(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1644].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7808(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1645].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7809(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1646].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7815(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1647].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7818(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1648].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7819(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1649].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7820(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1650].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7821(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1651].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7822(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1652].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7823(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1653].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7824(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1654].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7830(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1655].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7833(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1656].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7834(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1657].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7835(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1658].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7836(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1659].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7837(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1660].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7838(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1661].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7839(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1662].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7858(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1665].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7859(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1666].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7864(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1667].int_values.idx))
	, 
		((float4)(input_map_values[1634].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7867(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1668].int_values.idx))
	, 
		((float4)(input_map_values[1630].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7873(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1670].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7874(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1671].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7876(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1672].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7877(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1673].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7885(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1675].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7886(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1676].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7888(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1677].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7889(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1678].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7902(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1679].int_values.idx))
	, 
		((float4)(input_map_values[1664].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7903(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1680].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7907(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1681].int_values.idx))
	, 
		((float4)(input_map_values[1674].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7916(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1683].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7919(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1684].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7920(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1685].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7921(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1686].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7925(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1687].int_values.idx))
	, 
		((float4)(input_map_values[1669].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7928(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1688].int_values.idx))
	, 
		((float4)(input_map_values[1663].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7932(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1690].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1690].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1689].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1691].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1691].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1691].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap7936(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1693].int_values.idx))
	, 
		((float4)(input_map_values[1664].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7942(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1695].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7948(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1696].int_values.idx))
	, 
		((float4)(input_map_values[1682].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7954(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1698].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7957(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1699].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7958(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1700].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7959(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1701].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7972(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1702].int_values.idx))
	, 
		((float4)(input_map_values[1694].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7973(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1703].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7974(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1704].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7978(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1705].int_values.idx))
	, 
		((float4)(input_map_values[1697].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7984(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1707].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7987(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1708].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7988(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1709].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7989(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1710].float_value.value, 0.0f))
	);
}
float4 ReadInputMap7993(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1711].int_values.idx))
	, 
		((float4)(input_map_values[1692].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap7997(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1713].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1713].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1712].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1714].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1714].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1714].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8000(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1715].int_values.idx))
	, 
		((float4)(input_map_values[1694].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1717].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8018(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1719].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8021(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1720].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8022(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1721].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8023(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1722].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8027(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1723].int_values.idx))
	, 
		((float4)(input_map_values[1706].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8033(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1725].int_values.idx))
	, 
		((float4)(input_map_values[1716].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8034(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1726].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8035(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1727].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8039(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1728].int_values.idx))
	, 
		((float4)(input_map_values[1718].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8045(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1730].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8048(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1731].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8049(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1732].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8050(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1733].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8063(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1734].int_values.idx))
	, 
		((float4)(input_map_values[1716].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8069(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1736].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8070(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1737].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8075(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1738].int_values.idx))
	, 
		((float4)(input_map_values[1729].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8081(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1739].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8084(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1740].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8085(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1741].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8086(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1742].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8087(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1743].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8090(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1744].int_values.idx))
	, 
		((float4)(input_map_values[1724].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8094(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1746].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1746].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1745].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1747].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1747].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1747].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8098(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1749].int_values.idx))
	, 
		((float4)(input_map_values[1735].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8099(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1750].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8131(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1752].int_values.idx))
	, 
		((float4)(input_map_values[1735].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8137(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1754].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8138(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1755].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8140(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1756].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8141(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1757].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8142(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1758].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8143(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1759].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8144(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1760].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8148(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1761].int_values.idx))
	, 
		((float4)(input_map_values[1748].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8152(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1763].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1763].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1762].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1764].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1764].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1764].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8155(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1765].int_values.idx))
	, 
		((float4)(input_map_values[1751].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8161(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1767].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8162(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1768].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8164(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1769].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8165(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1770].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8169(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1771].int_values.idx))
	, 
		((float4)(input_map_values[1753].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8175(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1773].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8181(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1774].int_values.idx))
	, 
		((float4)(input_map_values[1766].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8190(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1776].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8191(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1777].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8192(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1778].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8196(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1779].int_values.idx))
	, 
		((float4)(input_map_values[1772].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8197(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1780].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8198(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1781].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8202(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1782].int_values.idx))
	, 
		((float4)(input_map_values[1775].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8214(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1784].int_values.idx))
	, 
		((float4)(input_map_values[1772].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8220(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1786].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8221(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1787].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8226(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1788].int_values.idx))
	, 
		((float4)(input_map_values[1783].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8239(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1790].int_values.idx))
	, 
		((float4)(input_map_values[1785].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8240(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1791].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8244(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1792].int_values.idx))
	, 
		((float4)(input_map_values[1789].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8250(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1793].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8253(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1794].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8254(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1795].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8255(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1796].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8256(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1797].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8262(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1798].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8265(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1799].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8266(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1800].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8267(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1801].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8268(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1802].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8269(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1803].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8270(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1804].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8271(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1805].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8283(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1807].int_values.idx))
	, 
		((float4)(input_map_values[1785].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8289(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1809].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8290(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1810].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8295(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1811].int_values.idx))
	, 
		((float4)(input_map_values[1806].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8304(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1813].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8305(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1814].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8306(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1815].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8310(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1816].int_values.idx))
	, 
		((float4)(input_map_values[1808].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8311(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1817].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8331(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1820].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8334(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1821].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8335(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1822].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8336(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1823].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8340(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1824].int_values.idx))
	, 
		((float4)(input_map_values[1812].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8343(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1825].int_values.idx))
	, 
		((float4)(input_map_values[1808].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8349(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1827].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8350(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1828].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8360(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1830].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8361(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1831].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8363(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1832].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8364(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1833].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8368(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1834].int_values.idx))
	, 
		((float4)(input_map_values[1819].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8371(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1835].int_values.idx))
	, 
		((float4)(input_map_values[1818].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8380(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1837].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8381(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1838].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8383(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1839].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8384(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1840].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8388(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1841].int_values.idx))
	, 
		((float4)(input_map_values[1829].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8394(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1843].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8395(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1844].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8397(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1845].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1846].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8402(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1847].int_values.idx))
	, 
		((float4)(input_map_values[1826].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8403(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1848].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8410(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1850].int_values.idx))
	, 
		((float4)(input_map_values[1836].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8416(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1852].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8417(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1853].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8419(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1854].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8420(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1855].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8421(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1856].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8422(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1857].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8423(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1858].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8427(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1859].int_values.idx))
	, 
		((float4)(input_map_values[1842].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8430(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1860].int_values.idx))
	, 
		((float4)(input_map_values[1826].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8443(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1862].int_values.idx))
	, 
		((float4)(input_map_values[1851].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8457(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1864].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8458(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1865].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8470(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1867].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8473(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1868].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8474(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1869].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8475(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1870].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8476(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1871].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8477(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1872].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8478(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1873].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8482(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1874].int_values.idx))
	, 
		((float4)(input_map_values[1863].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8485(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1875].int_values.idx))
	, 
		((float4)(input_map_values[1849].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8489(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1877].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1877].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1876].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1878].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1878].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1878].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8493(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1880].int_values.idx))
	, 
		((float4)(input_map_values[1861].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8494(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1881].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8498(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1882].int_values.idx))
	, 
		((float4)(input_map_values[1866].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8504(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1884].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8505(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1885].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8507(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1886].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8508(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1887].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8534(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1889].int_values.idx))
	, 
		((float4)(input_map_values[1861].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8548(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1891].int_values.idx))
	, 
		((float4)(input_map_values[1883].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8554(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1893].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8557(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1894].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8558(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1895].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8559(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1896].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8560(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1897].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8561(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1898].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8562(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1899].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8566(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1900].int_values.idx))
	, 
		((float4)(input_map_values[1888].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1902].int_values.idx))
	, 
		((float4)(input_map_values[1879].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8582(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1904].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1904].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1903].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1905].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1905].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1905].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8585(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1906].int_values.idx))
	, 
		((float4)(input_map_values[1890].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8597(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1908].int_values.idx))
	, 
		((float4)(input_map_values[1901].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8603(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1910].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8604(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1911].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8609(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1912].int_values.idx))
	, 
		((float4)(input_map_values[1892].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8615(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1914].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8618(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1915].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8619(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1916].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8620(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1917].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8634(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1919].int_values.idx))
	, 
		((float4)(input_map_values[1907].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8641(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1921].int_values.idx))
	, 
		((float4)(input_map_values[1909].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8642(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1922].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8646(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1923].int_values.idx))
	, 
		((float4)(input_map_values[1913].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8649(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1924].int_values.idx))
	, 
		((float4)(input_map_values[1918].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8655(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1925].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8656(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1926].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8658(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1927].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8659(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1928].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8663(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1929].int_values.idx))
	, 
		((float4)(input_map_values[1909].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8675(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1931].int_values.idx))
	, 
		((float4)(input_map_values[1930].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8678(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1932].int_values.idx))
	, 
		((float4)(input_map_values[1920].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8682(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1934].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1934].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1933].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1935].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1935].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1935].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8701(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1938].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8702(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1939].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8704(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1940].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8705(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1941].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8709(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1942].int_values.idx))
	, 
		((float4)(input_map_values[1937].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8715(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1944].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8716(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1945].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8718(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1946].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8719(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1947].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8723(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1948].int_values.idx))
	, 
		((float4)(input_map_values[1936].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8727(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1950].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1950].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1949].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1951].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1951].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1951].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8731(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1953].int_values.idx))
	, 
		((float4)(input_map_values[1943].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8737(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1955].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8738(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1956].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8740(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1957].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8741(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1958].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8748(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1959].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8751(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1960].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8752(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1961].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8753(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1962].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8757(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1963].int_values.idx))
	, 
		((float4)(input_map_values[1954].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8767(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1964].int_values.idx))
	, 
		((float4)(input_map_values[1952].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8771(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1966].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1966].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1965].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1967].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1967].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1967].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8778(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1968].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8779(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1969].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8781(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1970].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8782(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1971].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8792(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1973].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8793(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1974].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8795(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1975].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8796(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1976].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8803(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1977].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8806(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1978].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8807(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1979].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8808(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1980].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8809(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1981].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8813(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[1983].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[1983].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1982].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[1984].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[1984].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[1984].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8817(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1986].int_values.idx))
	, 
		((float4)(input_map_values[1972].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8823(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1988].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8824(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1989].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8826(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1990].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8827(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1991].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8846(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1993].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8849(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1994].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8850(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1995].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8851(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1996].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8855(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[1997].int_values.idx))
	, 
		((float4)(input_map_values[1992].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8861(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[1999].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8862(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2000].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8864(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2001].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8865(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2002].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8869(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2003].int_values.idx))
	, 
		((float4)(input_map_values[1987].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8897(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2006].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8898(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2007].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2008].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8901(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2009].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8905(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2010].int_values.idx))
	, 
		((float4)(input_map_values[1985].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8909(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[2012].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[2012].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2011].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[2013].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[2013].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[2013].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap8913(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2015].int_values.idx))
	, 
		((float4)(input_map_values[1998].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8919(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2017].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8920(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2018].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8922(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2019].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8923(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2020].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8927(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2021].int_values.idx))
	, 
		((float4)(input_map_values[2005].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8939(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2023].int_values.idx))
	, 
		((float4)(input_map_values[2004].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8945(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2025].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8946(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2026].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8948(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2027].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8949(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2028].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8953(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2029].int_values.idx))
	, 
		((float4)(input_map_values[2016].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8959(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2030].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8969(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2032].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8970(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2033].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8972(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2034].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8973(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2035].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8977(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2036].int_values.idx))
	, 
		((float4)(input_map_values[2024].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap8983(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2038].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8984(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2039].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8986(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2040].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8987(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2041].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8988(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2042].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8989(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2043].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8990(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2044].float_value.value, 0.0f))
	);
}
float4 ReadInputMap8994(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2045].int_values.idx))
	, 
		((float4)(input_map_values[2022].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2047].int_values.idx))
	, 
		((float4)(input_map_values[2031].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9009(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2048].int_values.idx))
	, 
		((float4)(input_map_values[2014].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9010(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2049].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9011(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2050].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9012(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2051].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9016(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[2053].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[2053].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2052].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[2054].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[2054].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[2054].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap9022(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2055].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9025(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2056].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9026(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2057].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9027(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2058].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9028(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2059].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9029(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2060].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9030(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2061].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9042(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2063].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9045(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2064].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9046(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2065].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9047(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2066].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9054(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2068].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9057(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2069].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9058(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2070].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9059(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2071].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9063(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2072].int_values.idx))
	, 
		((float4)(input_map_values[2046].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9075(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2074].int_values.idx))
	, 
		((float4)(input_map_values[2037].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9079(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2076].int_values.idx))
	, 
		((float4)(input_map_values[2067].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9091(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2078].int_values.idx))
	, 
		((float4)(input_map_values[2073].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9097(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2080].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9098(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2081].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9100(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2082].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9101(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2083].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9105(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2084].int_values.idx))
	, 
		((float4)(input_map_values[2062].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9112(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2085].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9113(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2086].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9115(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2087].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9116(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2088].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9120(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2089].int_values.idx))
	, 
		((float4)(input_map_values[2079].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9123(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2090].int_values.idx))
	, 
		((float4)(input_map_values[2077].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9129(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2092].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9132(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2093].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9133(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2094].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9134(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2095].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9138(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2096].int_values.idx))
	, 
		((float4)(input_map_values[2075].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9142(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[2098].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[2098].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2097].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[2099].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[2099].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[2099].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap9148(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2101].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9149(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2102].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9151(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2103].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9152(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2104].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9167(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2106].int_values.idx))
	, 
		((float4)(input_map_values[2091].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9173(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2108].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9176(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2109].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9177(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2110].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9178(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2111].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9182(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2112].int_values.idx))
	, 
		((float4)(input_map_values[2105].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9194(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2114].int_values.idx))
	, 
		((float4)(input_map_values[2100].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9200(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2116].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9201(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2117].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9203(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2118].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9204(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2119].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9208(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2120].int_values.idx))
	, 
		((float4)(input_map_values[2113].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9211(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2121].int_values.idx))
	, 
		((float4)(input_map_values[2107].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9218(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2123].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9221(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2124].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9222(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2125].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9223(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2126].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9227(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2127].int_values.idx))
	, 
		((float4)(input_map_values[2115].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9233(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2129].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9234(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2130].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9236(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2131].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9237(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2132].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9247(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2134].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9248(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2135].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9250(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2136].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9251(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2137].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9255(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2138].int_values.idx))
	, 
		((float4)(input_map_values[2122].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9258(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2140].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9261(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2141].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9262(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2142].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9263(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2143].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9267(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2144].int_values.idx))
	, 
		((float4)(input_map_values[2128].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9273(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2146].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9274(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2147].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9276(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2148].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9277(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2149].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9281(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2150].int_values.idx))
	, 
		((float4)(input_map_values[2133].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9290(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2152].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9291(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2153].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9292(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2154].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9296(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2155].int_values.idx))
	, 
		((float4)(input_map_values[2145].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9299(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2156].int_values.idx))
	, 
		((float4)(input_map_values[2139].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9302(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2158].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9305(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2159].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9306(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2160].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9307(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2161].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9312(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2163].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9315(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2164].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9316(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2165].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9317(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2166].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9321(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2167].int_values.idx))
	, 
		((float4)(input_map_values[2151].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9327(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2169].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9330(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2170].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9331(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2171].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9332(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2172].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9336(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2173].int_values.idx))
	, 
		((float4)(input_map_values[2157].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9339(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2175].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9342(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2176].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9343(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2177].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9344(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2178].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9348(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2179].int_values.idx))
	, 
		((float4)(input_map_values[2162].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9351(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2180].int_values.idx))
	, 
		((float4)(input_map_values[2168].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9354(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2181].int_values.idx))
	, 
		((float4)(input_map_values[2174].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9365(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2183].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9366(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2184].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9368(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2185].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9369(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2186].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9378(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2188].int_values.idx))
	, 
		((float4)(input_map_values[2182].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9384(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2190].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9385(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2191].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9387(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2192].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9388(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2193].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9394(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2194].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9397(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2195].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2196].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9399(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2197].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9403(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2198].int_values.idx))
	, 
		((float4)(input_map_values[2187].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9415(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2200].int_values.idx))
	, 
		((float4)(input_map_values[2189].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9421(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2202].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9422(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2203].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9424(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2204].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9425(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2205].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9429(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2206].int_values.idx))
	, 
		((float4)(input_map_values[2201].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9435(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2208].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9436(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2209].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9438(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2210].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9439(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2211].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9443(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2212].int_values.idx))
	, 
		((float4)(input_map_values[2199].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9452(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2213].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9458(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2214].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9461(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2215].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9462(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2216].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9463(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2217].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9464(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2218].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9480(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2220].int_values.idx))
	, 
		((float4)(input_map_values[2207].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9486(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2222].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9487(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2223].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9489(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2224].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9490(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2225].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9505(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2227].int_values.idx))
	, 
		((float4)(input_map_values[2219].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9511(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2229].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9512(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2230].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9514(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2231].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9515(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2232].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9516(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2233].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9517(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2234].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9518(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2235].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9522(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2236].int_values.idx))
	, 
		((float4)(input_map_values[2226].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9534(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2238].int_values.idx))
	, 
		((float4)(input_map_values[2221].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9543(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2240].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9544(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2241].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9546(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2242].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9547(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2243].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9551(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2244].int_values.idx))
	, 
		((float4)(input_map_values[2228].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9563(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2246].int_values.idx))
	, 
		((float4)(input_map_values[2237].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9566(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2247].int_values.idx))
	, 
		((float4)(input_map_values[2239].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9569(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2249].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9572(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2250].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9573(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2251].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9574(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2252].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9578(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2253].int_values.idx))
	, 
		((float4)(input_map_values[2245].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9584(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2255].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9585(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2256].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9587(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2257].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9588(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2258].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9597(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2260].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9598(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2261].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9600(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2262].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9601(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2263].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9605(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2264].int_values.idx))
	, 
		((float4)(input_map_values[2248].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9611(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2266].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9612(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2267].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9614(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2268].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9615(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2269].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9616(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2270].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9617(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2271].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9618(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2272].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9622(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2273].int_values.idx))
	, 
		((float4)(input_map_values[2254].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9625(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2274].int_values.idx))
	, 
		((float4)(input_map_values[2259].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9631(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2276].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9632(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2277].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9634(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2278].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9635(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2279].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9646(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2281].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9647(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2282].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9649(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2283].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9650(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2284].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9654(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2285].int_values.idx))
	, 
		((float4)(input_map_values[2265].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9657(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2286].int_values.idx))
	, 
		((float4)(input_map_values[2280].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9663(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2288].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9664(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2289].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9666(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2290].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9667(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2291].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9671(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2292].int_values.idx))
	, 
		((float4)(input_map_values[2275].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9676(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2294].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9679(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2295].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9680(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2296].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9681(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2297].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9696(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2299].int_values.idx))
	, 
		((float4)(input_map_values[2287].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9708(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2301].int_values.idx))
	, 
		((float4)(input_map_values[2293].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9711(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2303].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9714(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2304].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9715(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2305].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9716(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2306].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9720(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2307].int_values.idx))
	, 
		((float4)(input_map_values[2298].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9726(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2309].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9729(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2310].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9730(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2311].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9731(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2312].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9735(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2313].int_values.idx))
	, 
		((float4)(input_map_values[2300].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9747(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2315].int_values.idx))
	, 
		((float4)(input_map_values[2302].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9756(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2317].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9757(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2318].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9762(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2319].int_values.idx))
	, 
		((float4)(input_map_values[2308].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9765(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2320].int_values.idx))
	, 
		((float4)(input_map_values[2314].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9773(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2322].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9776(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2323].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9777(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2324].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9778(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2325].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9779(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2326].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9780(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2327].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9781(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2328].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9785(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2329].int_values.idx))
	, 
		((float4)(input_map_values[2316].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9786(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2330].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9797(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2332].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9800(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2333].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9801(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2334].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9802(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2335].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9806(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2336].int_values.idx))
	, 
		((float4)(input_map_values[2321].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9812(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2338].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9813(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2339].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9818(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2340].int_values.idx))
	, 
		((float4)(input_map_values[2316].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9824(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2342].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9825(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2343].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9830(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2344].int_values.idx))
	, 
		((float4)(input_map_values[2331].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9836(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2346].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9839(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2347].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9840(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2348].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9841(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2349].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9845(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2350].int_values.idx))
	, 
		((float4)(input_map_values[2337].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9846(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2351].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9850(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2352].int_values.idx))
	, 
		((float4)(input_map_values[2341].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9851(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2353].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9855(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2354].int_values.idx))
	, 
		((float4)(input_map_values[2345].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9867(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2356].int_values.idx))
	, 
		((float4)(input_map_values[2337].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9873(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2358].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9876(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2359].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9877(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2360].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9878(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2361].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9882(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2362].int_values.idx))
	, 
		((float4)(input_map_values[2341].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9885(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2364].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9888(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2365].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9889(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2366].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9890(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2367].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9894(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2368].int_values.idx))
	, 
		((float4)(input_map_values[2355].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9900(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2370].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9901(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2371].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9903(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2372].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9904(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2373].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9908(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2374].int_values.idx))
	, 
		((float4)(input_map_values[2357].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9911(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2376].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9914(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2377].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9915(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2378].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9916(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2379].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9920(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2380].int_values.idx))
	, 
		((float4)(input_map_values[2363].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9923(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2381].int_values.idx))
	, 
		((float4)(input_map_values[2369].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9941(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2383].int_values.idx))
	, 
		((float4)(input_map_values[2375].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9944(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2385].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9947(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2386].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9948(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2387].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9949(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2388].float_value.value, 0.0f))
	);
}
float4 ReadInputMap9967(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2390].int_values.idx))
	, 
		((float4)(input_map_values[2382].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9979(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2392].int_values.idx))
	, 
		((float4)(input_map_values[2384].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9982(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2393].int_values.idx))
	, 
		((float4)(input_map_values[2389].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap9994(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2395].int_values.idx))
	, 
		((float4)(input_map_values[2391].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10006(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2397].int_values.idx))
	, 
		((float4)(input_map_values[2394].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10018(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2399].int_values.idx))
	, 
		((float4)(input_map_values[2396].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10030(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2401].int_values.idx))
	, 
		((float4)(input_map_values[2398].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10042(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2403].int_values.idx))
	, 
		((float4)(input_map_values[2400].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10054(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2405].int_values.idx))
	, 
		((float4)(input_map_values[2404].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10066(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2407].int_values.idx))
	, 
		((float4)(input_map_values[2402].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10082(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2409].int_values.idx))
	, 
		((float4)(input_map_values[2406].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10085(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2410].int_values.idx))
	, 
		((float4)(input_map_values[2408].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10104(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2412].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10105(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2413].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10116(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2414].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10117(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2415].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10119(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2416].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10120(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2417].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10121(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2418].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10124(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2419].int_values.idx))
	, 
		((float4)(input_map_values[2411].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10125(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2420].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10129(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2421].int_values.idx))
	, 
		((float4)(input_map_values[2411].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10135(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2423].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10138(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2424].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10139(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2425].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10140(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2426].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10144(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2427].int_values.idx))
	, 
		((float4)(input_map_values[2422].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10150(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2429].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10153(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2430].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10154(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2431].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10155(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2432].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10159(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2433].int_values.idx))
	, 
		((float4)(input_map_values[2428].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10165(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2435].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10168(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2436].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10169(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2437].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10170(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2438].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10174(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2439].int_values.idx))
	, 
		((float4)(input_map_values[2434].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10180(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2441].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10183(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2442].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10184(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2443].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10185(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2444].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10189(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2445].int_values.idx))
	, 
		((float4)(input_map_values[2440].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10208(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2447].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10211(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2448].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10212(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2449].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10213(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2450].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10217(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2451].int_values.idx))
	, 
		((float4)(input_map_values[2446].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10223(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2453].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10224(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2454].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10226(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2455].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10227(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2456].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10228(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2457].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10229(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2458].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10230(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2459].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10234(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2460].int_values.idx))
	, 
		((float4)(input_map_values[2452].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10240(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2462].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10243(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2463].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10244(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2464].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10245(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2465].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10249(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2466].int_values.idx))
	, 
		((float4)(input_map_values[2461].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10255(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2468].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10258(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2469].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10259(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2470].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10260(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2471].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10264(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2472].int_values.idx))
	, 
		((float4)(input_map_values[2467].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10270(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2474].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10273(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2475].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10274(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2476].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10275(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2477].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10279(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2478].int_values.idx))
	, 
		((float4)(input_map_values[2473].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10285(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2480].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10288(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2481].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10289(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2482].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10290(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2483].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10294(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2484].int_values.idx))
	, 
		((float4)(input_map_values[2479].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10301(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2486].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10307(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2487].int_values.idx))
	, 
		((float4)(input_map_values[2485].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10308(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2488].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10309(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2489].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10313(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2490].int_values.idx))
	, 
		((float4)(input_map_values[2485].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10317(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	mix((float4)(
		((float4)(input_map_values[2492].float_value.value, 0.0f))
.x)	, 
		(float4)(
		((float4)(input_map_values[2492].float_value.value, 0.0f))
.y)	, 
		((
		(float4)(Texture_SampleBump(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2491].int_values.idx)), 1.0f)
) - 
		(
		((float4)(input_map_values[2493].float_value.value, 0.0f))
.x)) / 
		((
		((float4)(input_map_values[2493].float_value.value, 0.0f))
.y)  - 
		(
		((float4)(input_map_values[2493].float_value.value, 0.0f))
.x)))	
	);
}
float4 ReadInputMap10323(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2495].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10326(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2496].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10327(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2497].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10328(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2498].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10332(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2499].int_values.idx))
	, 
		((float4)(input_map_values[2494].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10338(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2501].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10341(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2502].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10342(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2503].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10343(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2504].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10347(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2505].int_values.idx))
	, 
		((float4)(input_map_values[2500].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10353(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2507].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10356(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2508].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10357(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2509].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10358(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2510].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10362(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2511].int_values.idx))
	, 
		((float4)(input_map_values[2506].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10368(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2513].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10371(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2514].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10372(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2515].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10373(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2516].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10377(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2517].int_values.idx))
	, 
		((float4)(input_map_values[2512].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10383(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2519].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10386(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2520].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10387(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2521].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10388(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2522].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10392(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2523].int_values.idx))
	, 
		((float4)(input_map_values[2518].float_value.value, 0.0f))
.x	)
	);
}
float4 ReadInputMap10398(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2525].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10399(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2526].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10401(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2527].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10402(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	((float4)(input_map_values[2528].float_value.value, 0.0f))
	);
}
float4 ReadInputMap10406(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return (float4)(
	pow(
		Texture_Sample2D(dg->uv, TEXTURE_ARGS_IDX(input_map_values[2529].int_values.idx))
	, 
		((float4)(input_map_values[2524].float_value.value, 0.0f))
.x	)
	);
}
float4 GetInputMapFloat4(uint input_id, DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	switch(input_id)
	{
		case 1: return ReadInputMap1(dg, input_map_values, TEXTURE_ARGS);
		case 2: return ReadInputMap2(dg, input_map_values, TEXTURE_ARGS);
		case 4: return ReadInputMap4(dg, input_map_values, TEXTURE_ARGS);
		case 5: return ReadInputMap5(dg, input_map_values, TEXTURE_ARGS);
		case 6: return ReadInputMap6(dg, input_map_values, TEXTURE_ARGS);
		case 886: return ReadInputMap886(dg, input_map_values, TEXTURE_ARGS);
		case 888: return ReadInputMap888(dg, input_map_values, TEXTURE_ARGS);
		case 889: return ReadInputMap889(dg, input_map_values, TEXTURE_ARGS);
		case 890: return ReadInputMap890(dg, input_map_values, TEXTURE_ARGS);
		case 894: return ReadInputMap894(dg, input_map_values, TEXTURE_ARGS);
		case 900: return ReadInputMap900(dg, input_map_values, TEXTURE_ARGS);
		case 902: return ReadInputMap902(dg, input_map_values, TEXTURE_ARGS);
		case 903: return ReadInputMap903(dg, input_map_values, TEXTURE_ARGS);
		case 904: return ReadInputMap904(dg, input_map_values, TEXTURE_ARGS);
		case 908: return ReadInputMap908(dg, input_map_values, TEXTURE_ARGS);
		case 914: return ReadInputMap914(dg, input_map_values, TEXTURE_ARGS);
		case 916: return ReadInputMap916(dg, input_map_values, TEXTURE_ARGS);
		case 917: return ReadInputMap917(dg, input_map_values, TEXTURE_ARGS);
		case 918: return ReadInputMap918(dg, input_map_values, TEXTURE_ARGS);
		case 922: return ReadInputMap922(dg, input_map_values, TEXTURE_ARGS);
		case 928: return ReadInputMap928(dg, input_map_values, TEXTURE_ARGS);
		case 930: return ReadInputMap930(dg, input_map_values, TEXTURE_ARGS);
		case 931: return ReadInputMap931(dg, input_map_values, TEXTURE_ARGS);
		case 932: return ReadInputMap932(dg, input_map_values, TEXTURE_ARGS);
		case 936: return ReadInputMap936(dg, input_map_values, TEXTURE_ARGS);
		case 942: return ReadInputMap942(dg, input_map_values, TEXTURE_ARGS);
		case 944: return ReadInputMap944(dg, input_map_values, TEXTURE_ARGS);
		case 945: return ReadInputMap945(dg, input_map_values, TEXTURE_ARGS);
		case 946: return ReadInputMap946(dg, input_map_values, TEXTURE_ARGS);
		case 950: return ReadInputMap950(dg, input_map_values, TEXTURE_ARGS);
		case 956: return ReadInputMap956(dg, input_map_values, TEXTURE_ARGS);
		case 958: return ReadInputMap958(dg, input_map_values, TEXTURE_ARGS);
		case 959: return ReadInputMap959(dg, input_map_values, TEXTURE_ARGS);
		case 960: return ReadInputMap960(dg, input_map_values, TEXTURE_ARGS);
		case 964: return ReadInputMap964(dg, input_map_values, TEXTURE_ARGS);
		case 970: return ReadInputMap970(dg, input_map_values, TEXTURE_ARGS);
		case 972: return ReadInputMap972(dg, input_map_values, TEXTURE_ARGS);
		case 973: return ReadInputMap973(dg, input_map_values, TEXTURE_ARGS);
		case 974: return ReadInputMap974(dg, input_map_values, TEXTURE_ARGS);
		case 978: return ReadInputMap978(dg, input_map_values, TEXTURE_ARGS);
		case 984: return ReadInputMap984(dg, input_map_values, TEXTURE_ARGS);
		case 986: return ReadInputMap986(dg, input_map_values, TEXTURE_ARGS);
		case 987: return ReadInputMap987(dg, input_map_values, TEXTURE_ARGS);
		case 988: return ReadInputMap988(dg, input_map_values, TEXTURE_ARGS);
		case 992: return ReadInputMap992(dg, input_map_values, TEXTURE_ARGS);
		case 998: return ReadInputMap998(dg, input_map_values, TEXTURE_ARGS);
		case 1000: return ReadInputMap1000(dg, input_map_values, TEXTURE_ARGS);
		case 1001: return ReadInputMap1001(dg, input_map_values, TEXTURE_ARGS);
		case 1002: return ReadInputMap1002(dg, input_map_values, TEXTURE_ARGS);
		case 1006: return ReadInputMap1006(dg, input_map_values, TEXTURE_ARGS);
		case 1012: return ReadInputMap1012(dg, input_map_values, TEXTURE_ARGS);
		case 1014: return ReadInputMap1014(dg, input_map_values, TEXTURE_ARGS);
		case 1015: return ReadInputMap1015(dg, input_map_values, TEXTURE_ARGS);
		case 1016: return ReadInputMap1016(dg, input_map_values, TEXTURE_ARGS);
		case 1020: return ReadInputMap1020(dg, input_map_values, TEXTURE_ARGS);
		case 3211: return ReadInputMap3211(dg, input_map_values, TEXTURE_ARGS);
		case 3213: return ReadInputMap3213(dg, input_map_values, TEXTURE_ARGS);
		case 3214: return ReadInputMap3214(dg, input_map_values, TEXTURE_ARGS);
		case 3215: return ReadInputMap3215(dg, input_map_values, TEXTURE_ARGS);
		case 3219: return ReadInputMap3219(dg, input_map_values, TEXTURE_ARGS);
		case 3225: return ReadInputMap3225(dg, input_map_values, TEXTURE_ARGS);
		case 3227: return ReadInputMap3227(dg, input_map_values, TEXTURE_ARGS);
		case 3228: return ReadInputMap3228(dg, input_map_values, TEXTURE_ARGS);
		case 3229: return ReadInputMap3229(dg, input_map_values, TEXTURE_ARGS);
		case 3233: return ReadInputMap3233(dg, input_map_values, TEXTURE_ARGS);
		case 3271: return ReadInputMap3271(dg, input_map_values, TEXTURE_ARGS);
		case 3273: return ReadInputMap3273(dg, input_map_values, TEXTURE_ARGS);
		case 3274: return ReadInputMap3274(dg, input_map_values, TEXTURE_ARGS);
		case 3275: return ReadInputMap3275(dg, input_map_values, TEXTURE_ARGS);
		case 3279: return ReadInputMap3279(dg, input_map_values, TEXTURE_ARGS);
		case 3285: return ReadInputMap3285(dg, input_map_values, TEXTURE_ARGS);
		case 3287: return ReadInputMap3287(dg, input_map_values, TEXTURE_ARGS);
		case 3288: return ReadInputMap3288(dg, input_map_values, TEXTURE_ARGS);
		case 3289: return ReadInputMap3289(dg, input_map_values, TEXTURE_ARGS);
		case 3293: return ReadInputMap3293(dg, input_map_values, TEXTURE_ARGS);
		case 3560: return ReadInputMap3560(dg, input_map_values, TEXTURE_ARGS);
		case 3562: return ReadInputMap3562(dg, input_map_values, TEXTURE_ARGS);
		case 3563: return ReadInputMap3563(dg, input_map_values, TEXTURE_ARGS);
		case 3564: return ReadInputMap3564(dg, input_map_values, TEXTURE_ARGS);
		case 3568: return ReadInputMap3568(dg, input_map_values, TEXTURE_ARGS);
		case 3574: return ReadInputMap3574(dg, input_map_values, TEXTURE_ARGS);
		case 3576: return ReadInputMap3576(dg, input_map_values, TEXTURE_ARGS);
		case 3577: return ReadInputMap3577(dg, input_map_values, TEXTURE_ARGS);
		case 3578: return ReadInputMap3578(dg, input_map_values, TEXTURE_ARGS);
		case 3582: return ReadInputMap3582(dg, input_map_values, TEXTURE_ARGS);
		case 3608: return ReadInputMap3608(dg, input_map_values, TEXTURE_ARGS);
		case 3609: return ReadInputMap3609(dg, input_map_values, TEXTURE_ARGS);
		case 3610: return ReadInputMap3610(dg, input_map_values, TEXTURE_ARGS);
		case 3619: return ReadInputMap3619(dg, input_map_values, TEXTURE_ARGS);
		case 3620: return ReadInputMap3620(dg, input_map_values, TEXTURE_ARGS);
		case 3622: return ReadInputMap3622(dg, input_map_values, TEXTURE_ARGS);
		case 3623: return ReadInputMap3623(dg, input_map_values, TEXTURE_ARGS);
		case 3624: return ReadInputMap3624(dg, input_map_values, TEXTURE_ARGS);
		case 3625: return ReadInputMap3625(dg, input_map_values, TEXTURE_ARGS);
		case 3626: return ReadInputMap3626(dg, input_map_values, TEXTURE_ARGS);
		case 3633: return ReadInputMap3633(dg, input_map_values, TEXTURE_ARGS);
		case 3634: return ReadInputMap3634(dg, input_map_values, TEXTURE_ARGS);
		case 3636: return ReadInputMap3636(dg, input_map_values, TEXTURE_ARGS);
		case 3637: return ReadInputMap3637(dg, input_map_values, TEXTURE_ARGS);
		case 3638: return ReadInputMap3638(dg, input_map_values, TEXTURE_ARGS);
		case 3642: return ReadInputMap3642(dg, input_map_values, TEXTURE_ARGS);
		case 3646: return ReadInputMap3646(dg, input_map_values, TEXTURE_ARGS);
		case 3652: return ReadInputMap3652(dg, input_map_values, TEXTURE_ARGS);
		case 3653: return ReadInputMap3653(dg, input_map_values, TEXTURE_ARGS);
		case 3655: return ReadInputMap3655(dg, input_map_values, TEXTURE_ARGS);
		case 3656: return ReadInputMap3656(dg, input_map_values, TEXTURE_ARGS);
		case 3663: return ReadInputMap3663(dg, input_map_values, TEXTURE_ARGS);
		case 3666: return ReadInputMap3666(dg, input_map_values, TEXTURE_ARGS);
		case 3667: return ReadInputMap3667(dg, input_map_values, TEXTURE_ARGS);
		case 3668: return ReadInputMap3668(dg, input_map_values, TEXTURE_ARGS);
		case 3676: return ReadInputMap3676(dg, input_map_values, TEXTURE_ARGS);
		case 3677: return ReadInputMap3677(dg, input_map_values, TEXTURE_ARGS);
		case 3679: return ReadInputMap3679(dg, input_map_values, TEXTURE_ARGS);
		case 3680: return ReadInputMap3680(dg, input_map_values, TEXTURE_ARGS);
		case 3684: return ReadInputMap3684(dg, input_map_values, TEXTURE_ARGS);
		case 3689: return ReadInputMap3689(dg, input_map_values, TEXTURE_ARGS);
		case 3692: return ReadInputMap3692(dg, input_map_values, TEXTURE_ARGS);
		case 3693: return ReadInputMap3693(dg, input_map_values, TEXTURE_ARGS);
		case 3694: return ReadInputMap3694(dg, input_map_values, TEXTURE_ARGS);
		case 3698: return ReadInputMap3698(dg, input_map_values, TEXTURE_ARGS);
		case 3702: return ReadInputMap3702(dg, input_map_values, TEXTURE_ARGS);
		case 3715: return ReadInputMap3715(dg, input_map_values, TEXTURE_ARGS);
		case 3716: return ReadInputMap3716(dg, input_map_values, TEXTURE_ARGS);
		case 3718: return ReadInputMap3718(dg, input_map_values, TEXTURE_ARGS);
		case 3719: return ReadInputMap3719(dg, input_map_values, TEXTURE_ARGS);
		case 3720: return ReadInputMap3720(dg, input_map_values, TEXTURE_ARGS);
		case 3723: return ReadInputMap3723(dg, input_map_values, TEXTURE_ARGS);
		case 3732: return ReadInputMap3732(dg, input_map_values, TEXTURE_ARGS);
		case 3733: return ReadInputMap3733(dg, input_map_values, TEXTURE_ARGS);
		case 3735: return ReadInputMap3735(dg, input_map_values, TEXTURE_ARGS);
		case 3736: return ReadInputMap3736(dg, input_map_values, TEXTURE_ARGS);
		case 3737: return ReadInputMap3737(dg, input_map_values, TEXTURE_ARGS);
		case 3743: return ReadInputMap3743(dg, input_map_values, TEXTURE_ARGS);
		case 3744: return ReadInputMap3744(dg, input_map_values, TEXTURE_ARGS);
		case 3746: return ReadInputMap3746(dg, input_map_values, TEXTURE_ARGS);
		case 3747: return ReadInputMap3747(dg, input_map_values, TEXTURE_ARGS);
		case 3748: return ReadInputMap3748(dg, input_map_values, TEXTURE_ARGS);
		case 3754: return ReadInputMap3754(dg, input_map_values, TEXTURE_ARGS);
		case 3755: return ReadInputMap3755(dg, input_map_values, TEXTURE_ARGS);
		case 3757: return ReadInputMap3757(dg, input_map_values, TEXTURE_ARGS);
		case 3758: return ReadInputMap3758(dg, input_map_values, TEXTURE_ARGS);
		case 3759: return ReadInputMap3759(dg, input_map_values, TEXTURE_ARGS);
		case 3771: return ReadInputMap3771(dg, input_map_values, TEXTURE_ARGS);
		case 3772: return ReadInputMap3772(dg, input_map_values, TEXTURE_ARGS);
		case 3774: return ReadInputMap3774(dg, input_map_values, TEXTURE_ARGS);
		case 3775: return ReadInputMap3775(dg, input_map_values, TEXTURE_ARGS);
		case 3776: return ReadInputMap3776(dg, input_map_values, TEXTURE_ARGS);
		case 3782: return ReadInputMap3782(dg, input_map_values, TEXTURE_ARGS);
		case 3783: return ReadInputMap3783(dg, input_map_values, TEXTURE_ARGS);
		case 3785: return ReadInputMap3785(dg, input_map_values, TEXTURE_ARGS);
		case 3786: return ReadInputMap3786(dg, input_map_values, TEXTURE_ARGS);
		case 3787: return ReadInputMap3787(dg, input_map_values, TEXTURE_ARGS);
		case 3793: return ReadInputMap3793(dg, input_map_values, TEXTURE_ARGS);
		case 3794: return ReadInputMap3794(dg, input_map_values, TEXTURE_ARGS);
		case 3796: return ReadInputMap3796(dg, input_map_values, TEXTURE_ARGS);
		case 3797: return ReadInputMap3797(dg, input_map_values, TEXTURE_ARGS);
		case 3798: return ReadInputMap3798(dg, input_map_values, TEXTURE_ARGS);
		case 3804: return ReadInputMap3804(dg, input_map_values, TEXTURE_ARGS);
		case 3805: return ReadInputMap3805(dg, input_map_values, TEXTURE_ARGS);
		case 3807: return ReadInputMap3807(dg, input_map_values, TEXTURE_ARGS);
		case 3808: return ReadInputMap3808(dg, input_map_values, TEXTURE_ARGS);
		case 3809: return ReadInputMap3809(dg, input_map_values, TEXTURE_ARGS);
		case 3815: return ReadInputMap3815(dg, input_map_values, TEXTURE_ARGS);
		case 3816: return ReadInputMap3816(dg, input_map_values, TEXTURE_ARGS);
		case 3818: return ReadInputMap3818(dg, input_map_values, TEXTURE_ARGS);
		case 3819: return ReadInputMap3819(dg, input_map_values, TEXTURE_ARGS);
		case 3820: return ReadInputMap3820(dg, input_map_values, TEXTURE_ARGS);
		case 3826: return ReadInputMap3826(dg, input_map_values, TEXTURE_ARGS);
		case 3827: return ReadInputMap3827(dg, input_map_values, TEXTURE_ARGS);
		case 3829: return ReadInputMap3829(dg, input_map_values, TEXTURE_ARGS);
		case 3830: return ReadInputMap3830(dg, input_map_values, TEXTURE_ARGS);
		case 3831: return ReadInputMap3831(dg, input_map_values, TEXTURE_ARGS);
		case 3838: return ReadInputMap3838(dg, input_map_values, TEXTURE_ARGS);
		case 3844: return ReadInputMap3844(dg, input_map_values, TEXTURE_ARGS);
		case 3845: return ReadInputMap3845(dg, input_map_values, TEXTURE_ARGS);
		case 3847: return ReadInputMap3847(dg, input_map_values, TEXTURE_ARGS);
		case 3848: return ReadInputMap3848(dg, input_map_values, TEXTURE_ARGS);
		case 3855: return ReadInputMap3855(dg, input_map_values, TEXTURE_ARGS);
		case 3858: return ReadInputMap3858(dg, input_map_values, TEXTURE_ARGS);
		case 3859: return ReadInputMap3859(dg, input_map_values, TEXTURE_ARGS);
		case 3860: return ReadInputMap3860(dg, input_map_values, TEXTURE_ARGS);
		case 3879: return ReadInputMap3879(dg, input_map_values, TEXTURE_ARGS);
		case 3880: return ReadInputMap3880(dg, input_map_values, TEXTURE_ARGS);
		case 3882: return ReadInputMap3882(dg, input_map_values, TEXTURE_ARGS);
		case 3883: return ReadInputMap3883(dg, input_map_values, TEXTURE_ARGS);
		case 3887: return ReadInputMap3887(dg, input_map_values, TEXTURE_ARGS);
		case 3893: return ReadInputMap3893(dg, input_map_values, TEXTURE_ARGS);
		case 3899: return ReadInputMap3899(dg, input_map_values, TEXTURE_ARGS);
		case 3905: return ReadInputMap3905(dg, input_map_values, TEXTURE_ARGS);
		case 3906: return ReadInputMap3906(dg, input_map_values, TEXTURE_ARGS);
		case 3908: return ReadInputMap3908(dg, input_map_values, TEXTURE_ARGS);
		case 3909: return ReadInputMap3909(dg, input_map_values, TEXTURE_ARGS);
		case 3910: return ReadInputMap3910(dg, input_map_values, TEXTURE_ARGS);
		case 3911: return ReadInputMap3911(dg, input_map_values, TEXTURE_ARGS);
		case 3912: return ReadInputMap3912(dg, input_map_values, TEXTURE_ARGS);
		case 3913: return ReadInputMap3913(dg, input_map_values, TEXTURE_ARGS);
		case 3916: return ReadInputMap3916(dg, input_map_values, TEXTURE_ARGS);
		case 3925: return ReadInputMap3925(dg, input_map_values, TEXTURE_ARGS);
		case 3928: return ReadInputMap3928(dg, input_map_values, TEXTURE_ARGS);
		case 3929: return ReadInputMap3929(dg, input_map_values, TEXTURE_ARGS);
		case 3930: return ReadInputMap3930(dg, input_map_values, TEXTURE_ARGS);
		case 3931: return ReadInputMap3931(dg, input_map_values, TEXTURE_ARGS);
		case 3941: return ReadInputMap3941(dg, input_map_values, TEXTURE_ARGS);
		case 3942: return ReadInputMap3942(dg, input_map_values, TEXTURE_ARGS);
		case 3943: return ReadInputMap3943(dg, input_map_values, TEXTURE_ARGS);
		case 3944: return ReadInputMap3944(dg, input_map_values, TEXTURE_ARGS);
		case 3953: return ReadInputMap3953(dg, input_map_values, TEXTURE_ARGS);
		case 3954: return ReadInputMap3954(dg, input_map_values, TEXTURE_ARGS);
		case 3956: return ReadInputMap3956(dg, input_map_values, TEXTURE_ARGS);
		case 3957: return ReadInputMap3957(dg, input_map_values, TEXTURE_ARGS);
		case 3958: return ReadInputMap3958(dg, input_map_values, TEXTURE_ARGS);
		case 3964: return ReadInputMap3964(dg, input_map_values, TEXTURE_ARGS);
		case 3965: return ReadInputMap3965(dg, input_map_values, TEXTURE_ARGS);
		case 3967: return ReadInputMap3967(dg, input_map_values, TEXTURE_ARGS);
		case 3968: return ReadInputMap3968(dg, input_map_values, TEXTURE_ARGS);
		case 3969: return ReadInputMap3969(dg, input_map_values, TEXTURE_ARGS);
		case 3975: return ReadInputMap3975(dg, input_map_values, TEXTURE_ARGS);
		case 3976: return ReadInputMap3976(dg, input_map_values, TEXTURE_ARGS);
		case 3978: return ReadInputMap3978(dg, input_map_values, TEXTURE_ARGS);
		case 3979: return ReadInputMap3979(dg, input_map_values, TEXTURE_ARGS);
		case 3983: return ReadInputMap3983(dg, input_map_values, TEXTURE_ARGS);
		case 3989: return ReadInputMap3989(dg, input_map_values, TEXTURE_ARGS);
		case 4001: return ReadInputMap4001(dg, input_map_values, TEXTURE_ARGS);
		case 4004: return ReadInputMap4004(dg, input_map_values, TEXTURE_ARGS);
		case 4005: return ReadInputMap4005(dg, input_map_values, TEXTURE_ARGS);
		case 4006: return ReadInputMap4006(dg, input_map_values, TEXTURE_ARGS);
		case 4010: return ReadInputMap4010(dg, input_map_values, TEXTURE_ARGS);
		case 4011: return ReadInputMap4011(dg, input_map_values, TEXTURE_ARGS);
		case 4012: return ReadInputMap4012(dg, input_map_values, TEXTURE_ARGS);
		case 4013: return ReadInputMap4013(dg, input_map_values, TEXTURE_ARGS);
		case 4019: return ReadInputMap4019(dg, input_map_values, TEXTURE_ARGS);
		case 4022: return ReadInputMap4022(dg, input_map_values, TEXTURE_ARGS);
		case 4023: return ReadInputMap4023(dg, input_map_values, TEXTURE_ARGS);
		case 4024: return ReadInputMap4024(dg, input_map_values, TEXTURE_ARGS);
		case 4025: return ReadInputMap4025(dg, input_map_values, TEXTURE_ARGS);
		case 4026: return ReadInputMap4026(dg, input_map_values, TEXTURE_ARGS);
		case 4027: return ReadInputMap4027(dg, input_map_values, TEXTURE_ARGS);
		case 4028: return ReadInputMap4028(dg, input_map_values, TEXTURE_ARGS);
		case 4034: return ReadInputMap4034(dg, input_map_values, TEXTURE_ARGS);
		case 4040: return ReadInputMap4040(dg, input_map_values, TEXTURE_ARGS);
		case 4043: return ReadInputMap4043(dg, input_map_values, TEXTURE_ARGS);
		case 4049: return ReadInputMap4049(dg, input_map_values, TEXTURE_ARGS);
		case 4050: return ReadInputMap4050(dg, input_map_values, TEXTURE_ARGS);
		case 4052: return ReadInputMap4052(dg, input_map_values, TEXTURE_ARGS);
		case 4053: return ReadInputMap4053(dg, input_map_values, TEXTURE_ARGS);
		case 4057: return ReadInputMap4057(dg, input_map_values, TEXTURE_ARGS);
		case 4058: return ReadInputMap4058(dg, input_map_values, TEXTURE_ARGS);
		case 4059: return ReadInputMap4059(dg, input_map_values, TEXTURE_ARGS);
		case 4068: return ReadInputMap4068(dg, input_map_values, TEXTURE_ARGS);
		case 4071: return ReadInputMap4071(dg, input_map_values, TEXTURE_ARGS);
		case 4072: return ReadInputMap4072(dg, input_map_values, TEXTURE_ARGS);
		case 4073: return ReadInputMap4073(dg, input_map_values, TEXTURE_ARGS);
		case 4077: return ReadInputMap4077(dg, input_map_values, TEXTURE_ARGS);
		case 4078: return ReadInputMap4078(dg, input_map_values, TEXTURE_ARGS);
		case 4079: return ReadInputMap4079(dg, input_map_values, TEXTURE_ARGS);
		case 4080: return ReadInputMap4080(dg, input_map_values, TEXTURE_ARGS);
		case 4083: return ReadInputMap4083(dg, input_map_values, TEXTURE_ARGS);
		case 4089: return ReadInputMap4089(dg, input_map_values, TEXTURE_ARGS);
		case 4090: return ReadInputMap4090(dg, input_map_values, TEXTURE_ARGS);
		case 4092: return ReadInputMap4092(dg, input_map_values, TEXTURE_ARGS);
		case 4093: return ReadInputMap4093(dg, input_map_values, TEXTURE_ARGS);
		case 4094: return ReadInputMap4094(dg, input_map_values, TEXTURE_ARGS);
		case 4095: return ReadInputMap4095(dg, input_map_values, TEXTURE_ARGS);
		case 4096: return ReadInputMap4096(dg, input_map_values, TEXTURE_ARGS);
		case 4097: return ReadInputMap4097(dg, input_map_values, TEXTURE_ARGS);
		case 4100: return ReadInputMap4100(dg, input_map_values, TEXTURE_ARGS);
		case 4106: return ReadInputMap4106(dg, input_map_values, TEXTURE_ARGS);
		case 4107: return ReadInputMap4107(dg, input_map_values, TEXTURE_ARGS);
		case 4109: return ReadInputMap4109(dg, input_map_values, TEXTURE_ARGS);
		case 4110: return ReadInputMap4110(dg, input_map_values, TEXTURE_ARGS);
		case 4111: return ReadInputMap4111(dg, input_map_values, TEXTURE_ARGS);
		case 4112: return ReadInputMap4112(dg, input_map_values, TEXTURE_ARGS);
		case 4113: return ReadInputMap4113(dg, input_map_values, TEXTURE_ARGS);
		case 4117: return ReadInputMap4117(dg, input_map_values, TEXTURE_ARGS);
		case 4123: return ReadInputMap4123(dg, input_map_values, TEXTURE_ARGS);
		case 4126: return ReadInputMap4126(dg, input_map_values, TEXTURE_ARGS);
		case 4127: return ReadInputMap4127(dg, input_map_values, TEXTURE_ARGS);
		case 4128: return ReadInputMap4128(dg, input_map_values, TEXTURE_ARGS);
		case 4132: return ReadInputMap4132(dg, input_map_values, TEXTURE_ARGS);
		case 4135: return ReadInputMap4135(dg, input_map_values, TEXTURE_ARGS);
		case 4143: return ReadInputMap4143(dg, input_map_values, TEXTURE_ARGS);
		case 4144: return ReadInputMap4144(dg, input_map_values, TEXTURE_ARGS);
		case 4146: return ReadInputMap4146(dg, input_map_values, TEXTURE_ARGS);
		case 4147: return ReadInputMap4147(dg, input_map_values, TEXTURE_ARGS);
		case 4159: return ReadInputMap4159(dg, input_map_values, TEXTURE_ARGS);
		case 4165: return ReadInputMap4165(dg, input_map_values, TEXTURE_ARGS);
		case 4171: return ReadInputMap4171(dg, input_map_values, TEXTURE_ARGS);
		case 4172: return ReadInputMap4172(dg, input_map_values, TEXTURE_ARGS);
		case 4174: return ReadInputMap4174(dg, input_map_values, TEXTURE_ARGS);
		case 4175: return ReadInputMap4175(dg, input_map_values, TEXTURE_ARGS);
		case 4179: return ReadInputMap4179(dg, input_map_values, TEXTURE_ARGS);
		case 4180: return ReadInputMap4180(dg, input_map_values, TEXTURE_ARGS);
		case 4181: return ReadInputMap4181(dg, input_map_values, TEXTURE_ARGS);
		case 4185: return ReadInputMap4185(dg, input_map_values, TEXTURE_ARGS);
		case 4193: return ReadInputMap4193(dg, input_map_values, TEXTURE_ARGS);
		case 4194: return ReadInputMap4194(dg, input_map_values, TEXTURE_ARGS);
		case 4199: return ReadInputMap4199(dg, input_map_values, TEXTURE_ARGS);
		case 4205: return ReadInputMap4205(dg, input_map_values, TEXTURE_ARGS);
		case 4219: return ReadInputMap4219(dg, input_map_values, TEXTURE_ARGS);
		case 4222: return ReadInputMap4222(dg, input_map_values, TEXTURE_ARGS);
		case 4223: return ReadInputMap4223(dg, input_map_values, TEXTURE_ARGS);
		case 4224: return ReadInputMap4224(dg, input_map_values, TEXTURE_ARGS);
		case 4230: return ReadInputMap4230(dg, input_map_values, TEXTURE_ARGS);
		case 4233: return ReadInputMap4233(dg, input_map_values, TEXTURE_ARGS);
		case 4234: return ReadInputMap4234(dg, input_map_values, TEXTURE_ARGS);
		case 4235: return ReadInputMap4235(dg, input_map_values, TEXTURE_ARGS);
		case 4236: return ReadInputMap4236(dg, input_map_values, TEXTURE_ARGS);
		case 4242: return ReadInputMap4242(dg, input_map_values, TEXTURE_ARGS);
		case 4245: return ReadInputMap4245(dg, input_map_values, TEXTURE_ARGS);
		case 4246: return ReadInputMap4246(dg, input_map_values, TEXTURE_ARGS);
		case 4247: return ReadInputMap4247(dg, input_map_values, TEXTURE_ARGS);
		case 4248: return ReadInputMap4248(dg, input_map_values, TEXTURE_ARGS);
		case 4254: return ReadInputMap4254(dg, input_map_values, TEXTURE_ARGS);
		case 4257: return ReadInputMap4257(dg, input_map_values, TEXTURE_ARGS);
		case 4258: return ReadInputMap4258(dg, input_map_values, TEXTURE_ARGS);
		case 4259: return ReadInputMap4259(dg, input_map_values, TEXTURE_ARGS);
		case 4260: return ReadInputMap4260(dg, input_map_values, TEXTURE_ARGS);
		case 4266: return ReadInputMap4266(dg, input_map_values, TEXTURE_ARGS);
		case 4269: return ReadInputMap4269(dg, input_map_values, TEXTURE_ARGS);
		case 4270: return ReadInputMap4270(dg, input_map_values, TEXTURE_ARGS);
		case 4271: return ReadInputMap4271(dg, input_map_values, TEXTURE_ARGS);
		case 4279: return ReadInputMap4279(dg, input_map_values, TEXTURE_ARGS);
		case 4280: return ReadInputMap4280(dg, input_map_values, TEXTURE_ARGS);
		case 4282: return ReadInputMap4282(dg, input_map_values, TEXTURE_ARGS);
		case 4283: return ReadInputMap4283(dg, input_map_values, TEXTURE_ARGS);
		case 4287: return ReadInputMap4287(dg, input_map_values, TEXTURE_ARGS);
		case 4288: return ReadInputMap4288(dg, input_map_values, TEXTURE_ARGS);
		case 4289: return ReadInputMap4289(dg, input_map_values, TEXTURE_ARGS);
		case 4293: return ReadInputMap4293(dg, input_map_values, TEXTURE_ARGS);
		case 4294: return ReadInputMap4294(dg, input_map_values, TEXTURE_ARGS);
		case 4298: return ReadInputMap4298(dg, input_map_values, TEXTURE_ARGS);
		case 4304: return ReadInputMap4304(dg, input_map_values, TEXTURE_ARGS);
		case 4305: return ReadInputMap4305(dg, input_map_values, TEXTURE_ARGS);
		case 4307: return ReadInputMap4307(dg, input_map_values, TEXTURE_ARGS);
		case 4308: return ReadInputMap4308(dg, input_map_values, TEXTURE_ARGS);
		case 4309: return ReadInputMap4309(dg, input_map_values, TEXTURE_ARGS);
		case 4312: return ReadInputMap4312(dg, input_map_values, TEXTURE_ARGS);
		case 4318: return ReadInputMap4318(dg, input_map_values, TEXTURE_ARGS);
		case 4319: return ReadInputMap4319(dg, input_map_values, TEXTURE_ARGS);
		case 4324: return ReadInputMap4324(dg, input_map_values, TEXTURE_ARGS);
		case 4330: return ReadInputMap4330(dg, input_map_values, TEXTURE_ARGS);
		case 4341: return ReadInputMap4341(dg, input_map_values, TEXTURE_ARGS);
		case 4342: return ReadInputMap4342(dg, input_map_values, TEXTURE_ARGS);
		case 4344: return ReadInputMap4344(dg, input_map_values, TEXTURE_ARGS);
		case 4345: return ReadInputMap4345(dg, input_map_values, TEXTURE_ARGS);
		case 4349: return ReadInputMap4349(dg, input_map_values, TEXTURE_ARGS);
		case 4350: return ReadInputMap4350(dg, input_map_values, TEXTURE_ARGS);
		case 4351: return ReadInputMap4351(dg, input_map_values, TEXTURE_ARGS);
		case 4355: return ReadInputMap4355(dg, input_map_values, TEXTURE_ARGS);
		case 4361: return ReadInputMap4361(dg, input_map_values, TEXTURE_ARGS);
		case 4362: return ReadInputMap4362(dg, input_map_values, TEXTURE_ARGS);
		case 4364: return ReadInputMap4364(dg, input_map_values, TEXTURE_ARGS);
		case 4365: return ReadInputMap4365(dg, input_map_values, TEXTURE_ARGS);
		case 4369: return ReadInputMap4369(dg, input_map_values, TEXTURE_ARGS);
		case 4370: return ReadInputMap4370(dg, input_map_values, TEXTURE_ARGS);
		case 4374: return ReadInputMap4374(dg, input_map_values, TEXTURE_ARGS);
		case 4380: return ReadInputMap4380(dg, input_map_values, TEXTURE_ARGS);
		case 4386: return ReadInputMap4386(dg, input_map_values, TEXTURE_ARGS);
		case 4389: return ReadInputMap4389(dg, input_map_values, TEXTURE_ARGS);
		case 4397: return ReadInputMap4397(dg, input_map_values, TEXTURE_ARGS);
		case 4398: return ReadInputMap4398(dg, input_map_values, TEXTURE_ARGS);
		case 4400: return ReadInputMap4400(dg, input_map_values, TEXTURE_ARGS);
		case 4401: return ReadInputMap4401(dg, input_map_values, TEXTURE_ARGS);
		case 4402: return ReadInputMap4402(dg, input_map_values, TEXTURE_ARGS);
		case 4408: return ReadInputMap4408(dg, input_map_values, TEXTURE_ARGS);
		case 4409: return ReadInputMap4409(dg, input_map_values, TEXTURE_ARGS);
		case 4411: return ReadInputMap4411(dg, input_map_values, TEXTURE_ARGS);
		case 4412: return ReadInputMap4412(dg, input_map_values, TEXTURE_ARGS);
		case 4413: return ReadInputMap4413(dg, input_map_values, TEXTURE_ARGS);
		case 4425: return ReadInputMap4425(dg, input_map_values, TEXTURE_ARGS);
		case 4426: return ReadInputMap4426(dg, input_map_values, TEXTURE_ARGS);
		case 4428: return ReadInputMap4428(dg, input_map_values, TEXTURE_ARGS);
		case 4429: return ReadInputMap4429(dg, input_map_values, TEXTURE_ARGS);
		case 4430: return ReadInputMap4430(dg, input_map_values, TEXTURE_ARGS);
		case 4436: return ReadInputMap4436(dg, input_map_values, TEXTURE_ARGS);
		case 4437: return ReadInputMap4437(dg, input_map_values, TEXTURE_ARGS);
		case 4439: return ReadInputMap4439(dg, input_map_values, TEXTURE_ARGS);
		case 4440: return ReadInputMap4440(dg, input_map_values, TEXTURE_ARGS);
		case 4441: return ReadInputMap4441(dg, input_map_values, TEXTURE_ARGS);
		case 4447: return ReadInputMap4447(dg, input_map_values, TEXTURE_ARGS);
		case 4448: return ReadInputMap4448(dg, input_map_values, TEXTURE_ARGS);
		case 4450: return ReadInputMap4450(dg, input_map_values, TEXTURE_ARGS);
		case 4451: return ReadInputMap4451(dg, input_map_values, TEXTURE_ARGS);
		case 4452: return ReadInputMap4452(dg, input_map_values, TEXTURE_ARGS);
		case 4458: return ReadInputMap4458(dg, input_map_values, TEXTURE_ARGS);
		case 4459: return ReadInputMap4459(dg, input_map_values, TEXTURE_ARGS);
		case 4461: return ReadInputMap4461(dg, input_map_values, TEXTURE_ARGS);
		case 4462: return ReadInputMap4462(dg, input_map_values, TEXTURE_ARGS);
		case 4463: return ReadInputMap4463(dg, input_map_values, TEXTURE_ARGS);
		case 4469: return ReadInputMap4469(dg, input_map_values, TEXTURE_ARGS);
		case 4470: return ReadInputMap4470(dg, input_map_values, TEXTURE_ARGS);
		case 4472: return ReadInputMap4472(dg, input_map_values, TEXTURE_ARGS);
		case 4473: return ReadInputMap4473(dg, input_map_values, TEXTURE_ARGS);
		case 4474: return ReadInputMap4474(dg, input_map_values, TEXTURE_ARGS);
		case 4480: return ReadInputMap4480(dg, input_map_values, TEXTURE_ARGS);
		case 4481: return ReadInputMap4481(dg, input_map_values, TEXTURE_ARGS);
		case 4483: return ReadInputMap4483(dg, input_map_values, TEXTURE_ARGS);
		case 4484: return ReadInputMap4484(dg, input_map_values, TEXTURE_ARGS);
		case 4485: return ReadInputMap4485(dg, input_map_values, TEXTURE_ARGS);
		case 4492: return ReadInputMap4492(dg, input_map_values, TEXTURE_ARGS);
		case 4493: return ReadInputMap4493(dg, input_map_values, TEXTURE_ARGS);
		case 4495: return ReadInputMap4495(dg, input_map_values, TEXTURE_ARGS);
		case 4496: return ReadInputMap4496(dg, input_map_values, TEXTURE_ARGS);
		case 4497: return ReadInputMap4497(dg, input_map_values, TEXTURE_ARGS);
		case 4504: return ReadInputMap4504(dg, input_map_values, TEXTURE_ARGS);
		case 4505: return ReadInputMap4505(dg, input_map_values, TEXTURE_ARGS);
		case 4507: return ReadInputMap4507(dg, input_map_values, TEXTURE_ARGS);
		case 4508: return ReadInputMap4508(dg, input_map_values, TEXTURE_ARGS);
		case 4509: return ReadInputMap4509(dg, input_map_values, TEXTURE_ARGS);
		case 4521: return ReadInputMap4521(dg, input_map_values, TEXTURE_ARGS);
		case 4522: return ReadInputMap4522(dg, input_map_values, TEXTURE_ARGS);
		case 4524: return ReadInputMap4524(dg, input_map_values, TEXTURE_ARGS);
		case 4525: return ReadInputMap4525(dg, input_map_values, TEXTURE_ARGS);
		case 4526: return ReadInputMap4526(dg, input_map_values, TEXTURE_ARGS);
		case 4527: return ReadInputMap4527(dg, input_map_values, TEXTURE_ARGS);
		case 4528: return ReadInputMap4528(dg, input_map_values, TEXTURE_ARGS);
		case 4529: return ReadInputMap4529(dg, input_map_values, TEXTURE_ARGS);
		case 4535: return ReadInputMap4535(dg, input_map_values, TEXTURE_ARGS);
		case 4536: return ReadInputMap4536(dg, input_map_values, TEXTURE_ARGS);
		case 4538: return ReadInputMap4538(dg, input_map_values, TEXTURE_ARGS);
		case 4539: return ReadInputMap4539(dg, input_map_values, TEXTURE_ARGS);
		case 4540: return ReadInputMap4540(dg, input_map_values, TEXTURE_ARGS);
		case 4541: return ReadInputMap4541(dg, input_map_values, TEXTURE_ARGS);
		case 4542: return ReadInputMap4542(dg, input_map_values, TEXTURE_ARGS);
		case 4543: return ReadInputMap4543(dg, input_map_values, TEXTURE_ARGS);
		case 4549: return ReadInputMap4549(dg, input_map_values, TEXTURE_ARGS);
		case 4550: return ReadInputMap4550(dg, input_map_values, TEXTURE_ARGS);
		case 4552: return ReadInputMap4552(dg, input_map_values, TEXTURE_ARGS);
		case 4553: return ReadInputMap4553(dg, input_map_values, TEXTURE_ARGS);
		case 4554: return ReadInputMap4554(dg, input_map_values, TEXTURE_ARGS);
		case 4555: return ReadInputMap4555(dg, input_map_values, TEXTURE_ARGS);
		case 4556: return ReadInputMap4556(dg, input_map_values, TEXTURE_ARGS);
		case 4557: return ReadInputMap4557(dg, input_map_values, TEXTURE_ARGS);
		case 4563: return ReadInputMap4563(dg, input_map_values, TEXTURE_ARGS);
		case 4564: return ReadInputMap4564(dg, input_map_values, TEXTURE_ARGS);
		case 4566: return ReadInputMap4566(dg, input_map_values, TEXTURE_ARGS);
		case 4567: return ReadInputMap4567(dg, input_map_values, TEXTURE_ARGS);
		case 4568: return ReadInputMap4568(dg, input_map_values, TEXTURE_ARGS);
		case 4569: return ReadInputMap4569(dg, input_map_values, TEXTURE_ARGS);
		case 4570: return ReadInputMap4570(dg, input_map_values, TEXTURE_ARGS);
		case 4571: return ReadInputMap4571(dg, input_map_values, TEXTURE_ARGS);
		case 4577: return ReadInputMap4577(dg, input_map_values, TEXTURE_ARGS);
		case 4578: return ReadInputMap4578(dg, input_map_values, TEXTURE_ARGS);
		case 4580: return ReadInputMap4580(dg, input_map_values, TEXTURE_ARGS);
		case 4581: return ReadInputMap4581(dg, input_map_values, TEXTURE_ARGS);
		case 4582: return ReadInputMap4582(dg, input_map_values, TEXTURE_ARGS);
		case 4583: return ReadInputMap4583(dg, input_map_values, TEXTURE_ARGS);
		case 4584: return ReadInputMap4584(dg, input_map_values, TEXTURE_ARGS);
		case 4585: return ReadInputMap4585(dg, input_map_values, TEXTURE_ARGS);
		case 4591: return ReadInputMap4591(dg, input_map_values, TEXTURE_ARGS);
		case 4592: return ReadInputMap4592(dg, input_map_values, TEXTURE_ARGS);
		case 4594: return ReadInputMap4594(dg, input_map_values, TEXTURE_ARGS);
		case 4595: return ReadInputMap4595(dg, input_map_values, TEXTURE_ARGS);
		case 4596: return ReadInputMap4596(dg, input_map_values, TEXTURE_ARGS);
		case 4597: return ReadInputMap4597(dg, input_map_values, TEXTURE_ARGS);
		case 4598: return ReadInputMap4598(dg, input_map_values, TEXTURE_ARGS);
		case 4599: return ReadInputMap4599(dg, input_map_values, TEXTURE_ARGS);
		case 4602: return ReadInputMap4602(dg, input_map_values, TEXTURE_ARGS);
		case 4603: return ReadInputMap4603(dg, input_map_values, TEXTURE_ARGS);
		case 4604: return ReadInputMap4604(dg, input_map_values, TEXTURE_ARGS);
		case 4617: return ReadInputMap4617(dg, input_map_values, TEXTURE_ARGS);
		case 4618: return ReadInputMap4618(dg, input_map_values, TEXTURE_ARGS);
		case 4620: return ReadInputMap4620(dg, input_map_values, TEXTURE_ARGS);
		case 4621: return ReadInputMap4621(dg, input_map_values, TEXTURE_ARGS);
		case 4622: return ReadInputMap4622(dg, input_map_values, TEXTURE_ARGS);
		case 4632: return ReadInputMap4632(dg, input_map_values, TEXTURE_ARGS);
		case 4635: return ReadInputMap4635(dg, input_map_values, TEXTURE_ARGS);
		case 4636: return ReadInputMap4636(dg, input_map_values, TEXTURE_ARGS);
		case 4637: return ReadInputMap4637(dg, input_map_values, TEXTURE_ARGS);
		case 4641: return ReadInputMap4641(dg, input_map_values, TEXTURE_ARGS);
		case 4647: return ReadInputMap4647(dg, input_map_values, TEXTURE_ARGS);
		case 4656: return ReadInputMap4656(dg, input_map_values, TEXTURE_ARGS);
		case 4657: return ReadInputMap4657(dg, input_map_values, TEXTURE_ARGS);
		case 4659: return ReadInputMap4659(dg, input_map_values, TEXTURE_ARGS);
		case 4660: return ReadInputMap4660(dg, input_map_values, TEXTURE_ARGS);
		case 4661: return ReadInputMap4661(dg, input_map_values, TEXTURE_ARGS);
		case 4664: return ReadInputMap4664(dg, input_map_values, TEXTURE_ARGS);
		case 4670: return ReadInputMap4670(dg, input_map_values, TEXTURE_ARGS);
		case 4673: return ReadInputMap4673(dg, input_map_values, TEXTURE_ARGS);
		case 4674: return ReadInputMap4674(dg, input_map_values, TEXTURE_ARGS);
		case 4675: return ReadInputMap4675(dg, input_map_values, TEXTURE_ARGS);
		case 4682: return ReadInputMap4682(dg, input_map_values, TEXTURE_ARGS);
		case 4683: return ReadInputMap4683(dg, input_map_values, TEXTURE_ARGS);
		case 4685: return ReadInputMap4685(dg, input_map_values, TEXTURE_ARGS);
		case 4686: return ReadInputMap4686(dg, input_map_values, TEXTURE_ARGS);
		case 4687: return ReadInputMap4687(dg, input_map_values, TEXTURE_ARGS);
		case 4690: return ReadInputMap4690(dg, input_map_values, TEXTURE_ARGS);
		case 4691: return ReadInputMap4691(dg, input_map_values, TEXTURE_ARGS);
		case 4692: return ReadInputMap4692(dg, input_map_values, TEXTURE_ARGS);
		case 4696: return ReadInputMap4696(dg, input_map_values, TEXTURE_ARGS);
		case 4702: return ReadInputMap4702(dg, input_map_values, TEXTURE_ARGS);
		case 4703: return ReadInputMap4703(dg, input_map_values, TEXTURE_ARGS);
		case 4705: return ReadInputMap4705(dg, input_map_values, TEXTURE_ARGS);
		case 4706: return ReadInputMap4706(dg, input_map_values, TEXTURE_ARGS);
		case 4707: return ReadInputMap4707(dg, input_map_values, TEXTURE_ARGS);
		case 4708: return ReadInputMap4708(dg, input_map_values, TEXTURE_ARGS);
		case 4709: return ReadInputMap4709(dg, input_map_values, TEXTURE_ARGS);
		case 4716: return ReadInputMap4716(dg, input_map_values, TEXTURE_ARGS);
		case 4717: return ReadInputMap4717(dg, input_map_values, TEXTURE_ARGS);
		case 4719: return ReadInputMap4719(dg, input_map_values, TEXTURE_ARGS);
		case 4720: return ReadInputMap4720(dg, input_map_values, TEXTURE_ARGS);
		case 4721: return ReadInputMap4721(dg, input_map_values, TEXTURE_ARGS);
		case 4724: return ReadInputMap4724(dg, input_map_values, TEXTURE_ARGS);
		case 4727: return ReadInputMap4727(dg, input_map_values, TEXTURE_ARGS);
		case 4733: return ReadInputMap4733(dg, input_map_values, TEXTURE_ARGS);
		case 4734: return ReadInputMap4734(dg, input_map_values, TEXTURE_ARGS);
		case 4736: return ReadInputMap4736(dg, input_map_values, TEXTURE_ARGS);
		case 4737: return ReadInputMap4737(dg, input_map_values, TEXTURE_ARGS);
		case 4738: return ReadInputMap4738(dg, input_map_values, TEXTURE_ARGS);
		case 4739: return ReadInputMap4739(dg, input_map_values, TEXTURE_ARGS);
		case 4740: return ReadInputMap4740(dg, input_map_values, TEXTURE_ARGS);
		case 4749: return ReadInputMap4749(dg, input_map_values, TEXTURE_ARGS);
		case 4761: return ReadInputMap4761(dg, input_map_values, TEXTURE_ARGS);
		case 4764: return ReadInputMap4764(dg, input_map_values, TEXTURE_ARGS);
		case 4765: return ReadInputMap4765(dg, input_map_values, TEXTURE_ARGS);
		case 4766: return ReadInputMap4766(dg, input_map_values, TEXTURE_ARGS);
		case 4773: return ReadInputMap4773(dg, input_map_values, TEXTURE_ARGS);
		case 4774: return ReadInputMap4774(dg, input_map_values, TEXTURE_ARGS);
		case 4776: return ReadInputMap4776(dg, input_map_values, TEXTURE_ARGS);
		case 4777: return ReadInputMap4777(dg, input_map_values, TEXTURE_ARGS);
		case 4778: return ReadInputMap4778(dg, input_map_values, TEXTURE_ARGS);
		case 4781: return ReadInputMap4781(dg, input_map_values, TEXTURE_ARGS);
		case 4784: return ReadInputMap4784(dg, input_map_values, TEXTURE_ARGS);
		case 4790: return ReadInputMap4790(dg, input_map_values, TEXTURE_ARGS);
		case 4793: return ReadInputMap4793(dg, input_map_values, TEXTURE_ARGS);
		case 4794: return ReadInputMap4794(dg, input_map_values, TEXTURE_ARGS);
		case 4795: return ReadInputMap4795(dg, input_map_values, TEXTURE_ARGS);
		case 4799: return ReadInputMap4799(dg, input_map_values, TEXTURE_ARGS);
		case 4800: return ReadInputMap4800(dg, input_map_values, TEXTURE_ARGS);
		case 4801: return ReadInputMap4801(dg, input_map_values, TEXTURE_ARGS);
		case 4808: return ReadInputMap4808(dg, input_map_values, TEXTURE_ARGS);
		case 4809: return ReadInputMap4809(dg, input_map_values, TEXTURE_ARGS);
		case 4811: return ReadInputMap4811(dg, input_map_values, TEXTURE_ARGS);
		case 4812: return ReadInputMap4812(dg, input_map_values, TEXTURE_ARGS);
		case 4813: return ReadInputMap4813(dg, input_map_values, TEXTURE_ARGS);
		case 4821: return ReadInputMap4821(dg, input_map_values, TEXTURE_ARGS);
		case 4822: return ReadInputMap4822(dg, input_map_values, TEXTURE_ARGS);
		case 4824: return ReadInputMap4824(dg, input_map_values, TEXTURE_ARGS);
		case 4825: return ReadInputMap4825(dg, input_map_values, TEXTURE_ARGS);
		case 4829: return ReadInputMap4829(dg, input_map_values, TEXTURE_ARGS);
		case 4835: return ReadInputMap4835(dg, input_map_values, TEXTURE_ARGS);
		case 4838: return ReadInputMap4838(dg, input_map_values, TEXTURE_ARGS);
		case 4839: return ReadInputMap4839(dg, input_map_values, TEXTURE_ARGS);
		case 4840: return ReadInputMap4840(dg, input_map_values, TEXTURE_ARGS);
		case 4844: return ReadInputMap4844(dg, input_map_values, TEXTURE_ARGS);
		case 4850: return ReadInputMap4850(dg, input_map_values, TEXTURE_ARGS);
		case 4851: return ReadInputMap4851(dg, input_map_values, TEXTURE_ARGS);
		case 4862: return ReadInputMap4862(dg, input_map_values, TEXTURE_ARGS);
		case 4863: return ReadInputMap4863(dg, input_map_values, TEXTURE_ARGS);
		case 4865: return ReadInputMap4865(dg, input_map_values, TEXTURE_ARGS);
		case 4866: return ReadInputMap4866(dg, input_map_values, TEXTURE_ARGS);
		case 4870: return ReadInputMap4870(dg, input_map_values, TEXTURE_ARGS);
		case 4882: return ReadInputMap4882(dg, input_map_values, TEXTURE_ARGS);
		case 4883: return ReadInputMap4883(dg, input_map_values, TEXTURE_ARGS);
		case 4887: return ReadInputMap4887(dg, input_map_values, TEXTURE_ARGS);
		case 4890: return ReadInputMap4890(dg, input_map_values, TEXTURE_ARGS);
		case 4896: return ReadInputMap4896(dg, input_map_values, TEXTURE_ARGS);
		case 4897: return ReadInputMap4897(dg, input_map_values, TEXTURE_ARGS);
		case 4899: return ReadInputMap4899(dg, input_map_values, TEXTURE_ARGS);
		case 4900: return ReadInputMap4900(dg, input_map_values, TEXTURE_ARGS);
		case 4904: return ReadInputMap4904(dg, input_map_values, TEXTURE_ARGS);
		case 4907: return ReadInputMap4907(dg, input_map_values, TEXTURE_ARGS);
		case 4919: return ReadInputMap4919(dg, input_map_values, TEXTURE_ARGS);
		case 4922: return ReadInputMap4922(dg, input_map_values, TEXTURE_ARGS);
		case 4923: return ReadInputMap4923(dg, input_map_values, TEXTURE_ARGS);
		case 4924: return ReadInputMap4924(dg, input_map_values, TEXTURE_ARGS);
		case 4928: return ReadInputMap4928(dg, input_map_values, TEXTURE_ARGS);
		case 4934: return ReadInputMap4934(dg, input_map_values, TEXTURE_ARGS);
		case 4935: return ReadInputMap4935(dg, input_map_values, TEXTURE_ARGS);
		case 4937: return ReadInputMap4937(dg, input_map_values, TEXTURE_ARGS);
		case 4938: return ReadInputMap4938(dg, input_map_values, TEXTURE_ARGS);
		case 4942: return ReadInputMap4942(dg, input_map_values, TEXTURE_ARGS);
		case 4945: return ReadInputMap4945(dg, input_map_values, TEXTURE_ARGS);
		case 4951: return ReadInputMap4951(dg, input_map_values, TEXTURE_ARGS);
		case 4954: return ReadInputMap4954(dg, input_map_values, TEXTURE_ARGS);
		case 4955: return ReadInputMap4955(dg, input_map_values, TEXTURE_ARGS);
		case 4956: return ReadInputMap4956(dg, input_map_values, TEXTURE_ARGS);
		case 4965: return ReadInputMap4965(dg, input_map_values, TEXTURE_ARGS);
		case 4968: return ReadInputMap4968(dg, input_map_values, TEXTURE_ARGS);
		case 4969: return ReadInputMap4969(dg, input_map_values, TEXTURE_ARGS);
		case 4970: return ReadInputMap4970(dg, input_map_values, TEXTURE_ARGS);
		case 4974: return ReadInputMap4974(dg, input_map_values, TEXTURE_ARGS);
		case 4980: return ReadInputMap4980(dg, input_map_values, TEXTURE_ARGS);
		case 4981: return ReadInputMap4981(dg, input_map_values, TEXTURE_ARGS);
		case 4983: return ReadInputMap4983(dg, input_map_values, TEXTURE_ARGS);
		case 4984: return ReadInputMap4984(dg, input_map_values, TEXTURE_ARGS);
		case 4997: return ReadInputMap4997(dg, input_map_values, TEXTURE_ARGS);
		case 4998: return ReadInputMap4998(dg, input_map_values, TEXTURE_ARGS);
		case 5000: return ReadInputMap5000(dg, input_map_values, TEXTURE_ARGS);
		case 5001: return ReadInputMap5001(dg, input_map_values, TEXTURE_ARGS);
		case 5005: return ReadInputMap5005(dg, input_map_values, TEXTURE_ARGS);
		case 5011: return ReadInputMap5011(dg, input_map_values, TEXTURE_ARGS);
		case 5012: return ReadInputMap5012(dg, input_map_values, TEXTURE_ARGS);
		case 5014: return ReadInputMap5014(dg, input_map_values, TEXTURE_ARGS);
		case 5015: return ReadInputMap5015(dg, input_map_values, TEXTURE_ARGS);
		case 5019: return ReadInputMap5019(dg, input_map_values, TEXTURE_ARGS);
		case 5025: return ReadInputMap5025(dg, input_map_values, TEXTURE_ARGS);
		case 5028: return ReadInputMap5028(dg, input_map_values, TEXTURE_ARGS);
		case 5029: return ReadInputMap5029(dg, input_map_values, TEXTURE_ARGS);
		case 5030: return ReadInputMap5030(dg, input_map_values, TEXTURE_ARGS);
		case 5034: return ReadInputMap5034(dg, input_map_values, TEXTURE_ARGS);
		case 5040: return ReadInputMap5040(dg, input_map_values, TEXTURE_ARGS);
		case 5041: return ReadInputMap5041(dg, input_map_values, TEXTURE_ARGS);
		case 5043: return ReadInputMap5043(dg, input_map_values, TEXTURE_ARGS);
		case 5044: return ReadInputMap5044(dg, input_map_values, TEXTURE_ARGS);
		case 5048: return ReadInputMap5048(dg, input_map_values, TEXTURE_ARGS);
		case 5059: return ReadInputMap5059(dg, input_map_values, TEXTURE_ARGS);
		case 5060: return ReadInputMap5060(dg, input_map_values, TEXTURE_ARGS);
		case 5062: return ReadInputMap5062(dg, input_map_values, TEXTURE_ARGS);
		case 5063: return ReadInputMap5063(dg, input_map_values, TEXTURE_ARGS);
		case 5067: return ReadInputMap5067(dg, input_map_values, TEXTURE_ARGS);
		case 5073: return ReadInputMap5073(dg, input_map_values, TEXTURE_ARGS);
		case 5076: return ReadInputMap5076(dg, input_map_values, TEXTURE_ARGS);
		case 5077: return ReadInputMap5077(dg, input_map_values, TEXTURE_ARGS);
		case 5078: return ReadInputMap5078(dg, input_map_values, TEXTURE_ARGS);
		case 5082: return ReadInputMap5082(dg, input_map_values, TEXTURE_ARGS);
		case 5088: return ReadInputMap5088(dg, input_map_values, TEXTURE_ARGS);
		case 5089: return ReadInputMap5089(dg, input_map_values, TEXTURE_ARGS);
		case 5091: return ReadInputMap5091(dg, input_map_values, TEXTURE_ARGS);
		case 5092: return ReadInputMap5092(dg, input_map_values, TEXTURE_ARGS);
		case 5093: return ReadInputMap5093(dg, input_map_values, TEXTURE_ARGS);
		case 5094: return ReadInputMap5094(dg, input_map_values, TEXTURE_ARGS);
		case 5095: return ReadInputMap5095(dg, input_map_values, TEXTURE_ARGS);
		case 5102: return ReadInputMap5102(dg, input_map_values, TEXTURE_ARGS);
		case 5105: return ReadInputMap5105(dg, input_map_values, TEXTURE_ARGS);
		case 5106: return ReadInputMap5106(dg, input_map_values, TEXTURE_ARGS);
		case 5107: return ReadInputMap5107(dg, input_map_values, TEXTURE_ARGS);
		case 5111: return ReadInputMap5111(dg, input_map_values, TEXTURE_ARGS);
		case 5117: return ReadInputMap5117(dg, input_map_values, TEXTURE_ARGS);
		case 5118: return ReadInputMap5118(dg, input_map_values, TEXTURE_ARGS);
		case 5120: return ReadInputMap5120(dg, input_map_values, TEXTURE_ARGS);
		case 5121: return ReadInputMap5121(dg, input_map_values, TEXTURE_ARGS);
		case 5125: return ReadInputMap5125(dg, input_map_values, TEXTURE_ARGS);
		case 5131: return ReadInputMap5131(dg, input_map_values, TEXTURE_ARGS);
		case 5134: return ReadInputMap5134(dg, input_map_values, TEXTURE_ARGS);
		case 5135: return ReadInputMap5135(dg, input_map_values, TEXTURE_ARGS);
		case 5136: return ReadInputMap5136(dg, input_map_values, TEXTURE_ARGS);
		case 5140: return ReadInputMap5140(dg, input_map_values, TEXTURE_ARGS);
		case 5146: return ReadInputMap5146(dg, input_map_values, TEXTURE_ARGS);
		case 5147: return ReadInputMap5147(dg, input_map_values, TEXTURE_ARGS);
		case 5149: return ReadInputMap5149(dg, input_map_values, TEXTURE_ARGS);
		case 5150: return ReadInputMap5150(dg, input_map_values, TEXTURE_ARGS);
		case 5154: return ReadInputMap5154(dg, input_map_values, TEXTURE_ARGS);
		case 5158: return ReadInputMap5158(dg, input_map_values, TEXTURE_ARGS);
		case 5170: return ReadInputMap5170(dg, input_map_values, TEXTURE_ARGS);
		case 5173: return ReadInputMap5173(dg, input_map_values, TEXTURE_ARGS);
		case 5180: return ReadInputMap5180(dg, input_map_values, TEXTURE_ARGS);
		case 5183: return ReadInputMap5183(dg, input_map_values, TEXTURE_ARGS);
		case 5184: return ReadInputMap5184(dg, input_map_values, TEXTURE_ARGS);
		case 5185: return ReadInputMap5185(dg, input_map_values, TEXTURE_ARGS);
		case 5193: return ReadInputMap5193(dg, input_map_values, TEXTURE_ARGS);
		case 5194: return ReadInputMap5194(dg, input_map_values, TEXTURE_ARGS);
		case 5196: return ReadInputMap5196(dg, input_map_values, TEXTURE_ARGS);
		case 5197: return ReadInputMap5197(dg, input_map_values, TEXTURE_ARGS);
		case 5201: return ReadInputMap5201(dg, input_map_values, TEXTURE_ARGS);
		case 5207: return ReadInputMap5207(dg, input_map_values, TEXTURE_ARGS);
		case 5208: return ReadInputMap5208(dg, input_map_values, TEXTURE_ARGS);
		case 5210: return ReadInputMap5210(dg, input_map_values, TEXTURE_ARGS);
		case 5211: return ReadInputMap5211(dg, input_map_values, TEXTURE_ARGS);
		case 5215: return ReadInputMap5215(dg, input_map_values, TEXTURE_ARGS);
		case 5218: return ReadInputMap5218(dg, input_map_values, TEXTURE_ARGS);
		case 5221: return ReadInputMap5221(dg, input_map_values, TEXTURE_ARGS);
		case 5224: return ReadInputMap5224(dg, input_map_values, TEXTURE_ARGS);
		case 5247: return ReadInputMap5247(dg, input_map_values, TEXTURE_ARGS);
		case 5259: return ReadInputMap5259(dg, input_map_values, TEXTURE_ARGS);
		case 5270: return ReadInputMap5270(dg, input_map_values, TEXTURE_ARGS);
		case 5276: return ReadInputMap5276(dg, input_map_values, TEXTURE_ARGS);
		case 5279: return ReadInputMap5279(dg, input_map_values, TEXTURE_ARGS);
		case 5280: return ReadInputMap5280(dg, input_map_values, TEXTURE_ARGS);
		case 5281: return ReadInputMap5281(dg, input_map_values, TEXTURE_ARGS);
		case 5285: return ReadInputMap5285(dg, input_map_values, TEXTURE_ARGS);
		case 5291: return ReadInputMap5291(dg, input_map_values, TEXTURE_ARGS);
		case 5294: return ReadInputMap5294(dg, input_map_values, TEXTURE_ARGS);
		case 5295: return ReadInputMap5295(dg, input_map_values, TEXTURE_ARGS);
		case 5296: return ReadInputMap5296(dg, input_map_values, TEXTURE_ARGS);
		case 5300: return ReadInputMap5300(dg, input_map_values, TEXTURE_ARGS);
		case 5306: return ReadInputMap5306(dg, input_map_values, TEXTURE_ARGS);
		case 5309: return ReadInputMap5309(dg, input_map_values, TEXTURE_ARGS);
		case 5310: return ReadInputMap5310(dg, input_map_values, TEXTURE_ARGS);
		case 5311: return ReadInputMap5311(dg, input_map_values, TEXTURE_ARGS);
		case 5330: return ReadInputMap5330(dg, input_map_values, TEXTURE_ARGS);
		case 5336: return ReadInputMap5336(dg, input_map_values, TEXTURE_ARGS);
		case 5342: return ReadInputMap5342(dg, input_map_values, TEXTURE_ARGS);
		case 5348: return ReadInputMap5348(dg, input_map_values, TEXTURE_ARGS);
		case 5349: return ReadInputMap5349(dg, input_map_values, TEXTURE_ARGS);
		case 5351: return ReadInputMap5351(dg, input_map_values, TEXTURE_ARGS);
		case 5352: return ReadInputMap5352(dg, input_map_values, TEXTURE_ARGS);
		case 5358: return ReadInputMap5358(dg, input_map_values, TEXTURE_ARGS);
		case 5359: return ReadInputMap5359(dg, input_map_values, TEXTURE_ARGS);
		case 5360: return ReadInputMap5360(dg, input_map_values, TEXTURE_ARGS);
		case 5361: return ReadInputMap5361(dg, input_map_values, TEXTURE_ARGS);
		case 5362: return ReadInputMap5362(dg, input_map_values, TEXTURE_ARGS);
		case 5363: return ReadInputMap5363(dg, input_map_values, TEXTURE_ARGS);
		case 5367: return ReadInputMap5367(dg, input_map_values, TEXTURE_ARGS);
		case 5379: return ReadInputMap5379(dg, input_map_values, TEXTURE_ARGS);
		case 5385: return ReadInputMap5385(dg, input_map_values, TEXTURE_ARGS);
		case 5386: return ReadInputMap5386(dg, input_map_values, TEXTURE_ARGS);
		case 5388: return ReadInputMap5388(dg, input_map_values, TEXTURE_ARGS);
		case 5389: return ReadInputMap5389(dg, input_map_values, TEXTURE_ARGS);
		case 5390: return ReadInputMap5390(dg, input_map_values, TEXTURE_ARGS);
		case 5391: return ReadInputMap5391(dg, input_map_values, TEXTURE_ARGS);
		case 5392: return ReadInputMap5392(dg, input_map_values, TEXTURE_ARGS);
		case 5405: return ReadInputMap5405(dg, input_map_values, TEXTURE_ARGS);
		case 5417: return ReadInputMap5417(dg, input_map_values, TEXTURE_ARGS);
		case 5420: return ReadInputMap5420(dg, input_map_values, TEXTURE_ARGS);
		case 5424: return ReadInputMap5424(dg, input_map_values, TEXTURE_ARGS);
		case 5427: return ReadInputMap5427(dg, input_map_values, TEXTURE_ARGS);
		case 5439: return ReadInputMap5439(dg, input_map_values, TEXTURE_ARGS);
		case 5451: return ReadInputMap5451(dg, input_map_values, TEXTURE_ARGS);
		case 5461: return ReadInputMap5461(dg, input_map_values, TEXTURE_ARGS);
		case 5464: return ReadInputMap5464(dg, input_map_values, TEXTURE_ARGS);
		case 5465: return ReadInputMap5465(dg, input_map_values, TEXTURE_ARGS);
		case 5466: return ReadInputMap5466(dg, input_map_values, TEXTURE_ARGS);
		case 5478: return ReadInputMap5478(dg, input_map_values, TEXTURE_ARGS);
		case 5479: return ReadInputMap5479(dg, input_map_values, TEXTURE_ARGS);
		case 5481: return ReadInputMap5481(dg, input_map_values, TEXTURE_ARGS);
		case 5482: return ReadInputMap5482(dg, input_map_values, TEXTURE_ARGS);
		case 5510: return ReadInputMap5510(dg, input_map_values, TEXTURE_ARGS);
		case 5516: return ReadInputMap5516(dg, input_map_values, TEXTURE_ARGS);
		case 5517: return ReadInputMap5517(dg, input_map_values, TEXTURE_ARGS);
		case 5519: return ReadInputMap5519(dg, input_map_values, TEXTURE_ARGS);
		case 5520: return ReadInputMap5520(dg, input_map_values, TEXTURE_ARGS);
		case 5524: return ReadInputMap5524(dg, input_map_values, TEXTURE_ARGS);
		case 5530: return ReadInputMap5530(dg, input_map_values, TEXTURE_ARGS);
		case 5531: return ReadInputMap5531(dg, input_map_values, TEXTURE_ARGS);
		case 5536: return ReadInputMap5536(dg, input_map_values, TEXTURE_ARGS);
		case 5542: return ReadInputMap5542(dg, input_map_values, TEXTURE_ARGS);
		case 5548: return ReadInputMap5548(dg, input_map_values, TEXTURE_ARGS);
		case 5554: return ReadInputMap5554(dg, input_map_values, TEXTURE_ARGS);
		case 5555: return ReadInputMap5555(dg, input_map_values, TEXTURE_ARGS);
		case 5557: return ReadInputMap5557(dg, input_map_values, TEXTURE_ARGS);
		case 5558: return ReadInputMap5558(dg, input_map_values, TEXTURE_ARGS);
		case 5562: return ReadInputMap5562(dg, input_map_values, TEXTURE_ARGS);
		case 5568: return ReadInputMap5568(dg, input_map_values, TEXTURE_ARGS);
		case 5569: return ReadInputMap5569(dg, input_map_values, TEXTURE_ARGS);
		case 5571: return ReadInputMap5571(dg, input_map_values, TEXTURE_ARGS);
		case 5572: return ReadInputMap5572(dg, input_map_values, TEXTURE_ARGS);
		case 5576: return ReadInputMap5576(dg, input_map_values, TEXTURE_ARGS);
		case 5577: return ReadInputMap5577(dg, input_map_values, TEXTURE_ARGS);
		case 5578: return ReadInputMap5578(dg, input_map_values, TEXTURE_ARGS);
		case 5582: return ReadInputMap5582(dg, input_map_values, TEXTURE_ARGS);
		case 5583: return ReadInputMap5583(dg, input_map_values, TEXTURE_ARGS);
		case 5587: return ReadInputMap5587(dg, input_map_values, TEXTURE_ARGS);
		case 5590: return ReadInputMap5590(dg, input_map_values, TEXTURE_ARGS);
		case 5597: return ReadInputMap5597(dg, input_map_values, TEXTURE_ARGS);
		case 5598: return ReadInputMap5598(dg, input_map_values, TEXTURE_ARGS);
		case 5600: return ReadInputMap5600(dg, input_map_values, TEXTURE_ARGS);
		case 5601: return ReadInputMap5601(dg, input_map_values, TEXTURE_ARGS);
		case 5605: return ReadInputMap5605(dg, input_map_values, TEXTURE_ARGS);
		case 5608: return ReadInputMap5608(dg, input_map_values, TEXTURE_ARGS);
		case 5620: return ReadInputMap5620(dg, input_map_values, TEXTURE_ARGS);
		case 5627: return ReadInputMap5627(dg, input_map_values, TEXTURE_ARGS);
		case 5628: return ReadInputMap5628(dg, input_map_values, TEXTURE_ARGS);
		case 5633: return ReadInputMap5633(dg, input_map_values, TEXTURE_ARGS);
		case 5639: return ReadInputMap5639(dg, input_map_values, TEXTURE_ARGS);
		case 5642: return ReadInputMap5642(dg, input_map_values, TEXTURE_ARGS);
		case 5643: return ReadInputMap5643(dg, input_map_values, TEXTURE_ARGS);
		case 5644: return ReadInputMap5644(dg, input_map_values, TEXTURE_ARGS);
		case 5648: return ReadInputMap5648(dg, input_map_values, TEXTURE_ARGS);
		case 5649: return ReadInputMap5649(dg, input_map_values, TEXTURE_ARGS);
		case 5653: return ReadInputMap5653(dg, input_map_values, TEXTURE_ARGS);
		case 5658: return ReadInputMap5658(dg, input_map_values, TEXTURE_ARGS);
		case 5673: return ReadInputMap5673(dg, input_map_values, TEXTURE_ARGS);
		case 5676: return ReadInputMap5676(dg, input_map_values, TEXTURE_ARGS);
		case 5677: return ReadInputMap5677(dg, input_map_values, TEXTURE_ARGS);
		case 5678: return ReadInputMap5678(dg, input_map_values, TEXTURE_ARGS);
		case 5689: return ReadInputMap5689(dg, input_map_values, TEXTURE_ARGS);
		case 5690: return ReadInputMap5690(dg, input_map_values, TEXTURE_ARGS);
		case 5692: return ReadInputMap5692(dg, input_map_values, TEXTURE_ARGS);
		case 5693: return ReadInputMap5693(dg, input_map_values, TEXTURE_ARGS);
		case 5694: return ReadInputMap5694(dg, input_map_values, TEXTURE_ARGS);
		case 5695: return ReadInputMap5695(dg, input_map_values, TEXTURE_ARGS);
		case 5696: return ReadInputMap5696(dg, input_map_values, TEXTURE_ARGS);
		case 5700: return ReadInputMap5700(dg, input_map_values, TEXTURE_ARGS);
		case 5706: return ReadInputMap5706(dg, input_map_values, TEXTURE_ARGS);
		case 5707: return ReadInputMap5707(dg, input_map_values, TEXTURE_ARGS);
		case 5709: return ReadInputMap5709(dg, input_map_values, TEXTURE_ARGS);
		case 5710: return ReadInputMap5710(dg, input_map_values, TEXTURE_ARGS);
		case 5714: return ReadInputMap5714(dg, input_map_values, TEXTURE_ARGS);
		case 5717: return ReadInputMap5717(dg, input_map_values, TEXTURE_ARGS);
		case 5723: return ReadInputMap5723(dg, input_map_values, TEXTURE_ARGS);
		case 5724: return ReadInputMap5724(dg, input_map_values, TEXTURE_ARGS);
		case 5726: return ReadInputMap5726(dg, input_map_values, TEXTURE_ARGS);
		case 5727: return ReadInputMap5727(dg, input_map_values, TEXTURE_ARGS);
		case 5737: return ReadInputMap5737(dg, input_map_values, TEXTURE_ARGS);
		case 5738: return ReadInputMap5738(dg, input_map_values, TEXTURE_ARGS);
		case 5740: return ReadInputMap5740(dg, input_map_values, TEXTURE_ARGS);
		case 5741: return ReadInputMap5741(dg, input_map_values, TEXTURE_ARGS);
		case 5745: return ReadInputMap5745(dg, input_map_values, TEXTURE_ARGS);
		case 5751: return ReadInputMap5751(dg, input_map_values, TEXTURE_ARGS);
		case 5754: return ReadInputMap5754(dg, input_map_values, TEXTURE_ARGS);
		case 5755: return ReadInputMap5755(dg, input_map_values, TEXTURE_ARGS);
		case 5756: return ReadInputMap5756(dg, input_map_values, TEXTURE_ARGS);
		case 5762: return ReadInputMap5762(dg, input_map_values, TEXTURE_ARGS);
		case 5765: return ReadInputMap5765(dg, input_map_values, TEXTURE_ARGS);
		case 5766: return ReadInputMap5766(dg, input_map_values, TEXTURE_ARGS);
		case 5767: return ReadInputMap5767(dg, input_map_values, TEXTURE_ARGS);
		case 5771: return ReadInputMap5771(dg, input_map_values, TEXTURE_ARGS);
		case 5775: return ReadInputMap5775(dg, input_map_values, TEXTURE_ARGS);
		case 5778: return ReadInputMap5778(dg, input_map_values, TEXTURE_ARGS);
		case 5779: return ReadInputMap5779(dg, input_map_values, TEXTURE_ARGS);
		case 5780: return ReadInputMap5780(dg, input_map_values, TEXTURE_ARGS);
		case 5784: return ReadInputMap5784(dg, input_map_values, TEXTURE_ARGS);
		case 5790: return ReadInputMap5790(dg, input_map_values, TEXTURE_ARGS);
		case 5791: return ReadInputMap5791(dg, input_map_values, TEXTURE_ARGS);
		case 5793: return ReadInputMap5793(dg, input_map_values, TEXTURE_ARGS);
		case 5794: return ReadInputMap5794(dg, input_map_values, TEXTURE_ARGS);
		case 5798: return ReadInputMap5798(dg, input_map_values, TEXTURE_ARGS);
		case 5807: return ReadInputMap5807(dg, input_map_values, TEXTURE_ARGS);
		case 5808: return ReadInputMap5808(dg, input_map_values, TEXTURE_ARGS);
		case 5810: return ReadInputMap5810(dg, input_map_values, TEXTURE_ARGS);
		case 5811: return ReadInputMap5811(dg, input_map_values, TEXTURE_ARGS);
		case 5815: return ReadInputMap5815(dg, input_map_values, TEXTURE_ARGS);
		case 5821: return ReadInputMap5821(dg, input_map_values, TEXTURE_ARGS);
		case 5822: return ReadInputMap5822(dg, input_map_values, TEXTURE_ARGS);
		case 5824: return ReadInputMap5824(dg, input_map_values, TEXTURE_ARGS);
		case 5825: return ReadInputMap5825(dg, input_map_values, TEXTURE_ARGS);
		case 5829: return ReadInputMap5829(dg, input_map_values, TEXTURE_ARGS);
		case 5841: return ReadInputMap5841(dg, input_map_values, TEXTURE_ARGS);
		case 5859: return ReadInputMap5859(dg, input_map_values, TEXTURE_ARGS);
		case 5862: return ReadInputMap5862(dg, input_map_values, TEXTURE_ARGS);
		case 5868: return ReadInputMap5868(dg, input_map_values, TEXTURE_ARGS);
		case 5869: return ReadInputMap5869(dg, input_map_values, TEXTURE_ARGS);
		case 5871: return ReadInputMap5871(dg, input_map_values, TEXTURE_ARGS);
		case 5872: return ReadInputMap5872(dg, input_map_values, TEXTURE_ARGS);
		case 5876: return ReadInputMap5876(dg, input_map_values, TEXTURE_ARGS);
		case 5879: return ReadInputMap5879(dg, input_map_values, TEXTURE_ARGS);
		case 5885: return ReadInputMap5885(dg, input_map_values, TEXTURE_ARGS);
		case 5886: return ReadInputMap5886(dg, input_map_values, TEXTURE_ARGS);
		case 5888: return ReadInputMap5888(dg, input_map_values, TEXTURE_ARGS);
		case 5889: return ReadInputMap5889(dg, input_map_values, TEXTURE_ARGS);
		case 5893: return ReadInputMap5893(dg, input_map_values, TEXTURE_ARGS);
		case 5899: return ReadInputMap5899(dg, input_map_values, TEXTURE_ARGS);
		case 5900: return ReadInputMap5900(dg, input_map_values, TEXTURE_ARGS);
		case 5902: return ReadInputMap5902(dg, input_map_values, TEXTURE_ARGS);
		case 5903: return ReadInputMap5903(dg, input_map_values, TEXTURE_ARGS);
		case 5916: return ReadInputMap5916(dg, input_map_values, TEXTURE_ARGS);
		case 5917: return ReadInputMap5917(dg, input_map_values, TEXTURE_ARGS);
		case 5919: return ReadInputMap5919(dg, input_map_values, TEXTURE_ARGS);
		case 5920: return ReadInputMap5920(dg, input_map_values, TEXTURE_ARGS);
		case 5921: return ReadInputMap5921(dg, input_map_values, TEXTURE_ARGS);
		case 5922: return ReadInputMap5922(dg, input_map_values, TEXTURE_ARGS);
		case 5923: return ReadInputMap5923(dg, input_map_values, TEXTURE_ARGS);
		case 5927: return ReadInputMap5927(dg, input_map_values, TEXTURE_ARGS);
		case 5930: return ReadInputMap5930(dg, input_map_values, TEXTURE_ARGS);
		case 5936: return ReadInputMap5936(dg, input_map_values, TEXTURE_ARGS);
		case 5939: return ReadInputMap5939(dg, input_map_values, TEXTURE_ARGS);
		case 5940: return ReadInputMap5940(dg, input_map_values, TEXTURE_ARGS);
		case 5941: return ReadInputMap5941(dg, input_map_values, TEXTURE_ARGS);
		case 5949: return ReadInputMap5949(dg, input_map_values, TEXTURE_ARGS);
		case 5950: return ReadInputMap5950(dg, input_map_values, TEXTURE_ARGS);
		case 5952: return ReadInputMap5952(dg, input_map_values, TEXTURE_ARGS);
		case 5953: return ReadInputMap5953(dg, input_map_values, TEXTURE_ARGS);
		case 5957: return ReadInputMap5957(dg, input_map_values, TEXTURE_ARGS);
		case 5963: return ReadInputMap5963(dg, input_map_values, TEXTURE_ARGS);
		case 5964: return ReadInputMap5964(dg, input_map_values, TEXTURE_ARGS);
		case 5966: return ReadInputMap5966(dg, input_map_values, TEXTURE_ARGS);
		case 5967: return ReadInputMap5967(dg, input_map_values, TEXTURE_ARGS);
		case 5980: return ReadInputMap5980(dg, input_map_values, TEXTURE_ARGS);
		case 5983: return ReadInputMap5983(dg, input_map_values, TEXTURE_ARGS);
		case 5995: return ReadInputMap5995(dg, input_map_values, TEXTURE_ARGS);
		case 6001: return ReadInputMap6001(dg, input_map_values, TEXTURE_ARGS);
		case 6002: return ReadInputMap6002(dg, input_map_values, TEXTURE_ARGS);
		case 6004: return ReadInputMap6004(dg, input_map_values, TEXTURE_ARGS);
		case 6005: return ReadInputMap6005(dg, input_map_values, TEXTURE_ARGS);
		case 6009: return ReadInputMap6009(dg, input_map_values, TEXTURE_ARGS);
		case 6013: return ReadInputMap6013(dg, input_map_values, TEXTURE_ARGS);
		case 6016: return ReadInputMap6016(dg, input_map_values, TEXTURE_ARGS);
		case 6022: return ReadInputMap6022(dg, input_map_values, TEXTURE_ARGS);
		case 6025: return ReadInputMap6025(dg, input_map_values, TEXTURE_ARGS);
		case 6026: return ReadInputMap6026(dg, input_map_values, TEXTURE_ARGS);
		case 6027: return ReadInputMap6027(dg, input_map_values, TEXTURE_ARGS);
		case 6028: return ReadInputMap6028(dg, input_map_values, TEXTURE_ARGS);
		case 6029: return ReadInputMap6029(dg, input_map_values, TEXTURE_ARGS);
		case 6030: return ReadInputMap6030(dg, input_map_values, TEXTURE_ARGS);
		case 6034: return ReadInputMap6034(dg, input_map_values, TEXTURE_ARGS);
		case 6040: return ReadInputMap6040(dg, input_map_values, TEXTURE_ARGS);
		case 6041: return ReadInputMap6041(dg, input_map_values, TEXTURE_ARGS);
		case 6043: return ReadInputMap6043(dg, input_map_values, TEXTURE_ARGS);
		case 6044: return ReadInputMap6044(dg, input_map_values, TEXTURE_ARGS);
		case 6048: return ReadInputMap6048(dg, input_map_values, TEXTURE_ARGS);
		case 6051: return ReadInputMap6051(dg, input_map_values, TEXTURE_ARGS);
		case 6069: return ReadInputMap6069(dg, input_map_values, TEXTURE_ARGS);
		case 6075: return ReadInputMap6075(dg, input_map_values, TEXTURE_ARGS);
		case 6078: return ReadInputMap6078(dg, input_map_values, TEXTURE_ARGS);
		case 6079: return ReadInputMap6079(dg, input_map_values, TEXTURE_ARGS);
		case 6080: return ReadInputMap6080(dg, input_map_values, TEXTURE_ARGS);
		case 6084: return ReadInputMap6084(dg, input_map_values, TEXTURE_ARGS);
		case 6088: return ReadInputMap6088(dg, input_map_values, TEXTURE_ARGS);
		case 6102: return ReadInputMap6102(dg, input_map_values, TEXTURE_ARGS);
		case 6103: return ReadInputMap6103(dg, input_map_values, TEXTURE_ARGS);
		case 6111: return ReadInputMap6111(dg, input_map_values, TEXTURE_ARGS);
		case 6114: return ReadInputMap6114(dg, input_map_values, TEXTURE_ARGS);
		case 6115: return ReadInputMap6115(dg, input_map_values, TEXTURE_ARGS);
		case 6116: return ReadInputMap6116(dg, input_map_values, TEXTURE_ARGS);
		case 6123: return ReadInputMap6123(dg, input_map_values, TEXTURE_ARGS);
		case 6126: return ReadInputMap6126(dg, input_map_values, TEXTURE_ARGS);
		case 6127: return ReadInputMap6127(dg, input_map_values, TEXTURE_ARGS);
		case 6128: return ReadInputMap6128(dg, input_map_values, TEXTURE_ARGS);
		case 6138: return ReadInputMap6138(dg, input_map_values, TEXTURE_ARGS);
		case 6144: return ReadInputMap6144(dg, input_map_values, TEXTURE_ARGS);
		case 6147: return ReadInputMap6147(dg, input_map_values, TEXTURE_ARGS);
		case 6148: return ReadInputMap6148(dg, input_map_values, TEXTURE_ARGS);
		case 6149: return ReadInputMap6149(dg, input_map_values, TEXTURE_ARGS);
		case 6153: return ReadInputMap6153(dg, input_map_values, TEXTURE_ARGS);
		case 6154: return ReadInputMap6154(dg, input_map_values, TEXTURE_ARGS);
		case 6161: return ReadInputMap6161(dg, input_map_values, TEXTURE_ARGS);
		case 6164: return ReadInputMap6164(dg, input_map_values, TEXTURE_ARGS);
		case 6165: return ReadInputMap6165(dg, input_map_values, TEXTURE_ARGS);
		case 6166: return ReadInputMap6166(dg, input_map_values, TEXTURE_ARGS);
		case 6170: return ReadInputMap6170(dg, input_map_values, TEXTURE_ARGS);
		case 6174: return ReadInputMap6174(dg, input_map_values, TEXTURE_ARGS);
		case 6178: return ReadInputMap6178(dg, input_map_values, TEXTURE_ARGS);
		case 6184: return ReadInputMap6184(dg, input_map_values, TEXTURE_ARGS);
		case 6185: return ReadInputMap6185(dg, input_map_values, TEXTURE_ARGS);
		case 6187: return ReadInputMap6187(dg, input_map_values, TEXTURE_ARGS);
		case 6188: return ReadInputMap6188(dg, input_map_values, TEXTURE_ARGS);
		case 6192: return ReadInputMap6192(dg, input_map_values, TEXTURE_ARGS);
		case 6204: return ReadInputMap6204(dg, input_map_values, TEXTURE_ARGS);
		case 6210: return ReadInputMap6210(dg, input_map_values, TEXTURE_ARGS);
		case 6213: return ReadInputMap6213(dg, input_map_values, TEXTURE_ARGS);
		case 6214: return ReadInputMap6214(dg, input_map_values, TEXTURE_ARGS);
		case 6215: return ReadInputMap6215(dg, input_map_values, TEXTURE_ARGS);
		case 6222: return ReadInputMap6222(dg, input_map_values, TEXTURE_ARGS);
		case 6225: return ReadInputMap6225(dg, input_map_values, TEXTURE_ARGS);
		case 6226: return ReadInputMap6226(dg, input_map_values, TEXTURE_ARGS);
		case 6227: return ReadInputMap6227(dg, input_map_values, TEXTURE_ARGS);
		case 6231: return ReadInputMap6231(dg, input_map_values, TEXTURE_ARGS);
		case 6237: return ReadInputMap6237(dg, input_map_values, TEXTURE_ARGS);
		case 6238: return ReadInputMap6238(dg, input_map_values, TEXTURE_ARGS);
		case 6240: return ReadInputMap6240(dg, input_map_values, TEXTURE_ARGS);
		case 6241: return ReadInputMap6241(dg, input_map_values, TEXTURE_ARGS);
		case 6245: return ReadInputMap6245(dg, input_map_values, TEXTURE_ARGS);
		case 6251: return ReadInputMap6251(dg, input_map_values, TEXTURE_ARGS);
		case 6254: return ReadInputMap6254(dg, input_map_values, TEXTURE_ARGS);
		case 6255: return ReadInputMap6255(dg, input_map_values, TEXTURE_ARGS);
		case 6256: return ReadInputMap6256(dg, input_map_values, TEXTURE_ARGS);
		case 6260: return ReadInputMap6260(dg, input_map_values, TEXTURE_ARGS);
		case 6272: return ReadInputMap6272(dg, input_map_values, TEXTURE_ARGS);
		case 6276: return ReadInputMap6276(dg, input_map_values, TEXTURE_ARGS);
		case 6280: return ReadInputMap6280(dg, input_map_values, TEXTURE_ARGS);
		case 6286: return ReadInputMap6286(dg, input_map_values, TEXTURE_ARGS);
		case 6287: return ReadInputMap6287(dg, input_map_values, TEXTURE_ARGS);
		case 6289: return ReadInputMap6289(dg, input_map_values, TEXTURE_ARGS);
		case 6290: return ReadInputMap6290(dg, input_map_values, TEXTURE_ARGS);
		case 6294: return ReadInputMap6294(dg, input_map_values, TEXTURE_ARGS);
		case 6297: return ReadInputMap6297(dg, input_map_values, TEXTURE_ARGS);
		case 6303: return ReadInputMap6303(dg, input_map_values, TEXTURE_ARGS);
		case 6306: return ReadInputMap6306(dg, input_map_values, TEXTURE_ARGS);
		case 6307: return ReadInputMap6307(dg, input_map_values, TEXTURE_ARGS);
		case 6308: return ReadInputMap6308(dg, input_map_values, TEXTURE_ARGS);
		case 6312: return ReadInputMap6312(dg, input_map_values, TEXTURE_ARGS);
		case 6318: return ReadInputMap6318(dg, input_map_values, TEXTURE_ARGS);
		case 6321: return ReadInputMap6321(dg, input_map_values, TEXTURE_ARGS);
		case 6322: return ReadInputMap6322(dg, input_map_values, TEXTURE_ARGS);
		case 6323: return ReadInputMap6323(dg, input_map_values, TEXTURE_ARGS);
		case 6333: return ReadInputMap6333(dg, input_map_values, TEXTURE_ARGS);
		case 6334: return ReadInputMap6334(dg, input_map_values, TEXTURE_ARGS);
		case 6336: return ReadInputMap6336(dg, input_map_values, TEXTURE_ARGS);
		case 6337: return ReadInputMap6337(dg, input_map_values, TEXTURE_ARGS);
		case 6341: return ReadInputMap6341(dg, input_map_values, TEXTURE_ARGS);
		case 6345: return ReadInputMap6345(dg, input_map_values, TEXTURE_ARGS);
		case 6349: return ReadInputMap6349(dg, input_map_values, TEXTURE_ARGS);
		case 6353: return ReadInputMap6353(dg, input_map_values, TEXTURE_ARGS);
		case 6359: return ReadInputMap6359(dg, input_map_values, TEXTURE_ARGS);
		case 6360: return ReadInputMap6360(dg, input_map_values, TEXTURE_ARGS);
		case 6368: return ReadInputMap6368(dg, input_map_values, TEXTURE_ARGS);
		case 6371: return ReadInputMap6371(dg, input_map_values, TEXTURE_ARGS);
		case 6372: return ReadInputMap6372(dg, input_map_values, TEXTURE_ARGS);
		case 6373: return ReadInputMap6373(dg, input_map_values, TEXTURE_ARGS);
		case 6386: return ReadInputMap6386(dg, input_map_values, TEXTURE_ARGS);
		case 6387: return ReadInputMap6387(dg, input_map_values, TEXTURE_ARGS);
		case 6391: return ReadInputMap6391(dg, input_map_values, TEXTURE_ARGS);
		case 6395: return ReadInputMap6395(dg, input_map_values, TEXTURE_ARGS);
		case 6398: return ReadInputMap6398(dg, input_map_values, TEXTURE_ARGS);
		case 6401: return ReadInputMap6401(dg, input_map_values, TEXTURE_ARGS);
		case 6405: return ReadInputMap6405(dg, input_map_values, TEXTURE_ARGS);
		case 6430: return ReadInputMap6430(dg, input_map_values, TEXTURE_ARGS);
		case 6436: return ReadInputMap6436(dg, input_map_values, TEXTURE_ARGS);
		case 6437: return ReadInputMap6437(dg, input_map_values, TEXTURE_ARGS);
		case 6439: return ReadInputMap6439(dg, input_map_values, TEXTURE_ARGS);
		case 6440: return ReadInputMap6440(dg, input_map_values, TEXTURE_ARGS);
		case 6448: return ReadInputMap6448(dg, input_map_values, TEXTURE_ARGS);
		case 6451: return ReadInputMap6451(dg, input_map_values, TEXTURE_ARGS);
		case 6452: return ReadInputMap6452(dg, input_map_values, TEXTURE_ARGS);
		case 6453: return ReadInputMap6453(dg, input_map_values, TEXTURE_ARGS);
		case 6457: return ReadInputMap6457(dg, input_map_values, TEXTURE_ARGS);
		case 6461: return ReadInputMap6461(dg, input_map_values, TEXTURE_ARGS);
		case 6464: return ReadInputMap6464(dg, input_map_values, TEXTURE_ARGS);
		case 6468: return ReadInputMap6468(dg, input_map_values, TEXTURE_ARGS);
		case 6471: return ReadInputMap6471(dg, input_map_values, TEXTURE_ARGS);
		case 6477: return ReadInputMap6477(dg, input_map_values, TEXTURE_ARGS);
		case 6488: return ReadInputMap6488(dg, input_map_values, TEXTURE_ARGS);
		case 6489: return ReadInputMap6489(dg, input_map_values, TEXTURE_ARGS);
		case 6491: return ReadInputMap6491(dg, input_map_values, TEXTURE_ARGS);
		case 6492: return ReadInputMap6492(dg, input_map_values, TEXTURE_ARGS);
		case 6496: return ReadInputMap6496(dg, input_map_values, TEXTURE_ARGS);
		case 6497: return ReadInputMap6497(dg, input_map_values, TEXTURE_ARGS);
		case 6498: return ReadInputMap6498(dg, input_map_values, TEXTURE_ARGS);
		case 6502: return ReadInputMap6502(dg, input_map_values, TEXTURE_ARGS);
		case 6508: return ReadInputMap6508(dg, input_map_values, TEXTURE_ARGS);
		case 6509: return ReadInputMap6509(dg, input_map_values, TEXTURE_ARGS);
		case 6511: return ReadInputMap6511(dg, input_map_values, TEXTURE_ARGS);
		case 6512: return ReadInputMap6512(dg, input_map_values, TEXTURE_ARGS);
		case 6516: return ReadInputMap6516(dg, input_map_values, TEXTURE_ARGS);
		case 6519: return ReadInputMap6519(dg, input_map_values, TEXTURE_ARGS);
		case 6522: return ReadInputMap6522(dg, input_map_values, TEXTURE_ARGS);
		case 6523: return ReadInputMap6523(dg, input_map_values, TEXTURE_ARGS);
		case 6524: return ReadInputMap6524(dg, input_map_values, TEXTURE_ARGS);
		case 6528: return ReadInputMap6528(dg, input_map_values, TEXTURE_ARGS);
		case 6531: return ReadInputMap6531(dg, input_map_values, TEXTURE_ARGS);
		case 6537: return ReadInputMap6537(dg, input_map_values, TEXTURE_ARGS);
		case 6543: return ReadInputMap6543(dg, input_map_values, TEXTURE_ARGS);
		case 6544: return ReadInputMap6544(dg, input_map_values, TEXTURE_ARGS);
		case 6545: return ReadInputMap6545(dg, input_map_values, TEXTURE_ARGS);
		case 6549: return ReadInputMap6549(dg, input_map_values, TEXTURE_ARGS);
		case 6558: return ReadInputMap6558(dg, input_map_values, TEXTURE_ARGS);
		case 6561: return ReadInputMap6561(dg, input_map_values, TEXTURE_ARGS);
		case 6562: return ReadInputMap6562(dg, input_map_values, TEXTURE_ARGS);
		case 6563: return ReadInputMap6563(dg, input_map_values, TEXTURE_ARGS);
		case 6567: return ReadInputMap6567(dg, input_map_values, TEXTURE_ARGS);
		case 6573: return ReadInputMap6573(dg, input_map_values, TEXTURE_ARGS);
		case 6576: return ReadInputMap6576(dg, input_map_values, TEXTURE_ARGS);
		case 6577: return ReadInputMap6577(dg, input_map_values, TEXTURE_ARGS);
		case 6578: return ReadInputMap6578(dg, input_map_values, TEXTURE_ARGS);
		case 6582: return ReadInputMap6582(dg, input_map_values, TEXTURE_ARGS);
		case 6588: return ReadInputMap6588(dg, input_map_values, TEXTURE_ARGS);
		case 6591: return ReadInputMap6591(dg, input_map_values, TEXTURE_ARGS);
		case 6592: return ReadInputMap6592(dg, input_map_values, TEXTURE_ARGS);
		case 6593: return ReadInputMap6593(dg, input_map_values, TEXTURE_ARGS);
		case 6608: return ReadInputMap6608(dg, input_map_values, TEXTURE_ARGS);
		case 6609: return ReadInputMap6609(dg, input_map_values, TEXTURE_ARGS);
		case 6611: return ReadInputMap6611(dg, input_map_values, TEXTURE_ARGS);
		case 6612: return ReadInputMap6612(dg, input_map_values, TEXTURE_ARGS);
		case 6619: return ReadInputMap6619(dg, input_map_values, TEXTURE_ARGS);
		case 6631: return ReadInputMap6631(dg, input_map_values, TEXTURE_ARGS);
		case 6637: return ReadInputMap6637(dg, input_map_values, TEXTURE_ARGS);
		case 6638: return ReadInputMap6638(dg, input_map_values, TEXTURE_ARGS);
		case 6640: return ReadInputMap6640(dg, input_map_values, TEXTURE_ARGS);
		case 6641: return ReadInputMap6641(dg, input_map_values, TEXTURE_ARGS);
		case 6649: return ReadInputMap6649(dg, input_map_values, TEXTURE_ARGS);
		case 6668: return ReadInputMap6668(dg, input_map_values, TEXTURE_ARGS);
		case 6674: return ReadInputMap6674(dg, input_map_values, TEXTURE_ARGS);
		case 6675: return ReadInputMap6675(dg, input_map_values, TEXTURE_ARGS);
		case 6677: return ReadInputMap6677(dg, input_map_values, TEXTURE_ARGS);
		case 6678: return ReadInputMap6678(dg, input_map_values, TEXTURE_ARGS);
		case 6685: return ReadInputMap6685(dg, input_map_values, TEXTURE_ARGS);
		case 6686: return ReadInputMap6686(dg, input_map_values, TEXTURE_ARGS);
		case 6688: return ReadInputMap6688(dg, input_map_values, TEXTURE_ARGS);
		case 6689: return ReadInputMap6689(dg, input_map_values, TEXTURE_ARGS);
		case 6699: return ReadInputMap6699(dg, input_map_values, TEXTURE_ARGS);
		case 6703: return ReadInputMap6703(dg, input_map_values, TEXTURE_ARGS);
		case 6707: return ReadInputMap6707(dg, input_map_values, TEXTURE_ARGS);
		case 6719: return ReadInputMap6719(dg, input_map_values, TEXTURE_ARGS);
		case 6725: return ReadInputMap6725(dg, input_map_values, TEXTURE_ARGS);
		case 6726: return ReadInputMap6726(dg, input_map_values, TEXTURE_ARGS);
		case 6728: return ReadInputMap6728(dg, input_map_values, TEXTURE_ARGS);
		case 6729: return ReadInputMap6729(dg, input_map_values, TEXTURE_ARGS);
		case 6733: return ReadInputMap6733(dg, input_map_values, TEXTURE_ARGS);
		case 6739: return ReadInputMap6739(dg, input_map_values, TEXTURE_ARGS);
		case 6740: return ReadInputMap6740(dg, input_map_values, TEXTURE_ARGS);
		case 6742: return ReadInputMap6742(dg, input_map_values, TEXTURE_ARGS);
		case 6743: return ReadInputMap6743(dg, input_map_values, TEXTURE_ARGS);
		case 6753: return ReadInputMap6753(dg, input_map_values, TEXTURE_ARGS);
		case 6757: return ReadInputMap6757(dg, input_map_values, TEXTURE_ARGS);
		case 6761: return ReadInputMap6761(dg, input_map_values, TEXTURE_ARGS);
		case 6769: return ReadInputMap6769(dg, input_map_values, TEXTURE_ARGS);
		case 6770: return ReadInputMap6770(dg, input_map_values, TEXTURE_ARGS);
		case 6772: return ReadInputMap6772(dg, input_map_values, TEXTURE_ARGS);
		case 6773: return ReadInputMap6773(dg, input_map_values, TEXTURE_ARGS);
		case 6783: return ReadInputMap6783(dg, input_map_values, TEXTURE_ARGS);
		case 6787: return ReadInputMap6787(dg, input_map_values, TEXTURE_ARGS);
		case 6791: return ReadInputMap6791(dg, input_map_values, TEXTURE_ARGS);
		case 6797: return ReadInputMap6797(dg, input_map_values, TEXTURE_ARGS);
		case 6798: return ReadInputMap6798(dg, input_map_values, TEXTURE_ARGS);
		case 6800: return ReadInputMap6800(dg, input_map_values, TEXTURE_ARGS);
		case 6801: return ReadInputMap6801(dg, input_map_values, TEXTURE_ARGS);
		case 6805: return ReadInputMap6805(dg, input_map_values, TEXTURE_ARGS);
		case 6811: return ReadInputMap6811(dg, input_map_values, TEXTURE_ARGS);
		case 6812: return ReadInputMap6812(dg, input_map_values, TEXTURE_ARGS);
		case 6814: return ReadInputMap6814(dg, input_map_values, TEXTURE_ARGS);
		case 6815: return ReadInputMap6815(dg, input_map_values, TEXTURE_ARGS);
		case 6819: return ReadInputMap6819(dg, input_map_values, TEXTURE_ARGS);
		case 6831: return ReadInputMap6831(dg, input_map_values, TEXTURE_ARGS);
		case 6837: return ReadInputMap6837(dg, input_map_values, TEXTURE_ARGS);
		case 6838: return ReadInputMap6838(dg, input_map_values, TEXTURE_ARGS);
		case 6840: return ReadInputMap6840(dg, input_map_values, TEXTURE_ARGS);
		case 6841: return ReadInputMap6841(dg, input_map_values, TEXTURE_ARGS);
		case 6845: return ReadInputMap6845(dg, input_map_values, TEXTURE_ARGS);
		case 6851: return ReadInputMap6851(dg, input_map_values, TEXTURE_ARGS);
		case 6852: return ReadInputMap6852(dg, input_map_values, TEXTURE_ARGS);
		case 6854: return ReadInputMap6854(dg, input_map_values, TEXTURE_ARGS);
		case 6855: return ReadInputMap6855(dg, input_map_values, TEXTURE_ARGS);
		case 6865: return ReadInputMap6865(dg, input_map_values, TEXTURE_ARGS);
		case 6869: return ReadInputMap6869(dg, input_map_values, TEXTURE_ARGS);
		case 6872: return ReadInputMap6872(dg, input_map_values, TEXTURE_ARGS);
		case 6875: return ReadInputMap6875(dg, input_map_values, TEXTURE_ARGS);
		case 6879: return ReadInputMap6879(dg, input_map_values, TEXTURE_ARGS);
		case 6885: return ReadInputMap6885(dg, input_map_values, TEXTURE_ARGS);
		case 6888: return ReadInputMap6888(dg, input_map_values, TEXTURE_ARGS);
		case 6889: return ReadInputMap6889(dg, input_map_values, TEXTURE_ARGS);
		case 6890: return ReadInputMap6890(dg, input_map_values, TEXTURE_ARGS);
		case 6896: return ReadInputMap6896(dg, input_map_values, TEXTURE_ARGS);
		case 6897: return ReadInputMap6897(dg, input_map_values, TEXTURE_ARGS);
		case 6899: return ReadInputMap6899(dg, input_map_values, TEXTURE_ARGS);
		case 6900: return ReadInputMap6900(dg, input_map_values, TEXTURE_ARGS);
		case 6913: return ReadInputMap6913(dg, input_map_values, TEXTURE_ARGS);
		case 6921: return ReadInputMap6921(dg, input_map_values, TEXTURE_ARGS);
		case 6922: return ReadInputMap6922(dg, input_map_values, TEXTURE_ARGS);
		case 6924: return ReadInputMap6924(dg, input_map_values, TEXTURE_ARGS);
		case 6925: return ReadInputMap6925(dg, input_map_values, TEXTURE_ARGS);
		case 6949: return ReadInputMap6949(dg, input_map_values, TEXTURE_ARGS);
		case 6950: return ReadInputMap6950(dg, input_map_values, TEXTURE_ARGS);
		case 6955: return ReadInputMap6955(dg, input_map_values, TEXTURE_ARGS);
		case 6959: return ReadInputMap6959(dg, input_map_values, TEXTURE_ARGS);
		case 6963: return ReadInputMap6963(dg, input_map_values, TEXTURE_ARGS);
		case 6969: return ReadInputMap6969(dg, input_map_values, TEXTURE_ARGS);
		case 6970: return ReadInputMap6970(dg, input_map_values, TEXTURE_ARGS);
		case 6972: return ReadInputMap6972(dg, input_map_values, TEXTURE_ARGS);
		case 6973: return ReadInputMap6973(dg, input_map_values, TEXTURE_ARGS);
		case 6977: return ReadInputMap6977(dg, input_map_values, TEXTURE_ARGS);
		case 6983: return ReadInputMap6983(dg, input_map_values, TEXTURE_ARGS);
		case 6984: return ReadInputMap6984(dg, input_map_values, TEXTURE_ARGS);
		case 6989: return ReadInputMap6989(dg, input_map_values, TEXTURE_ARGS);
		case 6990: return ReadInputMap6990(dg, input_map_values, TEXTURE_ARGS);
		case 6994: return ReadInputMap6994(dg, input_map_values, TEXTURE_ARGS);
		case 6995: return ReadInputMap6995(dg, input_map_values, TEXTURE_ARGS);
		case 6999: return ReadInputMap6999(dg, input_map_values, TEXTURE_ARGS);
		case 7005: return ReadInputMap7005(dg, input_map_values, TEXTURE_ARGS);
		case 7006: return ReadInputMap7006(dg, input_map_values, TEXTURE_ARGS);
		case 7008: return ReadInputMap7008(dg, input_map_values, TEXTURE_ARGS);
		case 7009: return ReadInputMap7009(dg, input_map_values, TEXTURE_ARGS);
		case 7016: return ReadInputMap7016(dg, input_map_values, TEXTURE_ARGS);
		case 7017: return ReadInputMap7017(dg, input_map_values, TEXTURE_ARGS);
		case 7019: return ReadInputMap7019(dg, input_map_values, TEXTURE_ARGS);
		case 7020: return ReadInputMap7020(dg, input_map_values, TEXTURE_ARGS);
		case 7024: return ReadInputMap7024(dg, input_map_values, TEXTURE_ARGS);
		case 7030: return ReadInputMap7030(dg, input_map_values, TEXTURE_ARGS);
		case 7033: return ReadInputMap7033(dg, input_map_values, TEXTURE_ARGS);
		case 7034: return ReadInputMap7034(dg, input_map_values, TEXTURE_ARGS);
		case 7035: return ReadInputMap7035(dg, input_map_values, TEXTURE_ARGS);
		case 7039: return ReadInputMap7039(dg, input_map_values, TEXTURE_ARGS);
		case 7042: return ReadInputMap7042(dg, input_map_values, TEXTURE_ARGS);
		case 7048: return ReadInputMap7048(dg, input_map_values, TEXTURE_ARGS);
		case 7049: return ReadInputMap7049(dg, input_map_values, TEXTURE_ARGS);
		case 7051: return ReadInputMap7051(dg, input_map_values, TEXTURE_ARGS);
		case 7052: return ReadInputMap7052(dg, input_map_values, TEXTURE_ARGS);
		case 7056: return ReadInputMap7056(dg, input_map_values, TEXTURE_ARGS);
		case 7060: return ReadInputMap7060(dg, input_map_values, TEXTURE_ARGS);
		case 7068: return ReadInputMap7068(dg, input_map_values, TEXTURE_ARGS);
		case 7069: return ReadInputMap7069(dg, input_map_values, TEXTURE_ARGS);
		case 7071: return ReadInputMap7071(dg, input_map_values, TEXTURE_ARGS);
		case 7072: return ReadInputMap7072(dg, input_map_values, TEXTURE_ARGS);
		case 7076: return ReadInputMap7076(dg, input_map_values, TEXTURE_ARGS);
		case 7088: return ReadInputMap7088(dg, input_map_values, TEXTURE_ARGS);
		case 7100: return ReadInputMap7100(dg, input_map_values, TEXTURE_ARGS);
		case 7103: return ReadInputMap7103(dg, input_map_values, TEXTURE_ARGS);
		case 7109: return ReadInputMap7109(dg, input_map_values, TEXTURE_ARGS);
		case 7110: return ReadInputMap7110(dg, input_map_values, TEXTURE_ARGS);
		case 7112: return ReadInputMap7112(dg, input_map_values, TEXTURE_ARGS);
		case 7113: return ReadInputMap7113(dg, input_map_values, TEXTURE_ARGS);
		case 7117: return ReadInputMap7117(dg, input_map_values, TEXTURE_ARGS);
		case 7120: return ReadInputMap7120(dg, input_map_values, TEXTURE_ARGS);
		case 7129: return ReadInputMap7129(dg, input_map_values, TEXTURE_ARGS);
		case 7130: return ReadInputMap7130(dg, input_map_values, TEXTURE_ARGS);
		case 7132: return ReadInputMap7132(dg, input_map_values, TEXTURE_ARGS);
		case 7133: return ReadInputMap7133(dg, input_map_values, TEXTURE_ARGS);
		case 7145: return ReadInputMap7145(dg, input_map_values, TEXTURE_ARGS);
		case 7146: return ReadInputMap7146(dg, input_map_values, TEXTURE_ARGS);
		case 7147: return ReadInputMap7147(dg, input_map_values, TEXTURE_ARGS);
		case 7151: return ReadInputMap7151(dg, input_map_values, TEXTURE_ARGS);
		case 7171: return ReadInputMap7171(dg, input_map_values, TEXTURE_ARGS);
		case 7174: return ReadInputMap7174(dg, input_map_values, TEXTURE_ARGS);
		case 7175: return ReadInputMap7175(dg, input_map_values, TEXTURE_ARGS);
		case 7176: return ReadInputMap7176(dg, input_map_values, TEXTURE_ARGS);
		case 7180: return ReadInputMap7180(dg, input_map_values, TEXTURE_ARGS);
		case 7199: return ReadInputMap7199(dg, input_map_values, TEXTURE_ARGS);
		case 7200: return ReadInputMap7200(dg, input_map_values, TEXTURE_ARGS);
		case 7202: return ReadInputMap7202(dg, input_map_values, TEXTURE_ARGS);
		case 7203: return ReadInputMap7203(dg, input_map_values, TEXTURE_ARGS);
		case 7207: return ReadInputMap7207(dg, input_map_values, TEXTURE_ARGS);
		case 7213: return ReadInputMap7213(dg, input_map_values, TEXTURE_ARGS);
		case 7214: return ReadInputMap7214(dg, input_map_values, TEXTURE_ARGS);
		case 7216: return ReadInputMap7216(dg, input_map_values, TEXTURE_ARGS);
		case 7217: return ReadInputMap7217(dg, input_map_values, TEXTURE_ARGS);
		case 7221: return ReadInputMap7221(dg, input_map_values, TEXTURE_ARGS);
		case 7227: return ReadInputMap7227(dg, input_map_values, TEXTURE_ARGS);
		case 7230: return ReadInputMap7230(dg, input_map_values, TEXTURE_ARGS);
		case 7231: return ReadInputMap7231(dg, input_map_values, TEXTURE_ARGS);
		case 7232: return ReadInputMap7232(dg, input_map_values, TEXTURE_ARGS);
		case 7236: return ReadInputMap7236(dg, input_map_values, TEXTURE_ARGS);
		case 7245: return ReadInputMap7245(dg, input_map_values, TEXTURE_ARGS);
		case 7246: return ReadInputMap7246(dg, input_map_values, TEXTURE_ARGS);
		case 7248: return ReadInputMap7248(dg, input_map_values, TEXTURE_ARGS);
		case 7249: return ReadInputMap7249(dg, input_map_values, TEXTURE_ARGS);
		case 7253: return ReadInputMap7253(dg, input_map_values, TEXTURE_ARGS);
		case 7259: return ReadInputMap7259(dg, input_map_values, TEXTURE_ARGS);
		case 7260: return ReadInputMap7260(dg, input_map_values, TEXTURE_ARGS);
		case 7262: return ReadInputMap7262(dg, input_map_values, TEXTURE_ARGS);
		case 7263: return ReadInputMap7263(dg, input_map_values, TEXTURE_ARGS);
		case 7267: return ReadInputMap7267(dg, input_map_values, TEXTURE_ARGS);
		case 7270: return ReadInputMap7270(dg, input_map_values, TEXTURE_ARGS);
		case 7276: return ReadInputMap7276(dg, input_map_values, TEXTURE_ARGS);
		case 7277: return ReadInputMap7277(dg, input_map_values, TEXTURE_ARGS);
		case 7279: return ReadInputMap7279(dg, input_map_values, TEXTURE_ARGS);
		case 7280: return ReadInputMap7280(dg, input_map_values, TEXTURE_ARGS);
		case 7284: return ReadInputMap7284(dg, input_map_values, TEXTURE_ARGS);
		case 7290: return ReadInputMap7290(dg, input_map_values, TEXTURE_ARGS);
		case 7293: return ReadInputMap7293(dg, input_map_values, TEXTURE_ARGS);
		case 7294: return ReadInputMap7294(dg, input_map_values, TEXTURE_ARGS);
		case 7295: return ReadInputMap7295(dg, input_map_values, TEXTURE_ARGS);
		case 7299: return ReadInputMap7299(dg, input_map_values, TEXTURE_ARGS);
		case 7309: return ReadInputMap7309(dg, input_map_values, TEXTURE_ARGS);
		case 7312: return ReadInputMap7312(dg, input_map_values, TEXTURE_ARGS);
		case 7313: return ReadInputMap7313(dg, input_map_values, TEXTURE_ARGS);
		case 7314: return ReadInputMap7314(dg, input_map_values, TEXTURE_ARGS);
		case 7318: return ReadInputMap7318(dg, input_map_values, TEXTURE_ARGS);
		case 7324: return ReadInputMap7324(dg, input_map_values, TEXTURE_ARGS);
		case 7325: return ReadInputMap7325(dg, input_map_values, TEXTURE_ARGS);
		case 7327: return ReadInputMap7327(dg, input_map_values, TEXTURE_ARGS);
		case 7328: return ReadInputMap7328(dg, input_map_values, TEXTURE_ARGS);
		case 7332: return ReadInputMap7332(dg, input_map_values, TEXTURE_ARGS);
		case 7338: return ReadInputMap7338(dg, input_map_values, TEXTURE_ARGS);
		case 7339: return ReadInputMap7339(dg, input_map_values, TEXTURE_ARGS);
		case 7346: return ReadInputMap7346(dg, input_map_values, TEXTURE_ARGS);
		case 7347: return ReadInputMap7347(dg, input_map_values, TEXTURE_ARGS);
		case 7349: return ReadInputMap7349(dg, input_map_values, TEXTURE_ARGS);
		case 7350: return ReadInputMap7350(dg, input_map_values, TEXTURE_ARGS);
		case 7354: return ReadInputMap7354(dg, input_map_values, TEXTURE_ARGS);
		case 7357: return ReadInputMap7357(dg, input_map_values, TEXTURE_ARGS);
		case 7360: return ReadInputMap7360(dg, input_map_values, TEXTURE_ARGS);
		case 7363: return ReadInputMap7363(dg, input_map_values, TEXTURE_ARGS);
		case 7364: return ReadInputMap7364(dg, input_map_values, TEXTURE_ARGS);
		case 7365: return ReadInputMap7365(dg, input_map_values, TEXTURE_ARGS);
		case 7369: return ReadInputMap7369(dg, input_map_values, TEXTURE_ARGS);
		case 7370: return ReadInputMap7370(dg, input_map_values, TEXTURE_ARGS);
		case 7382: return ReadInputMap7382(dg, input_map_values, TEXTURE_ARGS);
		case 7385: return ReadInputMap7385(dg, input_map_values, TEXTURE_ARGS);
		case 7386: return ReadInputMap7386(dg, input_map_values, TEXTURE_ARGS);
		case 7387: return ReadInputMap7387(dg, input_map_values, TEXTURE_ARGS);
		case 7391: return ReadInputMap7391(dg, input_map_values, TEXTURE_ARGS);
		case 7395: return ReadInputMap7395(dg, input_map_values, TEXTURE_ARGS);
		case 7398: return ReadInputMap7398(dg, input_map_values, TEXTURE_ARGS);
		case 7401: return ReadInputMap7401(dg, input_map_values, TEXTURE_ARGS);
		case 7402: return ReadInputMap7402(dg, input_map_values, TEXTURE_ARGS);
		case 7403: return ReadInputMap7403(dg, input_map_values, TEXTURE_ARGS);
		case 7407: return ReadInputMap7407(dg, input_map_values, TEXTURE_ARGS);
		case 7413: return ReadInputMap7413(dg, input_map_values, TEXTURE_ARGS);
		case 7414: return ReadInputMap7414(dg, input_map_values, TEXTURE_ARGS);
		case 7419: return ReadInputMap7419(dg, input_map_values, TEXTURE_ARGS);
		case 7425: return ReadInputMap7425(dg, input_map_values, TEXTURE_ARGS);
		case 7428: return ReadInputMap7428(dg, input_map_values, TEXTURE_ARGS);
		case 7429: return ReadInputMap7429(dg, input_map_values, TEXTURE_ARGS);
		case 7430: return ReadInputMap7430(dg, input_map_values, TEXTURE_ARGS);
		case 7443: return ReadInputMap7443(dg, input_map_values, TEXTURE_ARGS);
		case 7444: return ReadInputMap7444(dg, input_map_values, TEXTURE_ARGS);
		case 7448: return ReadInputMap7448(dg, input_map_values, TEXTURE_ARGS);
		case 7451: return ReadInputMap7451(dg, input_map_values, TEXTURE_ARGS);
		case 7454: return ReadInputMap7454(dg, input_map_values, TEXTURE_ARGS);
		case 7455: return ReadInputMap7455(dg, input_map_values, TEXTURE_ARGS);
		case 7456: return ReadInputMap7456(dg, input_map_values, TEXTURE_ARGS);
		case 7460: return ReadInputMap7460(dg, input_map_values, TEXTURE_ARGS);
		case 7464: return ReadInputMap7464(dg, input_map_values, TEXTURE_ARGS);
		case 7467: return ReadInputMap7467(dg, input_map_values, TEXTURE_ARGS);
		case 7473: return ReadInputMap7473(dg, input_map_values, TEXTURE_ARGS);
		case 7476: return ReadInputMap7476(dg, input_map_values, TEXTURE_ARGS);
		case 7477: return ReadInputMap7477(dg, input_map_values, TEXTURE_ARGS);
		case 7478: return ReadInputMap7478(dg, input_map_values, TEXTURE_ARGS);
		case 7486: return ReadInputMap7486(dg, input_map_values, TEXTURE_ARGS);
		case 7487: return ReadInputMap7487(dg, input_map_values, TEXTURE_ARGS);
		case 7492: return ReadInputMap7492(dg, input_map_values, TEXTURE_ARGS);
		case 7495: return ReadInputMap7495(dg, input_map_values, TEXTURE_ARGS);
		case 7498: return ReadInputMap7498(dg, input_map_values, TEXTURE_ARGS);
		case 7499: return ReadInputMap7499(dg, input_map_values, TEXTURE_ARGS);
		case 7500: return ReadInputMap7500(dg, input_map_values, TEXTURE_ARGS);
		case 7504: return ReadInputMap7504(dg, input_map_values, TEXTURE_ARGS);
		case 7512: return ReadInputMap7512(dg, input_map_values, TEXTURE_ARGS);
		case 7524: return ReadInputMap7524(dg, input_map_values, TEXTURE_ARGS);
		case 7525: return ReadInputMap7525(dg, input_map_values, TEXTURE_ARGS);
		case 7532: return ReadInputMap7532(dg, input_map_values, TEXTURE_ARGS);
		case 7535: return ReadInputMap7535(dg, input_map_values, TEXTURE_ARGS);
		case 7536: return ReadInputMap7536(dg, input_map_values, TEXTURE_ARGS);
		case 7537: return ReadInputMap7537(dg, input_map_values, TEXTURE_ARGS);
		case 7541: return ReadInputMap7541(dg, input_map_values, TEXTURE_ARGS);
		case 7544: return ReadInputMap7544(dg, input_map_values, TEXTURE_ARGS);
		case 7547: return ReadInputMap7547(dg, input_map_values, TEXTURE_ARGS);
		case 7548: return ReadInputMap7548(dg, input_map_values, TEXTURE_ARGS);
		case 7549: return ReadInputMap7549(dg, input_map_values, TEXTURE_ARGS);
		case 7553: return ReadInputMap7553(dg, input_map_values, TEXTURE_ARGS);
		case 7559: return ReadInputMap7559(dg, input_map_values, TEXTURE_ARGS);
		case 7562: return ReadInputMap7562(dg, input_map_values, TEXTURE_ARGS);
		case 7563: return ReadInputMap7563(dg, input_map_values, TEXTURE_ARGS);
		case 7564: return ReadInputMap7564(dg, input_map_values, TEXTURE_ARGS);
		case 7568: return ReadInputMap7568(dg, input_map_values, TEXTURE_ARGS);
		case 7574: return ReadInputMap7574(dg, input_map_values, TEXTURE_ARGS);
		case 7577: return ReadInputMap7577(dg, input_map_values, TEXTURE_ARGS);
		case 7578: return ReadInputMap7578(dg, input_map_values, TEXTURE_ARGS);
		case 7579: return ReadInputMap7579(dg, input_map_values, TEXTURE_ARGS);
		case 7580: return ReadInputMap7580(dg, input_map_values, TEXTURE_ARGS);
		case 7586: return ReadInputMap7586(dg, input_map_values, TEXTURE_ARGS);
		case 7589: return ReadInputMap7589(dg, input_map_values, TEXTURE_ARGS);
		case 7590: return ReadInputMap7590(dg, input_map_values, TEXTURE_ARGS);
		case 7591: return ReadInputMap7591(dg, input_map_values, TEXTURE_ARGS);
		case 7595: return ReadInputMap7595(dg, input_map_values, TEXTURE_ARGS);
		case 7598: return ReadInputMap7598(dg, input_map_values, TEXTURE_ARGS);
		case 7610: return ReadInputMap7610(dg, input_map_values, TEXTURE_ARGS);
		case 7613: return ReadInputMap7613(dg, input_map_values, TEXTURE_ARGS);
		case 7614: return ReadInputMap7614(dg, input_map_values, TEXTURE_ARGS);
		case 7615: return ReadInputMap7615(dg, input_map_values, TEXTURE_ARGS);
		case 7628: return ReadInputMap7628(dg, input_map_values, TEXTURE_ARGS);
		case 7631: return ReadInputMap7631(dg, input_map_values, TEXTURE_ARGS);
		case 7637: return ReadInputMap7637(dg, input_map_values, TEXTURE_ARGS);
		case 7640: return ReadInputMap7640(dg, input_map_values, TEXTURE_ARGS);
		case 7641: return ReadInputMap7641(dg, input_map_values, TEXTURE_ARGS);
		case 7642: return ReadInputMap7642(dg, input_map_values, TEXTURE_ARGS);
		case 7643: return ReadInputMap7643(dg, input_map_values, TEXTURE_ARGS);
		case 7649: return ReadInputMap7649(dg, input_map_values, TEXTURE_ARGS);
		case 7652: return ReadInputMap7652(dg, input_map_values, TEXTURE_ARGS);
		case 7653: return ReadInputMap7653(dg, input_map_values, TEXTURE_ARGS);
		case 7654: return ReadInputMap7654(dg, input_map_values, TEXTURE_ARGS);
		case 7655: return ReadInputMap7655(dg, input_map_values, TEXTURE_ARGS);
		case 7658: return ReadInputMap7658(dg, input_map_values, TEXTURE_ARGS);
		case 7670: return ReadInputMap7670(dg, input_map_values, TEXTURE_ARGS);
		case 7676: return ReadInputMap7676(dg, input_map_values, TEXTURE_ARGS);
		case 7677: return ReadInputMap7677(dg, input_map_values, TEXTURE_ARGS);
		case 7679: return ReadInputMap7679(dg, input_map_values, TEXTURE_ARGS);
		case 7680: return ReadInputMap7680(dg, input_map_values, TEXTURE_ARGS);
		case 7687: return ReadInputMap7687(dg, input_map_values, TEXTURE_ARGS);
		case 7693: return ReadInputMap7693(dg, input_map_values, TEXTURE_ARGS);
		case 7694: return ReadInputMap7694(dg, input_map_values, TEXTURE_ARGS);
		case 7696: return ReadInputMap7696(dg, input_map_values, TEXTURE_ARGS);
		case 7697: return ReadInputMap7697(dg, input_map_values, TEXTURE_ARGS);
		case 7701: return ReadInputMap7701(dg, input_map_values, TEXTURE_ARGS);
		case 7707: return ReadInputMap7707(dg, input_map_values, TEXTURE_ARGS);
		case 7713: return ReadInputMap7713(dg, input_map_values, TEXTURE_ARGS);
		case 7714: return ReadInputMap7714(dg, input_map_values, TEXTURE_ARGS);
		case 7715: return ReadInputMap7715(dg, input_map_values, TEXTURE_ARGS);
		case 7719: return ReadInputMap7719(dg, input_map_values, TEXTURE_ARGS);
		case 7725: return ReadInputMap7725(dg, input_map_values, TEXTURE_ARGS);
		case 7726: return ReadInputMap7726(dg, input_map_values, TEXTURE_ARGS);
		case 7728: return ReadInputMap7728(dg, input_map_values, TEXTURE_ARGS);
		case 7729: return ReadInputMap7729(dg, input_map_values, TEXTURE_ARGS);
		case 7733: return ReadInputMap7733(dg, input_map_values, TEXTURE_ARGS);
		case 7736: return ReadInputMap7736(dg, input_map_values, TEXTURE_ARGS);
		case 7739: return ReadInputMap7739(dg, input_map_values, TEXTURE_ARGS);
		case 7742: return ReadInputMap7742(dg, input_map_values, TEXTURE_ARGS);
		case 7743: return ReadInputMap7743(dg, input_map_values, TEXTURE_ARGS);
		case 7744: return ReadInputMap7744(dg, input_map_values, TEXTURE_ARGS);
		case 7755: return ReadInputMap7755(dg, input_map_values, TEXTURE_ARGS);
		case 7758: return ReadInputMap7758(dg, input_map_values, TEXTURE_ARGS);
		case 7759: return ReadInputMap7759(dg, input_map_values, TEXTURE_ARGS);
		case 7760: return ReadInputMap7760(dg, input_map_values, TEXTURE_ARGS);
		case 7761: return ReadInputMap7761(dg, input_map_values, TEXTURE_ARGS);
		case 7762: return ReadInputMap7762(dg, input_map_values, TEXTURE_ARGS);
		case 7763: return ReadInputMap7763(dg, input_map_values, TEXTURE_ARGS);
		case 7772: return ReadInputMap7772(dg, input_map_values, TEXTURE_ARGS);
		case 7773: return ReadInputMap7773(dg, input_map_values, TEXTURE_ARGS);
		case 7778: return ReadInputMap7778(dg, input_map_values, TEXTURE_ARGS);
		case 7784: return ReadInputMap7784(dg, input_map_values, TEXTURE_ARGS);
		case 7785: return ReadInputMap7785(dg, input_map_values, TEXTURE_ARGS);
		case 7787: return ReadInputMap7787(dg, input_map_values, TEXTURE_ARGS);
		case 7788: return ReadInputMap7788(dg, input_map_values, TEXTURE_ARGS);
		case 7792: return ReadInputMap7792(dg, input_map_values, TEXTURE_ARGS);
		case 7793: return ReadInputMap7793(dg, input_map_values, TEXTURE_ARGS);
		case 7797: return ReadInputMap7797(dg, input_map_values, TEXTURE_ARGS);
		case 7803: return ReadInputMap7803(dg, input_map_values, TEXTURE_ARGS);
		case 7806: return ReadInputMap7806(dg, input_map_values, TEXTURE_ARGS);
		case 7807: return ReadInputMap7807(dg, input_map_values, TEXTURE_ARGS);
		case 7808: return ReadInputMap7808(dg, input_map_values, TEXTURE_ARGS);
		case 7809: return ReadInputMap7809(dg, input_map_values, TEXTURE_ARGS);
		case 7815: return ReadInputMap7815(dg, input_map_values, TEXTURE_ARGS);
		case 7818: return ReadInputMap7818(dg, input_map_values, TEXTURE_ARGS);
		case 7819: return ReadInputMap7819(dg, input_map_values, TEXTURE_ARGS);
		case 7820: return ReadInputMap7820(dg, input_map_values, TEXTURE_ARGS);
		case 7821: return ReadInputMap7821(dg, input_map_values, TEXTURE_ARGS);
		case 7822: return ReadInputMap7822(dg, input_map_values, TEXTURE_ARGS);
		case 7823: return ReadInputMap7823(dg, input_map_values, TEXTURE_ARGS);
		case 7824: return ReadInputMap7824(dg, input_map_values, TEXTURE_ARGS);
		case 7830: return ReadInputMap7830(dg, input_map_values, TEXTURE_ARGS);
		case 7833: return ReadInputMap7833(dg, input_map_values, TEXTURE_ARGS);
		case 7834: return ReadInputMap7834(dg, input_map_values, TEXTURE_ARGS);
		case 7835: return ReadInputMap7835(dg, input_map_values, TEXTURE_ARGS);
		case 7836: return ReadInputMap7836(dg, input_map_values, TEXTURE_ARGS);
		case 7837: return ReadInputMap7837(dg, input_map_values, TEXTURE_ARGS);
		case 7838: return ReadInputMap7838(dg, input_map_values, TEXTURE_ARGS);
		case 7839: return ReadInputMap7839(dg, input_map_values, TEXTURE_ARGS);
		case 7858: return ReadInputMap7858(dg, input_map_values, TEXTURE_ARGS);
		case 7859: return ReadInputMap7859(dg, input_map_values, TEXTURE_ARGS);
		case 7864: return ReadInputMap7864(dg, input_map_values, TEXTURE_ARGS);
		case 7867: return ReadInputMap7867(dg, input_map_values, TEXTURE_ARGS);
		case 7873: return ReadInputMap7873(dg, input_map_values, TEXTURE_ARGS);
		case 7874: return ReadInputMap7874(dg, input_map_values, TEXTURE_ARGS);
		case 7876: return ReadInputMap7876(dg, input_map_values, TEXTURE_ARGS);
		case 7877: return ReadInputMap7877(dg, input_map_values, TEXTURE_ARGS);
		case 7885: return ReadInputMap7885(dg, input_map_values, TEXTURE_ARGS);
		case 7886: return ReadInputMap7886(dg, input_map_values, TEXTURE_ARGS);
		case 7888: return ReadInputMap7888(dg, input_map_values, TEXTURE_ARGS);
		case 7889: return ReadInputMap7889(dg, input_map_values, TEXTURE_ARGS);
		case 7902: return ReadInputMap7902(dg, input_map_values, TEXTURE_ARGS);
		case 7903: return ReadInputMap7903(dg, input_map_values, TEXTURE_ARGS);
		case 7907: return ReadInputMap7907(dg, input_map_values, TEXTURE_ARGS);
		case 7916: return ReadInputMap7916(dg, input_map_values, TEXTURE_ARGS);
		case 7919: return ReadInputMap7919(dg, input_map_values, TEXTURE_ARGS);
		case 7920: return ReadInputMap7920(dg, input_map_values, TEXTURE_ARGS);
		case 7921: return ReadInputMap7921(dg, input_map_values, TEXTURE_ARGS);
		case 7925: return ReadInputMap7925(dg, input_map_values, TEXTURE_ARGS);
		case 7928: return ReadInputMap7928(dg, input_map_values, TEXTURE_ARGS);
		case 7932: return ReadInputMap7932(dg, input_map_values, TEXTURE_ARGS);
		case 7936: return ReadInputMap7936(dg, input_map_values, TEXTURE_ARGS);
		case 7942: return ReadInputMap7942(dg, input_map_values, TEXTURE_ARGS);
		case 7948: return ReadInputMap7948(dg, input_map_values, TEXTURE_ARGS);
		case 7954: return ReadInputMap7954(dg, input_map_values, TEXTURE_ARGS);
		case 7957: return ReadInputMap7957(dg, input_map_values, TEXTURE_ARGS);
		case 7958: return ReadInputMap7958(dg, input_map_values, TEXTURE_ARGS);
		case 7959: return ReadInputMap7959(dg, input_map_values, TEXTURE_ARGS);
		case 7972: return ReadInputMap7972(dg, input_map_values, TEXTURE_ARGS);
		case 7973: return ReadInputMap7973(dg, input_map_values, TEXTURE_ARGS);
		case 7974: return ReadInputMap7974(dg, input_map_values, TEXTURE_ARGS);
		case 7978: return ReadInputMap7978(dg, input_map_values, TEXTURE_ARGS);
		case 7984: return ReadInputMap7984(dg, input_map_values, TEXTURE_ARGS);
		case 7987: return ReadInputMap7987(dg, input_map_values, TEXTURE_ARGS);
		case 7988: return ReadInputMap7988(dg, input_map_values, TEXTURE_ARGS);
		case 7989: return ReadInputMap7989(dg, input_map_values, TEXTURE_ARGS);
		case 7993: return ReadInputMap7993(dg, input_map_values, TEXTURE_ARGS);
		case 7997: return ReadInputMap7997(dg, input_map_values, TEXTURE_ARGS);
		case 8000: return ReadInputMap8000(dg, input_map_values, TEXTURE_ARGS);
		case 8006: return ReadInputMap8006(dg, input_map_values, TEXTURE_ARGS);
		case 8018: return ReadInputMap8018(dg, input_map_values, TEXTURE_ARGS);
		case 8021: return ReadInputMap8021(dg, input_map_values, TEXTURE_ARGS);
		case 8022: return ReadInputMap8022(dg, input_map_values, TEXTURE_ARGS);
		case 8023: return ReadInputMap8023(dg, input_map_values, TEXTURE_ARGS);
		case 8027: return ReadInputMap8027(dg, input_map_values, TEXTURE_ARGS);
		case 8033: return ReadInputMap8033(dg, input_map_values, TEXTURE_ARGS);
		case 8034: return ReadInputMap8034(dg, input_map_values, TEXTURE_ARGS);
		case 8035: return ReadInputMap8035(dg, input_map_values, TEXTURE_ARGS);
		case 8039: return ReadInputMap8039(dg, input_map_values, TEXTURE_ARGS);
		case 8045: return ReadInputMap8045(dg, input_map_values, TEXTURE_ARGS);
		case 8048: return ReadInputMap8048(dg, input_map_values, TEXTURE_ARGS);
		case 8049: return ReadInputMap8049(dg, input_map_values, TEXTURE_ARGS);
		case 8050: return ReadInputMap8050(dg, input_map_values, TEXTURE_ARGS);
		case 8063: return ReadInputMap8063(dg, input_map_values, TEXTURE_ARGS);
		case 8069: return ReadInputMap8069(dg, input_map_values, TEXTURE_ARGS);
		case 8070: return ReadInputMap8070(dg, input_map_values, TEXTURE_ARGS);
		case 8075: return ReadInputMap8075(dg, input_map_values, TEXTURE_ARGS);
		case 8081: return ReadInputMap8081(dg, input_map_values, TEXTURE_ARGS);
		case 8084: return ReadInputMap8084(dg, input_map_values, TEXTURE_ARGS);
		case 8085: return ReadInputMap8085(dg, input_map_values, TEXTURE_ARGS);
		case 8086: return ReadInputMap8086(dg, input_map_values, TEXTURE_ARGS);
		case 8087: return ReadInputMap8087(dg, input_map_values, TEXTURE_ARGS);
		case 8090: return ReadInputMap8090(dg, input_map_values, TEXTURE_ARGS);
		case 8094: return ReadInputMap8094(dg, input_map_values, TEXTURE_ARGS);
		case 8098: return ReadInputMap8098(dg, input_map_values, TEXTURE_ARGS);
		case 8099: return ReadInputMap8099(dg, input_map_values, TEXTURE_ARGS);
		case 8131: return ReadInputMap8131(dg, input_map_values, TEXTURE_ARGS);
		case 8137: return ReadInputMap8137(dg, input_map_values, TEXTURE_ARGS);
		case 8138: return ReadInputMap8138(dg, input_map_values, TEXTURE_ARGS);
		case 8140: return ReadInputMap8140(dg, input_map_values, TEXTURE_ARGS);
		case 8141: return ReadInputMap8141(dg, input_map_values, TEXTURE_ARGS);
		case 8142: return ReadInputMap8142(dg, input_map_values, TEXTURE_ARGS);
		case 8143: return ReadInputMap8143(dg, input_map_values, TEXTURE_ARGS);
		case 8144: return ReadInputMap8144(dg, input_map_values, TEXTURE_ARGS);
		case 8148: return ReadInputMap8148(dg, input_map_values, TEXTURE_ARGS);
		case 8152: return ReadInputMap8152(dg, input_map_values, TEXTURE_ARGS);
		case 8155: return ReadInputMap8155(dg, input_map_values, TEXTURE_ARGS);
		case 8161: return ReadInputMap8161(dg, input_map_values, TEXTURE_ARGS);
		case 8162: return ReadInputMap8162(dg, input_map_values, TEXTURE_ARGS);
		case 8164: return ReadInputMap8164(dg, input_map_values, TEXTURE_ARGS);
		case 8165: return ReadInputMap8165(dg, input_map_values, TEXTURE_ARGS);
		case 8169: return ReadInputMap8169(dg, input_map_values, TEXTURE_ARGS);
		case 8175: return ReadInputMap8175(dg, input_map_values, TEXTURE_ARGS);
		case 8181: return ReadInputMap8181(dg, input_map_values, TEXTURE_ARGS);
		case 8190: return ReadInputMap8190(dg, input_map_values, TEXTURE_ARGS);
		case 8191: return ReadInputMap8191(dg, input_map_values, TEXTURE_ARGS);
		case 8192: return ReadInputMap8192(dg, input_map_values, TEXTURE_ARGS);
		case 8196: return ReadInputMap8196(dg, input_map_values, TEXTURE_ARGS);
		case 8197: return ReadInputMap8197(dg, input_map_values, TEXTURE_ARGS);
		case 8198: return ReadInputMap8198(dg, input_map_values, TEXTURE_ARGS);
		case 8202: return ReadInputMap8202(dg, input_map_values, TEXTURE_ARGS);
		case 8214: return ReadInputMap8214(dg, input_map_values, TEXTURE_ARGS);
		case 8220: return ReadInputMap8220(dg, input_map_values, TEXTURE_ARGS);
		case 8221: return ReadInputMap8221(dg, input_map_values, TEXTURE_ARGS);
		case 8226: return ReadInputMap8226(dg, input_map_values, TEXTURE_ARGS);
		case 8239: return ReadInputMap8239(dg, input_map_values, TEXTURE_ARGS);
		case 8240: return ReadInputMap8240(dg, input_map_values, TEXTURE_ARGS);
		case 8244: return ReadInputMap8244(dg, input_map_values, TEXTURE_ARGS);
		case 8250: return ReadInputMap8250(dg, input_map_values, TEXTURE_ARGS);
		case 8253: return ReadInputMap8253(dg, input_map_values, TEXTURE_ARGS);
		case 8254: return ReadInputMap8254(dg, input_map_values, TEXTURE_ARGS);
		case 8255: return ReadInputMap8255(dg, input_map_values, TEXTURE_ARGS);
		case 8256: return ReadInputMap8256(dg, input_map_values, TEXTURE_ARGS);
		case 8262: return ReadInputMap8262(dg, input_map_values, TEXTURE_ARGS);
		case 8265: return ReadInputMap8265(dg, input_map_values, TEXTURE_ARGS);
		case 8266: return ReadInputMap8266(dg, input_map_values, TEXTURE_ARGS);
		case 8267: return ReadInputMap8267(dg, input_map_values, TEXTURE_ARGS);
		case 8268: return ReadInputMap8268(dg, input_map_values, TEXTURE_ARGS);
		case 8269: return ReadInputMap8269(dg, input_map_values, TEXTURE_ARGS);
		case 8270: return ReadInputMap8270(dg, input_map_values, TEXTURE_ARGS);
		case 8271: return ReadInputMap8271(dg, input_map_values, TEXTURE_ARGS);
		case 8283: return ReadInputMap8283(dg, input_map_values, TEXTURE_ARGS);
		case 8289: return ReadInputMap8289(dg, input_map_values, TEXTURE_ARGS);
		case 8290: return ReadInputMap8290(dg, input_map_values, TEXTURE_ARGS);
		case 8295: return ReadInputMap8295(dg, input_map_values, TEXTURE_ARGS);
		case 8304: return ReadInputMap8304(dg, input_map_values, TEXTURE_ARGS);
		case 8305: return ReadInputMap8305(dg, input_map_values, TEXTURE_ARGS);
		case 8306: return ReadInputMap8306(dg, input_map_values, TEXTURE_ARGS);
		case 8310: return ReadInputMap8310(dg, input_map_values, TEXTURE_ARGS);
		case 8311: return ReadInputMap8311(dg, input_map_values, TEXTURE_ARGS);
		case 8331: return ReadInputMap8331(dg, input_map_values, TEXTURE_ARGS);
		case 8334: return ReadInputMap8334(dg, input_map_values, TEXTURE_ARGS);
		case 8335: return ReadInputMap8335(dg, input_map_values, TEXTURE_ARGS);
		case 8336: return ReadInputMap8336(dg, input_map_values, TEXTURE_ARGS);
		case 8340: return ReadInputMap8340(dg, input_map_values, TEXTURE_ARGS);
		case 8343: return ReadInputMap8343(dg, input_map_values, TEXTURE_ARGS);
		case 8349: return ReadInputMap8349(dg, input_map_values, TEXTURE_ARGS);
		case 8350: return ReadInputMap8350(dg, input_map_values, TEXTURE_ARGS);
		case 8360: return ReadInputMap8360(dg, input_map_values, TEXTURE_ARGS);
		case 8361: return ReadInputMap8361(dg, input_map_values, TEXTURE_ARGS);
		case 8363: return ReadInputMap8363(dg, input_map_values, TEXTURE_ARGS);
		case 8364: return ReadInputMap8364(dg, input_map_values, TEXTURE_ARGS);
		case 8368: return ReadInputMap8368(dg, input_map_values, TEXTURE_ARGS);
		case 8371: return ReadInputMap8371(dg, input_map_values, TEXTURE_ARGS);
		case 8380: return ReadInputMap8380(dg, input_map_values, TEXTURE_ARGS);
		case 8381: return ReadInputMap8381(dg, input_map_values, TEXTURE_ARGS);
		case 8383: return ReadInputMap8383(dg, input_map_values, TEXTURE_ARGS);
		case 8384: return ReadInputMap8384(dg, input_map_values, TEXTURE_ARGS);
		case 8388: return ReadInputMap8388(dg, input_map_values, TEXTURE_ARGS);
		case 8394: return ReadInputMap8394(dg, input_map_values, TEXTURE_ARGS);
		case 8395: return ReadInputMap8395(dg, input_map_values, TEXTURE_ARGS);
		case 8397: return ReadInputMap8397(dg, input_map_values, TEXTURE_ARGS);
		case 8398: return ReadInputMap8398(dg, input_map_values, TEXTURE_ARGS);
		case 8402: return ReadInputMap8402(dg, input_map_values, TEXTURE_ARGS);
		case 8403: return ReadInputMap8403(dg, input_map_values, TEXTURE_ARGS);
		case 8410: return ReadInputMap8410(dg, input_map_values, TEXTURE_ARGS);
		case 8416: return ReadInputMap8416(dg, input_map_values, TEXTURE_ARGS);
		case 8417: return ReadInputMap8417(dg, input_map_values, TEXTURE_ARGS);
		case 8419: return ReadInputMap8419(dg, input_map_values, TEXTURE_ARGS);
		case 8420: return ReadInputMap8420(dg, input_map_values, TEXTURE_ARGS);
		case 8421: return ReadInputMap8421(dg, input_map_values, TEXTURE_ARGS);
		case 8422: return ReadInputMap8422(dg, input_map_values, TEXTURE_ARGS);
		case 8423: return ReadInputMap8423(dg, input_map_values, TEXTURE_ARGS);
		case 8427: return ReadInputMap8427(dg, input_map_values, TEXTURE_ARGS);
		case 8430: return ReadInputMap8430(dg, input_map_values, TEXTURE_ARGS);
		case 8443: return ReadInputMap8443(dg, input_map_values, TEXTURE_ARGS);
		case 8457: return ReadInputMap8457(dg, input_map_values, TEXTURE_ARGS);
		case 8458: return ReadInputMap8458(dg, input_map_values, TEXTURE_ARGS);
		case 8470: return ReadInputMap8470(dg, input_map_values, TEXTURE_ARGS);
		case 8473: return ReadInputMap8473(dg, input_map_values, TEXTURE_ARGS);
		case 8474: return ReadInputMap8474(dg, input_map_values, TEXTURE_ARGS);
		case 8475: return ReadInputMap8475(dg, input_map_values, TEXTURE_ARGS);
		case 8476: return ReadInputMap8476(dg, input_map_values, TEXTURE_ARGS);
		case 8477: return ReadInputMap8477(dg, input_map_values, TEXTURE_ARGS);
		case 8478: return ReadInputMap8478(dg, input_map_values, TEXTURE_ARGS);
		case 8482: return ReadInputMap8482(dg, input_map_values, TEXTURE_ARGS);
		case 8485: return ReadInputMap8485(dg, input_map_values, TEXTURE_ARGS);
		case 8489: return ReadInputMap8489(dg, input_map_values, TEXTURE_ARGS);
		case 8493: return ReadInputMap8493(dg, input_map_values, TEXTURE_ARGS);
		case 8494: return ReadInputMap8494(dg, input_map_values, TEXTURE_ARGS);
		case 8498: return ReadInputMap8498(dg, input_map_values, TEXTURE_ARGS);
		case 8504: return ReadInputMap8504(dg, input_map_values, TEXTURE_ARGS);
		case 8505: return ReadInputMap8505(dg, input_map_values, TEXTURE_ARGS);
		case 8507: return ReadInputMap8507(dg, input_map_values, TEXTURE_ARGS);
		case 8508: return ReadInputMap8508(dg, input_map_values, TEXTURE_ARGS);
		case 8534: return ReadInputMap8534(dg, input_map_values, TEXTURE_ARGS);
		case 8548: return ReadInputMap8548(dg, input_map_values, TEXTURE_ARGS);
		case 8554: return ReadInputMap8554(dg, input_map_values, TEXTURE_ARGS);
		case 8557: return ReadInputMap8557(dg, input_map_values, TEXTURE_ARGS);
		case 8558: return ReadInputMap8558(dg, input_map_values, TEXTURE_ARGS);
		case 8559: return ReadInputMap8559(dg, input_map_values, TEXTURE_ARGS);
		case 8560: return ReadInputMap8560(dg, input_map_values, TEXTURE_ARGS);
		case 8561: return ReadInputMap8561(dg, input_map_values, TEXTURE_ARGS);
		case 8562: return ReadInputMap8562(dg, input_map_values, TEXTURE_ARGS);
		case 8566: return ReadInputMap8566(dg, input_map_values, TEXTURE_ARGS);
		case 8578: return ReadInputMap8578(dg, input_map_values, TEXTURE_ARGS);
		case 8582: return ReadInputMap8582(dg, input_map_values, TEXTURE_ARGS);
		case 8585: return ReadInputMap8585(dg, input_map_values, TEXTURE_ARGS);
		case 8597: return ReadInputMap8597(dg, input_map_values, TEXTURE_ARGS);
		case 8603: return ReadInputMap8603(dg, input_map_values, TEXTURE_ARGS);
		case 8604: return ReadInputMap8604(dg, input_map_values, TEXTURE_ARGS);
		case 8609: return ReadInputMap8609(dg, input_map_values, TEXTURE_ARGS);
		case 8615: return ReadInputMap8615(dg, input_map_values, TEXTURE_ARGS);
		case 8618: return ReadInputMap8618(dg, input_map_values, TEXTURE_ARGS);
		case 8619: return ReadInputMap8619(dg, input_map_values, TEXTURE_ARGS);
		case 8620: return ReadInputMap8620(dg, input_map_values, TEXTURE_ARGS);
		case 8634: return ReadInputMap8634(dg, input_map_values, TEXTURE_ARGS);
		case 8641: return ReadInputMap8641(dg, input_map_values, TEXTURE_ARGS);
		case 8642: return ReadInputMap8642(dg, input_map_values, TEXTURE_ARGS);
		case 8646: return ReadInputMap8646(dg, input_map_values, TEXTURE_ARGS);
		case 8649: return ReadInputMap8649(dg, input_map_values, TEXTURE_ARGS);
		case 8655: return ReadInputMap8655(dg, input_map_values, TEXTURE_ARGS);
		case 8656: return ReadInputMap8656(dg, input_map_values, TEXTURE_ARGS);
		case 8658: return ReadInputMap8658(dg, input_map_values, TEXTURE_ARGS);
		case 8659: return ReadInputMap8659(dg, input_map_values, TEXTURE_ARGS);
		case 8663: return ReadInputMap8663(dg, input_map_values, TEXTURE_ARGS);
		case 8675: return ReadInputMap8675(dg, input_map_values, TEXTURE_ARGS);
		case 8678: return ReadInputMap8678(dg, input_map_values, TEXTURE_ARGS);
		case 8682: return ReadInputMap8682(dg, input_map_values, TEXTURE_ARGS);
		case 8701: return ReadInputMap8701(dg, input_map_values, TEXTURE_ARGS);
		case 8702: return ReadInputMap8702(dg, input_map_values, TEXTURE_ARGS);
		case 8704: return ReadInputMap8704(dg, input_map_values, TEXTURE_ARGS);
		case 8705: return ReadInputMap8705(dg, input_map_values, TEXTURE_ARGS);
		case 8709: return ReadInputMap8709(dg, input_map_values, TEXTURE_ARGS);
		case 8715: return ReadInputMap8715(dg, input_map_values, TEXTURE_ARGS);
		case 8716: return ReadInputMap8716(dg, input_map_values, TEXTURE_ARGS);
		case 8718: return ReadInputMap8718(dg, input_map_values, TEXTURE_ARGS);
		case 8719: return ReadInputMap8719(dg, input_map_values, TEXTURE_ARGS);
		case 8723: return ReadInputMap8723(dg, input_map_values, TEXTURE_ARGS);
		case 8727: return ReadInputMap8727(dg, input_map_values, TEXTURE_ARGS);
		case 8731: return ReadInputMap8731(dg, input_map_values, TEXTURE_ARGS);
		case 8737: return ReadInputMap8737(dg, input_map_values, TEXTURE_ARGS);
		case 8738: return ReadInputMap8738(dg, input_map_values, TEXTURE_ARGS);
		case 8740: return ReadInputMap8740(dg, input_map_values, TEXTURE_ARGS);
		case 8741: return ReadInputMap8741(dg, input_map_values, TEXTURE_ARGS);
		case 8748: return ReadInputMap8748(dg, input_map_values, TEXTURE_ARGS);
		case 8751: return ReadInputMap8751(dg, input_map_values, TEXTURE_ARGS);
		case 8752: return ReadInputMap8752(dg, input_map_values, TEXTURE_ARGS);
		case 8753: return ReadInputMap8753(dg, input_map_values, TEXTURE_ARGS);
		case 8757: return ReadInputMap8757(dg, input_map_values, TEXTURE_ARGS);
		case 8767: return ReadInputMap8767(dg, input_map_values, TEXTURE_ARGS);
		case 8771: return ReadInputMap8771(dg, input_map_values, TEXTURE_ARGS);
		case 8778: return ReadInputMap8778(dg, input_map_values, TEXTURE_ARGS);
		case 8779: return ReadInputMap8779(dg, input_map_values, TEXTURE_ARGS);
		case 8781: return ReadInputMap8781(dg, input_map_values, TEXTURE_ARGS);
		case 8782: return ReadInputMap8782(dg, input_map_values, TEXTURE_ARGS);
		case 8792: return ReadInputMap8792(dg, input_map_values, TEXTURE_ARGS);
		case 8793: return ReadInputMap8793(dg, input_map_values, TEXTURE_ARGS);
		case 8795: return ReadInputMap8795(dg, input_map_values, TEXTURE_ARGS);
		case 8796: return ReadInputMap8796(dg, input_map_values, TEXTURE_ARGS);
		case 8803: return ReadInputMap8803(dg, input_map_values, TEXTURE_ARGS);
		case 8806: return ReadInputMap8806(dg, input_map_values, TEXTURE_ARGS);
		case 8807: return ReadInputMap8807(dg, input_map_values, TEXTURE_ARGS);
		case 8808: return ReadInputMap8808(dg, input_map_values, TEXTURE_ARGS);
		case 8809: return ReadInputMap8809(dg, input_map_values, TEXTURE_ARGS);
		case 8813: return ReadInputMap8813(dg, input_map_values, TEXTURE_ARGS);
		case 8817: return ReadInputMap8817(dg, input_map_values, TEXTURE_ARGS);
		case 8823: return ReadInputMap8823(dg, input_map_values, TEXTURE_ARGS);
		case 8824: return ReadInputMap8824(dg, input_map_values, TEXTURE_ARGS);
		case 8826: return ReadInputMap8826(dg, input_map_values, TEXTURE_ARGS);
		case 8827: return ReadInputMap8827(dg, input_map_values, TEXTURE_ARGS);
		case 8846: return ReadInputMap8846(dg, input_map_values, TEXTURE_ARGS);
		case 8849: return ReadInputMap8849(dg, input_map_values, TEXTURE_ARGS);
		case 8850: return ReadInputMap8850(dg, input_map_values, TEXTURE_ARGS);
		case 8851: return ReadInputMap8851(dg, input_map_values, TEXTURE_ARGS);
		case 8855: return ReadInputMap8855(dg, input_map_values, TEXTURE_ARGS);
		case 8861: return ReadInputMap8861(dg, input_map_values, TEXTURE_ARGS);
		case 8862: return ReadInputMap8862(dg, input_map_values, TEXTURE_ARGS);
		case 8864: return ReadInputMap8864(dg, input_map_values, TEXTURE_ARGS);
		case 8865: return ReadInputMap8865(dg, input_map_values, TEXTURE_ARGS);
		case 8869: return ReadInputMap8869(dg, input_map_values, TEXTURE_ARGS);
		case 8897: return ReadInputMap8897(dg, input_map_values, TEXTURE_ARGS);
		case 8898: return ReadInputMap8898(dg, input_map_values, TEXTURE_ARGS);
		case 8900: return ReadInputMap8900(dg, input_map_values, TEXTURE_ARGS);
		case 8901: return ReadInputMap8901(dg, input_map_values, TEXTURE_ARGS);
		case 8905: return ReadInputMap8905(dg, input_map_values, TEXTURE_ARGS);
		case 8909: return ReadInputMap8909(dg, input_map_values, TEXTURE_ARGS);
		case 8913: return ReadInputMap8913(dg, input_map_values, TEXTURE_ARGS);
		case 8919: return ReadInputMap8919(dg, input_map_values, TEXTURE_ARGS);
		case 8920: return ReadInputMap8920(dg, input_map_values, TEXTURE_ARGS);
		case 8922: return ReadInputMap8922(dg, input_map_values, TEXTURE_ARGS);
		case 8923: return ReadInputMap8923(dg, input_map_values, TEXTURE_ARGS);
		case 8927: return ReadInputMap8927(dg, input_map_values, TEXTURE_ARGS);
		case 8939: return ReadInputMap8939(dg, input_map_values, TEXTURE_ARGS);
		case 8945: return ReadInputMap8945(dg, input_map_values, TEXTURE_ARGS);
		case 8946: return ReadInputMap8946(dg, input_map_values, TEXTURE_ARGS);
		case 8948: return ReadInputMap8948(dg, input_map_values, TEXTURE_ARGS);
		case 8949: return ReadInputMap8949(dg, input_map_values, TEXTURE_ARGS);
		case 8953: return ReadInputMap8953(dg, input_map_values, TEXTURE_ARGS);
		case 8959: return ReadInputMap8959(dg, input_map_values, TEXTURE_ARGS);
		case 8969: return ReadInputMap8969(dg, input_map_values, TEXTURE_ARGS);
		case 8970: return ReadInputMap8970(dg, input_map_values, TEXTURE_ARGS);
		case 8972: return ReadInputMap8972(dg, input_map_values, TEXTURE_ARGS);
		case 8973: return ReadInputMap8973(dg, input_map_values, TEXTURE_ARGS);
		case 8977: return ReadInputMap8977(dg, input_map_values, TEXTURE_ARGS);
		case 8983: return ReadInputMap8983(dg, input_map_values, TEXTURE_ARGS);
		case 8984: return ReadInputMap8984(dg, input_map_values, TEXTURE_ARGS);
		case 8986: return ReadInputMap8986(dg, input_map_values, TEXTURE_ARGS);
		case 8987: return ReadInputMap8987(dg, input_map_values, TEXTURE_ARGS);
		case 8988: return ReadInputMap8988(dg, input_map_values, TEXTURE_ARGS);
		case 8989: return ReadInputMap8989(dg, input_map_values, TEXTURE_ARGS);
		case 8990: return ReadInputMap8990(dg, input_map_values, TEXTURE_ARGS);
		case 8994: return ReadInputMap8994(dg, input_map_values, TEXTURE_ARGS);
		case 9006: return ReadInputMap9006(dg, input_map_values, TEXTURE_ARGS);
		case 9009: return ReadInputMap9009(dg, input_map_values, TEXTURE_ARGS);
		case 9010: return ReadInputMap9010(dg, input_map_values, TEXTURE_ARGS);
		case 9011: return ReadInputMap9011(dg, input_map_values, TEXTURE_ARGS);
		case 9012: return ReadInputMap9012(dg, input_map_values, TEXTURE_ARGS);
		case 9016: return ReadInputMap9016(dg, input_map_values, TEXTURE_ARGS);
		case 9022: return ReadInputMap9022(dg, input_map_values, TEXTURE_ARGS);
		case 9025: return ReadInputMap9025(dg, input_map_values, TEXTURE_ARGS);
		case 9026: return ReadInputMap9026(dg, input_map_values, TEXTURE_ARGS);
		case 9027: return ReadInputMap9027(dg, input_map_values, TEXTURE_ARGS);
		case 9028: return ReadInputMap9028(dg, input_map_values, TEXTURE_ARGS);
		case 9029: return ReadInputMap9029(dg, input_map_values, TEXTURE_ARGS);
		case 9030: return ReadInputMap9030(dg, input_map_values, TEXTURE_ARGS);
		case 9042: return ReadInputMap9042(dg, input_map_values, TEXTURE_ARGS);
		case 9045: return ReadInputMap9045(dg, input_map_values, TEXTURE_ARGS);
		case 9046: return ReadInputMap9046(dg, input_map_values, TEXTURE_ARGS);
		case 9047: return ReadInputMap9047(dg, input_map_values, TEXTURE_ARGS);
		case 9054: return ReadInputMap9054(dg, input_map_values, TEXTURE_ARGS);
		case 9057: return ReadInputMap9057(dg, input_map_values, TEXTURE_ARGS);
		case 9058: return ReadInputMap9058(dg, input_map_values, TEXTURE_ARGS);
		case 9059: return ReadInputMap9059(dg, input_map_values, TEXTURE_ARGS);
		case 9063: return ReadInputMap9063(dg, input_map_values, TEXTURE_ARGS);
		case 9075: return ReadInputMap9075(dg, input_map_values, TEXTURE_ARGS);
		case 9079: return ReadInputMap9079(dg, input_map_values, TEXTURE_ARGS);
		case 9091: return ReadInputMap9091(dg, input_map_values, TEXTURE_ARGS);
		case 9097: return ReadInputMap9097(dg, input_map_values, TEXTURE_ARGS);
		case 9098: return ReadInputMap9098(dg, input_map_values, TEXTURE_ARGS);
		case 9100: return ReadInputMap9100(dg, input_map_values, TEXTURE_ARGS);
		case 9101: return ReadInputMap9101(dg, input_map_values, TEXTURE_ARGS);
		case 9105: return ReadInputMap9105(dg, input_map_values, TEXTURE_ARGS);
		case 9112: return ReadInputMap9112(dg, input_map_values, TEXTURE_ARGS);
		case 9113: return ReadInputMap9113(dg, input_map_values, TEXTURE_ARGS);
		case 9115: return ReadInputMap9115(dg, input_map_values, TEXTURE_ARGS);
		case 9116: return ReadInputMap9116(dg, input_map_values, TEXTURE_ARGS);
		case 9120: return ReadInputMap9120(dg, input_map_values, TEXTURE_ARGS);
		case 9123: return ReadInputMap9123(dg, input_map_values, TEXTURE_ARGS);
		case 9129: return ReadInputMap9129(dg, input_map_values, TEXTURE_ARGS);
		case 9132: return ReadInputMap9132(dg, input_map_values, TEXTURE_ARGS);
		case 9133: return ReadInputMap9133(dg, input_map_values, TEXTURE_ARGS);
		case 9134: return ReadInputMap9134(dg, input_map_values, TEXTURE_ARGS);
		case 9138: return ReadInputMap9138(dg, input_map_values, TEXTURE_ARGS);
		case 9142: return ReadInputMap9142(dg, input_map_values, TEXTURE_ARGS);
		case 9148: return ReadInputMap9148(dg, input_map_values, TEXTURE_ARGS);
		case 9149: return ReadInputMap9149(dg, input_map_values, TEXTURE_ARGS);
		case 9151: return ReadInputMap9151(dg, input_map_values, TEXTURE_ARGS);
		case 9152: return ReadInputMap9152(dg, input_map_values, TEXTURE_ARGS);
		case 9167: return ReadInputMap9167(dg, input_map_values, TEXTURE_ARGS);
		case 9173: return ReadInputMap9173(dg, input_map_values, TEXTURE_ARGS);
		case 9176: return ReadInputMap9176(dg, input_map_values, TEXTURE_ARGS);
		case 9177: return ReadInputMap9177(dg, input_map_values, TEXTURE_ARGS);
		case 9178: return ReadInputMap9178(dg, input_map_values, TEXTURE_ARGS);
		case 9182: return ReadInputMap9182(dg, input_map_values, TEXTURE_ARGS);
		case 9194: return ReadInputMap9194(dg, input_map_values, TEXTURE_ARGS);
		case 9200: return ReadInputMap9200(dg, input_map_values, TEXTURE_ARGS);
		case 9201: return ReadInputMap9201(dg, input_map_values, TEXTURE_ARGS);
		case 9203: return ReadInputMap9203(dg, input_map_values, TEXTURE_ARGS);
		case 9204: return ReadInputMap9204(dg, input_map_values, TEXTURE_ARGS);
		case 9208: return ReadInputMap9208(dg, input_map_values, TEXTURE_ARGS);
		case 9211: return ReadInputMap9211(dg, input_map_values, TEXTURE_ARGS);
		case 9218: return ReadInputMap9218(dg, input_map_values, TEXTURE_ARGS);
		case 9221: return ReadInputMap9221(dg, input_map_values, TEXTURE_ARGS);
		case 9222: return ReadInputMap9222(dg, input_map_values, TEXTURE_ARGS);
		case 9223: return ReadInputMap9223(dg, input_map_values, TEXTURE_ARGS);
		case 9227: return ReadInputMap9227(dg, input_map_values, TEXTURE_ARGS);
		case 9233: return ReadInputMap9233(dg, input_map_values, TEXTURE_ARGS);
		case 9234: return ReadInputMap9234(dg, input_map_values, TEXTURE_ARGS);
		case 9236: return ReadInputMap9236(dg, input_map_values, TEXTURE_ARGS);
		case 9237: return ReadInputMap9237(dg, input_map_values, TEXTURE_ARGS);
		case 9247: return ReadInputMap9247(dg, input_map_values, TEXTURE_ARGS);
		case 9248: return ReadInputMap9248(dg, input_map_values, TEXTURE_ARGS);
		case 9250: return ReadInputMap9250(dg, input_map_values, TEXTURE_ARGS);
		case 9251: return ReadInputMap9251(dg, input_map_values, TEXTURE_ARGS);
		case 9255: return ReadInputMap9255(dg, input_map_values, TEXTURE_ARGS);
		case 9258: return ReadInputMap9258(dg, input_map_values, TEXTURE_ARGS);
		case 9261: return ReadInputMap9261(dg, input_map_values, TEXTURE_ARGS);
		case 9262: return ReadInputMap9262(dg, input_map_values, TEXTURE_ARGS);
		case 9263: return ReadInputMap9263(dg, input_map_values, TEXTURE_ARGS);
		case 9267: return ReadInputMap9267(dg, input_map_values, TEXTURE_ARGS);
		case 9273: return ReadInputMap9273(dg, input_map_values, TEXTURE_ARGS);
		case 9274: return ReadInputMap9274(dg, input_map_values, TEXTURE_ARGS);
		case 9276: return ReadInputMap9276(dg, input_map_values, TEXTURE_ARGS);
		case 9277: return ReadInputMap9277(dg, input_map_values, TEXTURE_ARGS);
		case 9281: return ReadInputMap9281(dg, input_map_values, TEXTURE_ARGS);
		case 9290: return ReadInputMap9290(dg, input_map_values, TEXTURE_ARGS);
		case 9291: return ReadInputMap9291(dg, input_map_values, TEXTURE_ARGS);
		case 9292: return ReadInputMap9292(dg, input_map_values, TEXTURE_ARGS);
		case 9296: return ReadInputMap9296(dg, input_map_values, TEXTURE_ARGS);
		case 9299: return ReadInputMap9299(dg, input_map_values, TEXTURE_ARGS);
		case 9302: return ReadInputMap9302(dg, input_map_values, TEXTURE_ARGS);
		case 9305: return ReadInputMap9305(dg, input_map_values, TEXTURE_ARGS);
		case 9306: return ReadInputMap9306(dg, input_map_values, TEXTURE_ARGS);
		case 9307: return ReadInputMap9307(dg, input_map_values, TEXTURE_ARGS);
		case 9312: return ReadInputMap9312(dg, input_map_values, TEXTURE_ARGS);
		case 9315: return ReadInputMap9315(dg, input_map_values, TEXTURE_ARGS);
		case 9316: return ReadInputMap9316(dg, input_map_values, TEXTURE_ARGS);
		case 9317: return ReadInputMap9317(dg, input_map_values, TEXTURE_ARGS);
		case 9321: return ReadInputMap9321(dg, input_map_values, TEXTURE_ARGS);
		case 9327: return ReadInputMap9327(dg, input_map_values, TEXTURE_ARGS);
		case 9330: return ReadInputMap9330(dg, input_map_values, TEXTURE_ARGS);
		case 9331: return ReadInputMap9331(dg, input_map_values, TEXTURE_ARGS);
		case 9332: return ReadInputMap9332(dg, input_map_values, TEXTURE_ARGS);
		case 9336: return ReadInputMap9336(dg, input_map_values, TEXTURE_ARGS);
		case 9339: return ReadInputMap9339(dg, input_map_values, TEXTURE_ARGS);
		case 9342: return ReadInputMap9342(dg, input_map_values, TEXTURE_ARGS);
		case 9343: return ReadInputMap9343(dg, input_map_values, TEXTURE_ARGS);
		case 9344: return ReadInputMap9344(dg, input_map_values, TEXTURE_ARGS);
		case 9348: return ReadInputMap9348(dg, input_map_values, TEXTURE_ARGS);
		case 9351: return ReadInputMap9351(dg, input_map_values, TEXTURE_ARGS);
		case 9354: return ReadInputMap9354(dg, input_map_values, TEXTURE_ARGS);
		case 9365: return ReadInputMap9365(dg, input_map_values, TEXTURE_ARGS);
		case 9366: return ReadInputMap9366(dg, input_map_values, TEXTURE_ARGS);
		case 9368: return ReadInputMap9368(dg, input_map_values, TEXTURE_ARGS);
		case 9369: return ReadInputMap9369(dg, input_map_values, TEXTURE_ARGS);
		case 9378: return ReadInputMap9378(dg, input_map_values, TEXTURE_ARGS);
		case 9384: return ReadInputMap9384(dg, input_map_values, TEXTURE_ARGS);
		case 9385: return ReadInputMap9385(dg, input_map_values, TEXTURE_ARGS);
		case 9387: return ReadInputMap9387(dg, input_map_values, TEXTURE_ARGS);
		case 9388: return ReadInputMap9388(dg, input_map_values, TEXTURE_ARGS);
		case 9394: return ReadInputMap9394(dg, input_map_values, TEXTURE_ARGS);
		case 9397: return ReadInputMap9397(dg, input_map_values, TEXTURE_ARGS);
		case 9398: return ReadInputMap9398(dg, input_map_values, TEXTURE_ARGS);
		case 9399: return ReadInputMap9399(dg, input_map_values, TEXTURE_ARGS);
		case 9403: return ReadInputMap9403(dg, input_map_values, TEXTURE_ARGS);
		case 9415: return ReadInputMap9415(dg, input_map_values, TEXTURE_ARGS);
		case 9421: return ReadInputMap9421(dg, input_map_values, TEXTURE_ARGS);
		case 9422: return ReadInputMap9422(dg, input_map_values, TEXTURE_ARGS);
		case 9424: return ReadInputMap9424(dg, input_map_values, TEXTURE_ARGS);
		case 9425: return ReadInputMap9425(dg, input_map_values, TEXTURE_ARGS);
		case 9429: return ReadInputMap9429(dg, input_map_values, TEXTURE_ARGS);
		case 9435: return ReadInputMap9435(dg, input_map_values, TEXTURE_ARGS);
		case 9436: return ReadInputMap9436(dg, input_map_values, TEXTURE_ARGS);
		case 9438: return ReadInputMap9438(dg, input_map_values, TEXTURE_ARGS);
		case 9439: return ReadInputMap9439(dg, input_map_values, TEXTURE_ARGS);
		case 9443: return ReadInputMap9443(dg, input_map_values, TEXTURE_ARGS);
		case 9452: return ReadInputMap9452(dg, input_map_values, TEXTURE_ARGS);
		case 9458: return ReadInputMap9458(dg, input_map_values, TEXTURE_ARGS);
		case 9461: return ReadInputMap9461(dg, input_map_values, TEXTURE_ARGS);
		case 9462: return ReadInputMap9462(dg, input_map_values, TEXTURE_ARGS);
		case 9463: return ReadInputMap9463(dg, input_map_values, TEXTURE_ARGS);
		case 9464: return ReadInputMap9464(dg, input_map_values, TEXTURE_ARGS);
		case 9480: return ReadInputMap9480(dg, input_map_values, TEXTURE_ARGS);
		case 9486: return ReadInputMap9486(dg, input_map_values, TEXTURE_ARGS);
		case 9487: return ReadInputMap9487(dg, input_map_values, TEXTURE_ARGS);
		case 9489: return ReadInputMap9489(dg, input_map_values, TEXTURE_ARGS);
		case 9490: return ReadInputMap9490(dg, input_map_values, TEXTURE_ARGS);
		case 9505: return ReadInputMap9505(dg, input_map_values, TEXTURE_ARGS);
		case 9511: return ReadInputMap9511(dg, input_map_values, TEXTURE_ARGS);
		case 9512: return ReadInputMap9512(dg, input_map_values, TEXTURE_ARGS);
		case 9514: return ReadInputMap9514(dg, input_map_values, TEXTURE_ARGS);
		case 9515: return ReadInputMap9515(dg, input_map_values, TEXTURE_ARGS);
		case 9516: return ReadInputMap9516(dg, input_map_values, TEXTURE_ARGS);
		case 9517: return ReadInputMap9517(dg, input_map_values, TEXTURE_ARGS);
		case 9518: return ReadInputMap9518(dg, input_map_values, TEXTURE_ARGS);
		case 9522: return ReadInputMap9522(dg, input_map_values, TEXTURE_ARGS);
		case 9534: return ReadInputMap9534(dg, input_map_values, TEXTURE_ARGS);
		case 9543: return ReadInputMap9543(dg, input_map_values, TEXTURE_ARGS);
		case 9544: return ReadInputMap9544(dg, input_map_values, TEXTURE_ARGS);
		case 9546: return ReadInputMap9546(dg, input_map_values, TEXTURE_ARGS);
		case 9547: return ReadInputMap9547(dg, input_map_values, TEXTURE_ARGS);
		case 9551: return ReadInputMap9551(dg, input_map_values, TEXTURE_ARGS);
		case 9563: return ReadInputMap9563(dg, input_map_values, TEXTURE_ARGS);
		case 9566: return ReadInputMap9566(dg, input_map_values, TEXTURE_ARGS);
		case 9569: return ReadInputMap9569(dg, input_map_values, TEXTURE_ARGS);
		case 9572: return ReadInputMap9572(dg, input_map_values, TEXTURE_ARGS);
		case 9573: return ReadInputMap9573(dg, input_map_values, TEXTURE_ARGS);
		case 9574: return ReadInputMap9574(dg, input_map_values, TEXTURE_ARGS);
		case 9578: return ReadInputMap9578(dg, input_map_values, TEXTURE_ARGS);
		case 9584: return ReadInputMap9584(dg, input_map_values, TEXTURE_ARGS);
		case 9585: return ReadInputMap9585(dg, input_map_values, TEXTURE_ARGS);
		case 9587: return ReadInputMap9587(dg, input_map_values, TEXTURE_ARGS);
		case 9588: return ReadInputMap9588(dg, input_map_values, TEXTURE_ARGS);
		case 9597: return ReadInputMap9597(dg, input_map_values, TEXTURE_ARGS);
		case 9598: return ReadInputMap9598(dg, input_map_values, TEXTURE_ARGS);
		case 9600: return ReadInputMap9600(dg, input_map_values, TEXTURE_ARGS);
		case 9601: return ReadInputMap9601(dg, input_map_values, TEXTURE_ARGS);
		case 9605: return ReadInputMap9605(dg, input_map_values, TEXTURE_ARGS);
		case 9611: return ReadInputMap9611(dg, input_map_values, TEXTURE_ARGS);
		case 9612: return ReadInputMap9612(dg, input_map_values, TEXTURE_ARGS);
		case 9614: return ReadInputMap9614(dg, input_map_values, TEXTURE_ARGS);
		case 9615: return ReadInputMap9615(dg, input_map_values, TEXTURE_ARGS);
		case 9616: return ReadInputMap9616(dg, input_map_values, TEXTURE_ARGS);
		case 9617: return ReadInputMap9617(dg, input_map_values, TEXTURE_ARGS);
		case 9618: return ReadInputMap9618(dg, input_map_values, TEXTURE_ARGS);
		case 9622: return ReadInputMap9622(dg, input_map_values, TEXTURE_ARGS);
		case 9625: return ReadInputMap9625(dg, input_map_values, TEXTURE_ARGS);
		case 9631: return ReadInputMap9631(dg, input_map_values, TEXTURE_ARGS);
		case 9632: return ReadInputMap9632(dg, input_map_values, TEXTURE_ARGS);
		case 9634: return ReadInputMap9634(dg, input_map_values, TEXTURE_ARGS);
		case 9635: return ReadInputMap9635(dg, input_map_values, TEXTURE_ARGS);
		case 9646: return ReadInputMap9646(dg, input_map_values, TEXTURE_ARGS);
		case 9647: return ReadInputMap9647(dg, input_map_values, TEXTURE_ARGS);
		case 9649: return ReadInputMap9649(dg, input_map_values, TEXTURE_ARGS);
		case 9650: return ReadInputMap9650(dg, input_map_values, TEXTURE_ARGS);
		case 9654: return ReadInputMap9654(dg, input_map_values, TEXTURE_ARGS);
		case 9657: return ReadInputMap9657(dg, input_map_values, TEXTURE_ARGS);
		case 9663: return ReadInputMap9663(dg, input_map_values, TEXTURE_ARGS);
		case 9664: return ReadInputMap9664(dg, input_map_values, TEXTURE_ARGS);
		case 9666: return ReadInputMap9666(dg, input_map_values, TEXTURE_ARGS);
		case 9667: return ReadInputMap9667(dg, input_map_values, TEXTURE_ARGS);
		case 9671: return ReadInputMap9671(dg, input_map_values, TEXTURE_ARGS);
		case 9676: return ReadInputMap9676(dg, input_map_values, TEXTURE_ARGS);
		case 9679: return ReadInputMap9679(dg, input_map_values, TEXTURE_ARGS);
		case 9680: return ReadInputMap9680(dg, input_map_values, TEXTURE_ARGS);
		case 9681: return ReadInputMap9681(dg, input_map_values, TEXTURE_ARGS);
		case 9696: return ReadInputMap9696(dg, input_map_values, TEXTURE_ARGS);
		case 9708: return ReadInputMap9708(dg, input_map_values, TEXTURE_ARGS);
		case 9711: return ReadInputMap9711(dg, input_map_values, TEXTURE_ARGS);
		case 9714: return ReadInputMap9714(dg, input_map_values, TEXTURE_ARGS);
		case 9715: return ReadInputMap9715(dg, input_map_values, TEXTURE_ARGS);
		case 9716: return ReadInputMap9716(dg, input_map_values, TEXTURE_ARGS);
		case 9720: return ReadInputMap9720(dg, input_map_values, TEXTURE_ARGS);
		case 9726: return ReadInputMap9726(dg, input_map_values, TEXTURE_ARGS);
		case 9729: return ReadInputMap9729(dg, input_map_values, TEXTURE_ARGS);
		case 9730: return ReadInputMap9730(dg, input_map_values, TEXTURE_ARGS);
		case 9731: return ReadInputMap9731(dg, input_map_values, TEXTURE_ARGS);
		case 9735: return ReadInputMap9735(dg, input_map_values, TEXTURE_ARGS);
		case 9747: return ReadInputMap9747(dg, input_map_values, TEXTURE_ARGS);
		case 9756: return ReadInputMap9756(dg, input_map_values, TEXTURE_ARGS);
		case 9757: return ReadInputMap9757(dg, input_map_values, TEXTURE_ARGS);
		case 9762: return ReadInputMap9762(dg, input_map_values, TEXTURE_ARGS);
		case 9765: return ReadInputMap9765(dg, input_map_values, TEXTURE_ARGS);
		case 9773: return ReadInputMap9773(dg, input_map_values, TEXTURE_ARGS);
		case 9776: return ReadInputMap9776(dg, input_map_values, TEXTURE_ARGS);
		case 9777: return ReadInputMap9777(dg, input_map_values, TEXTURE_ARGS);
		case 9778: return ReadInputMap9778(dg, input_map_values, TEXTURE_ARGS);
		case 9779: return ReadInputMap9779(dg, input_map_values, TEXTURE_ARGS);
		case 9780: return ReadInputMap9780(dg, input_map_values, TEXTURE_ARGS);
		case 9781: return ReadInputMap9781(dg, input_map_values, TEXTURE_ARGS);
		case 9785: return ReadInputMap9785(dg, input_map_values, TEXTURE_ARGS);
		case 9786: return ReadInputMap9786(dg, input_map_values, TEXTURE_ARGS);
		case 9797: return ReadInputMap9797(dg, input_map_values, TEXTURE_ARGS);
		case 9800: return ReadInputMap9800(dg, input_map_values, TEXTURE_ARGS);
		case 9801: return ReadInputMap9801(dg, input_map_values, TEXTURE_ARGS);
		case 9802: return ReadInputMap9802(dg, input_map_values, TEXTURE_ARGS);
		case 9806: return ReadInputMap9806(dg, input_map_values, TEXTURE_ARGS);
		case 9812: return ReadInputMap9812(dg, input_map_values, TEXTURE_ARGS);
		case 9813: return ReadInputMap9813(dg, input_map_values, TEXTURE_ARGS);
		case 9818: return ReadInputMap9818(dg, input_map_values, TEXTURE_ARGS);
		case 9824: return ReadInputMap9824(dg, input_map_values, TEXTURE_ARGS);
		case 9825: return ReadInputMap9825(dg, input_map_values, TEXTURE_ARGS);
		case 9830: return ReadInputMap9830(dg, input_map_values, TEXTURE_ARGS);
		case 9836: return ReadInputMap9836(dg, input_map_values, TEXTURE_ARGS);
		case 9839: return ReadInputMap9839(dg, input_map_values, TEXTURE_ARGS);
		case 9840: return ReadInputMap9840(dg, input_map_values, TEXTURE_ARGS);
		case 9841: return ReadInputMap9841(dg, input_map_values, TEXTURE_ARGS);
		case 9845: return ReadInputMap9845(dg, input_map_values, TEXTURE_ARGS);
		case 9846: return ReadInputMap9846(dg, input_map_values, TEXTURE_ARGS);
		case 9850: return ReadInputMap9850(dg, input_map_values, TEXTURE_ARGS);
		case 9851: return ReadInputMap9851(dg, input_map_values, TEXTURE_ARGS);
		case 9855: return ReadInputMap9855(dg, input_map_values, TEXTURE_ARGS);
		case 9867: return ReadInputMap9867(dg, input_map_values, TEXTURE_ARGS);
		case 9873: return ReadInputMap9873(dg, input_map_values, TEXTURE_ARGS);
		case 9876: return ReadInputMap9876(dg, input_map_values, TEXTURE_ARGS);
		case 9877: return ReadInputMap9877(dg, input_map_values, TEXTURE_ARGS);
		case 9878: return ReadInputMap9878(dg, input_map_values, TEXTURE_ARGS);
		case 9882: return ReadInputMap9882(dg, input_map_values, TEXTURE_ARGS);
		case 9885: return ReadInputMap9885(dg, input_map_values, TEXTURE_ARGS);
		case 9888: return ReadInputMap9888(dg, input_map_values, TEXTURE_ARGS);
		case 9889: return ReadInputMap9889(dg, input_map_values, TEXTURE_ARGS);
		case 9890: return ReadInputMap9890(dg, input_map_values, TEXTURE_ARGS);
		case 9894: return ReadInputMap9894(dg, input_map_values, TEXTURE_ARGS);
		case 9900: return ReadInputMap9900(dg, input_map_values, TEXTURE_ARGS);
		case 9901: return ReadInputMap9901(dg, input_map_values, TEXTURE_ARGS);
		case 9903: return ReadInputMap9903(dg, input_map_values, TEXTURE_ARGS);
		case 9904: return ReadInputMap9904(dg, input_map_values, TEXTURE_ARGS);
		case 9908: return ReadInputMap9908(dg, input_map_values, TEXTURE_ARGS);
		case 9911: return ReadInputMap9911(dg, input_map_values, TEXTURE_ARGS);
		case 9914: return ReadInputMap9914(dg, input_map_values, TEXTURE_ARGS);
		case 9915: return ReadInputMap9915(dg, input_map_values, TEXTURE_ARGS);
		case 9916: return ReadInputMap9916(dg, input_map_values, TEXTURE_ARGS);
		case 9920: return ReadInputMap9920(dg, input_map_values, TEXTURE_ARGS);
		case 9923: return ReadInputMap9923(dg, input_map_values, TEXTURE_ARGS);
		case 9941: return ReadInputMap9941(dg, input_map_values, TEXTURE_ARGS);
		case 9944: return ReadInputMap9944(dg, input_map_values, TEXTURE_ARGS);
		case 9947: return ReadInputMap9947(dg, input_map_values, TEXTURE_ARGS);
		case 9948: return ReadInputMap9948(dg, input_map_values, TEXTURE_ARGS);
		case 9949: return ReadInputMap9949(dg, input_map_values, TEXTURE_ARGS);
		case 9967: return ReadInputMap9967(dg, input_map_values, TEXTURE_ARGS);
		case 9979: return ReadInputMap9979(dg, input_map_values, TEXTURE_ARGS);
		case 9982: return ReadInputMap9982(dg, input_map_values, TEXTURE_ARGS);
		case 9994: return ReadInputMap9994(dg, input_map_values, TEXTURE_ARGS);
		case 10006: return ReadInputMap10006(dg, input_map_values, TEXTURE_ARGS);
		case 10018: return ReadInputMap10018(dg, input_map_values, TEXTURE_ARGS);
		case 10030: return ReadInputMap10030(dg, input_map_values, TEXTURE_ARGS);
		case 10042: return ReadInputMap10042(dg, input_map_values, TEXTURE_ARGS);
		case 10054: return ReadInputMap10054(dg, input_map_values, TEXTURE_ARGS);
		case 10066: return ReadInputMap10066(dg, input_map_values, TEXTURE_ARGS);
		case 10082: return ReadInputMap10082(dg, input_map_values, TEXTURE_ARGS);
		case 10085: return ReadInputMap10085(dg, input_map_values, TEXTURE_ARGS);
		case 10104: return ReadInputMap10104(dg, input_map_values, TEXTURE_ARGS);
		case 10105: return ReadInputMap10105(dg, input_map_values, TEXTURE_ARGS);
		case 10116: return ReadInputMap10116(dg, input_map_values, TEXTURE_ARGS);
		case 10117: return ReadInputMap10117(dg, input_map_values, TEXTURE_ARGS);
		case 10119: return ReadInputMap10119(dg, input_map_values, TEXTURE_ARGS);
		case 10120: return ReadInputMap10120(dg, input_map_values, TEXTURE_ARGS);
		case 10121: return ReadInputMap10121(dg, input_map_values, TEXTURE_ARGS);
		case 10124: return ReadInputMap10124(dg, input_map_values, TEXTURE_ARGS);
		case 10125: return ReadInputMap10125(dg, input_map_values, TEXTURE_ARGS);
		case 10129: return ReadInputMap10129(dg, input_map_values, TEXTURE_ARGS);
		case 10135: return ReadInputMap10135(dg, input_map_values, TEXTURE_ARGS);
		case 10138: return ReadInputMap10138(dg, input_map_values, TEXTURE_ARGS);
		case 10139: return ReadInputMap10139(dg, input_map_values, TEXTURE_ARGS);
		case 10140: return ReadInputMap10140(dg, input_map_values, TEXTURE_ARGS);
		case 10144: return ReadInputMap10144(dg, input_map_values, TEXTURE_ARGS);
		case 10150: return ReadInputMap10150(dg, input_map_values, TEXTURE_ARGS);
		case 10153: return ReadInputMap10153(dg, input_map_values, TEXTURE_ARGS);
		case 10154: return ReadInputMap10154(dg, input_map_values, TEXTURE_ARGS);
		case 10155: return ReadInputMap10155(dg, input_map_values, TEXTURE_ARGS);
		case 10159: return ReadInputMap10159(dg, input_map_values, TEXTURE_ARGS);
		case 10165: return ReadInputMap10165(dg, input_map_values, TEXTURE_ARGS);
		case 10168: return ReadInputMap10168(dg, input_map_values, TEXTURE_ARGS);
		case 10169: return ReadInputMap10169(dg, input_map_values, TEXTURE_ARGS);
		case 10170: return ReadInputMap10170(dg, input_map_values, TEXTURE_ARGS);
		case 10174: return ReadInputMap10174(dg, input_map_values, TEXTURE_ARGS);
		case 10180: return ReadInputMap10180(dg, input_map_values, TEXTURE_ARGS);
		case 10183: return ReadInputMap10183(dg, input_map_values, TEXTURE_ARGS);
		case 10184: return ReadInputMap10184(dg, input_map_values, TEXTURE_ARGS);
		case 10185: return ReadInputMap10185(dg, input_map_values, TEXTURE_ARGS);
		case 10189: return ReadInputMap10189(dg, input_map_values, TEXTURE_ARGS);
		case 10208: return ReadInputMap10208(dg, input_map_values, TEXTURE_ARGS);
		case 10211: return ReadInputMap10211(dg, input_map_values, TEXTURE_ARGS);
		case 10212: return ReadInputMap10212(dg, input_map_values, TEXTURE_ARGS);
		case 10213: return ReadInputMap10213(dg, input_map_values, TEXTURE_ARGS);
		case 10217: return ReadInputMap10217(dg, input_map_values, TEXTURE_ARGS);
		case 10223: return ReadInputMap10223(dg, input_map_values, TEXTURE_ARGS);
		case 10224: return ReadInputMap10224(dg, input_map_values, TEXTURE_ARGS);
		case 10226: return ReadInputMap10226(dg, input_map_values, TEXTURE_ARGS);
		case 10227: return ReadInputMap10227(dg, input_map_values, TEXTURE_ARGS);
		case 10228: return ReadInputMap10228(dg, input_map_values, TEXTURE_ARGS);
		case 10229: return ReadInputMap10229(dg, input_map_values, TEXTURE_ARGS);
		case 10230: return ReadInputMap10230(dg, input_map_values, TEXTURE_ARGS);
		case 10234: return ReadInputMap10234(dg, input_map_values, TEXTURE_ARGS);
		case 10240: return ReadInputMap10240(dg, input_map_values, TEXTURE_ARGS);
		case 10243: return ReadInputMap10243(dg, input_map_values, TEXTURE_ARGS);
		case 10244: return ReadInputMap10244(dg, input_map_values, TEXTURE_ARGS);
		case 10245: return ReadInputMap10245(dg, input_map_values, TEXTURE_ARGS);
		case 10249: return ReadInputMap10249(dg, input_map_values, TEXTURE_ARGS);
		case 10255: return ReadInputMap10255(dg, input_map_values, TEXTURE_ARGS);
		case 10258: return ReadInputMap10258(dg, input_map_values, TEXTURE_ARGS);
		case 10259: return ReadInputMap10259(dg, input_map_values, TEXTURE_ARGS);
		case 10260: return ReadInputMap10260(dg, input_map_values, TEXTURE_ARGS);
		case 10264: return ReadInputMap10264(dg, input_map_values, TEXTURE_ARGS);
		case 10270: return ReadInputMap10270(dg, input_map_values, TEXTURE_ARGS);
		case 10273: return ReadInputMap10273(dg, input_map_values, TEXTURE_ARGS);
		case 10274: return ReadInputMap10274(dg, input_map_values, TEXTURE_ARGS);
		case 10275: return ReadInputMap10275(dg, input_map_values, TEXTURE_ARGS);
		case 10279: return ReadInputMap10279(dg, input_map_values, TEXTURE_ARGS);
		case 10285: return ReadInputMap10285(dg, input_map_values, TEXTURE_ARGS);
		case 10288: return ReadInputMap10288(dg, input_map_values, TEXTURE_ARGS);
		case 10289: return ReadInputMap10289(dg, input_map_values, TEXTURE_ARGS);
		case 10290: return ReadInputMap10290(dg, input_map_values, TEXTURE_ARGS);
		case 10294: return ReadInputMap10294(dg, input_map_values, TEXTURE_ARGS);
		case 10301: return ReadInputMap10301(dg, input_map_values, TEXTURE_ARGS);
		case 10307: return ReadInputMap10307(dg, input_map_values, TEXTURE_ARGS);
		case 10308: return ReadInputMap10308(dg, input_map_values, TEXTURE_ARGS);
		case 10309: return ReadInputMap10309(dg, input_map_values, TEXTURE_ARGS);
		case 10313: return ReadInputMap10313(dg, input_map_values, TEXTURE_ARGS);
		case 10317: return ReadInputMap10317(dg, input_map_values, TEXTURE_ARGS);
		case 10323: return ReadInputMap10323(dg, input_map_values, TEXTURE_ARGS);
		case 10326: return ReadInputMap10326(dg, input_map_values, TEXTURE_ARGS);
		case 10327: return ReadInputMap10327(dg, input_map_values, TEXTURE_ARGS);
		case 10328: return ReadInputMap10328(dg, input_map_values, TEXTURE_ARGS);
		case 10332: return ReadInputMap10332(dg, input_map_values, TEXTURE_ARGS);
		case 10338: return ReadInputMap10338(dg, input_map_values, TEXTURE_ARGS);
		case 10341: return ReadInputMap10341(dg, input_map_values, TEXTURE_ARGS);
		case 10342: return ReadInputMap10342(dg, input_map_values, TEXTURE_ARGS);
		case 10343: return ReadInputMap10343(dg, input_map_values, TEXTURE_ARGS);
		case 10347: return ReadInputMap10347(dg, input_map_values, TEXTURE_ARGS);
		case 10353: return ReadInputMap10353(dg, input_map_values, TEXTURE_ARGS);
		case 10356: return ReadInputMap10356(dg, input_map_values, TEXTURE_ARGS);
		case 10357: return ReadInputMap10357(dg, input_map_values, TEXTURE_ARGS);
		case 10358: return ReadInputMap10358(dg, input_map_values, TEXTURE_ARGS);
		case 10362: return ReadInputMap10362(dg, input_map_values, TEXTURE_ARGS);
		case 10368: return ReadInputMap10368(dg, input_map_values, TEXTURE_ARGS);
		case 10371: return ReadInputMap10371(dg, input_map_values, TEXTURE_ARGS);
		case 10372: return ReadInputMap10372(dg, input_map_values, TEXTURE_ARGS);
		case 10373: return ReadInputMap10373(dg, input_map_values, TEXTURE_ARGS);
		case 10377: return ReadInputMap10377(dg, input_map_values, TEXTURE_ARGS);
		case 10383: return ReadInputMap10383(dg, input_map_values, TEXTURE_ARGS);
		case 10386: return ReadInputMap10386(dg, input_map_values, TEXTURE_ARGS);
		case 10387: return ReadInputMap10387(dg, input_map_values, TEXTURE_ARGS);
		case 10388: return ReadInputMap10388(dg, input_map_values, TEXTURE_ARGS);
		case 10392: return ReadInputMap10392(dg, input_map_values, TEXTURE_ARGS);
		case 10398: return ReadInputMap10398(dg, input_map_values, TEXTURE_ARGS);
		case 10399: return ReadInputMap10399(dg, input_map_values, TEXTURE_ARGS);
		case 10401: return ReadInputMap10401(dg, input_map_values, TEXTURE_ARGS);
		case 10402: return ReadInputMap10402(dg, input_map_values, TEXTURE_ARGS);
		case 10406: return ReadInputMap10406(dg, input_map_values, TEXTURE_ARGS);
	}
	return 0.0f;
}
float GetInputMapFloat(uint input_id, DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values, TEXTURE_ARG_LIST)
{
	return GetInputMapFloat4(input_id, dg, input_map_values, TEXTURE_ARGS).x;
}
#endif



typedef struct _UberV2ShaderData
{
    float4 diffuse_color;
    float4 reflection_color;
    float4 coating_color;
    float4 refraction_color;
    float4 emission_color;
    float4 sss_absorption_color;
    float4 sss_scatter_color;
    float4 sss_subsurface_color;
    float4 shading_normal;

    float reflection_roughness;
    float reflection_anisotropy;
    float reflection_anisotropy_rotation;
    float reflection_ior;

    float reflection_metalness;
    float coating_ior;
    float refraction_roughness;
    float refraction_ior;

    float transparency;
    float sss_absorption_distance;
    float sss_scatter_distance;
    float sss_scatter_direction;

} UberV2ShaderData;

bool UberV2IsTransmissive(
    // Layers
    int layers
    )
{
    return (layers & (kTransparencyLayer | kRefractionLayer)) != 0;
}

float4 GetUberV2EmissionColor(
    // Material offset
    int offset,
    // Geometry
    DifferentialGeometry const* dg,
    // Values for input maps
    GLOBAL InputMapData const* restrict input_map_values,
    // Material attributes
    GLOBAL int const* restrict material_attributes,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return GetInputMapFloat4(material_attributes[offset+1], dg, input_map_values, TEXTURE_ARGS);
}

#ifndef BXDF_UBERV2_BRICKS
#define BXDF_UBERV2_BRICKS

// Utility functions for Uberv2
/// Calculates Fresnel for provided parameters. Swaps IORs if needed
float CalculateFresnel(
    // IORs
    float top_ior,
    float bottom_ior,
    // Angle between normal and incoming ray
    float ndotwi
)
{
    float etai = top_ior;
    float etat = bottom_ior;
    float cosi = ndotwi;

    // Revert normal and eta if needed
    if (cosi < 0.f)
    {
        float tmp = etai;
        etai = etat;
        etat = tmp;
        cosi = -cosi;
    }

    float eta = etai / etat;
    float sini2 = 1.f - cosi * cosi;
    float sint2 = eta * eta * sini2;
    float fresnel = 1.f;

    if (sint2 < 1.f)
    {
        float cost = native_sqrt(max(0.f, 1.f - sint2));
        fresnel = FresnelDielectric(etai, etat, cosi, cost);
    }

    return fresnel;
}

// Calucates Fresnel blend for two float3 values.
// F(top_ior, bottom_ior) * top_value + (1 - F(top_ior, bottom_ior) * bottom_value)
float3 Fresnel_Blend(
    // IORs
    float top_ior,
    float bottom_ior,
    // Values to blend
    float3 top_value,
    float3 bottom_value,
    // Incoming direction
    float3 wi
)
{
    float fresnel = CalculateFresnel(top_ior, bottom_ior, wi.y);
    return fresnel * top_value + (1.f - fresnel) * bottom_value;
}

// Calucates Fresnel blend for two float values.
// F(top_ior, bottom_ior) * top_value + (1 - F(top_ior, bottom_ior) * bottom_value)
float Fresnel_Blend_F(
    // IORs
    float top_ior,
    float bottom_ior,
    // Values to blend
    float top_value,
    float bottom_value,
    // Incoming direction
    float3 wi
)
{
    float fresnel = CalculateFresnel(top_ior, bottom_ior, wi.y);
    return fresnel * top_value + (1.f - fresnel) * bottom_value;
}

// Diffuse layer
float3 UberV2_Lambert_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return shader_data->diffuse_color.xyz / PI;
}

float UberV2_Lambert_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return fabs(wo.y) / PI;
}

/// Lambert BRDF sampling
float3 UberV2_Lambert_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const float3 kd = UberV2_Lambert_Evaluate(shader_data, wi, *wo, TEXTURE_ARGS);

    *wo = Sample_MapToHemisphere(sample, make_float3(0.f, 1.f, 0.f), 1.f);

    *pdf = fabs((*wo).y) / PI;

    return kd;
}

// Reflection/Coating
/*
Microfacet GGX
*/
// Distribution fucntion
float UberV2_MicrofacetDistribution_GGX_D(float roughness, float3 m)
{
    float ndotm = fabs(m.y);
    float ndotm2 = ndotm * ndotm;
    float sinmn = native_sqrt(1.f - clamp(ndotm * ndotm, 0.f, 1.f));
    float tanmn = ndotm > DENOM_EPS ? sinmn / ndotm : 0.f;
    float a2 = roughness * roughness;
    float denom = (PI * ndotm2 * ndotm2 * (a2 + tanmn * tanmn) * (a2 + tanmn * tanmn));
    return denom > DENOM_EPS ? (a2 / denom) : 1.f;
}

// PDF of the given direction
float UberV2_MicrofacetDistribution_GGX_GetPdf(
    // Halfway vector
    float3 m,
    // Rougness
    float roughness,
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    float mpdf = UberV2_MicrofacetDistribution_GGX_D(roughness, m) * fabs(m.y);
    // See Humphreys and Pharr for derivation
    float denom = (4.f * fabs(dot(wo, m)));

    return denom > DENOM_EPS ? mpdf / denom : 0.f;
}

// Sample the distribution
void UberV2_MicrofacetDistribution_GGX_SampleNormal(
    // Roughness
    float roughness,
    // Differential geometry
    UberV2ShaderData const* shader_data,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wh
)
{
    float r1 = sample.x;
    float r2 = sample.y;

    // Sample halfway vector first, then reflect wi around that
    float theta = atan2(roughness * native_sqrt(r1), native_sqrt(1.f - r1));
    float costheta = native_cos(theta);
    float sintheta = native_sin(theta);

    // phi = 2*PI*ksi2
    float phi = 2.f * PI * r2;
    float cosphi = native_cos(phi);
    float sinphi = native_sin(phi);

    // Calculate wh
    *wh = make_float3(sintheta * cosphi, costheta, sintheta * sinphi);
}

//
float UberV2_MicrofacetDistribution_GGX_G1(float roughness, float3 v, float3 m)
{
    float ndotv = fabs(v.y);
    float mdotv = fabs(dot(m, v));

    float sinnv = native_sqrt(1.f - clamp(ndotv * ndotv, 0.f, 1.f));
    float tannv = ndotv > DENOM_EPS ? sinnv / ndotv : 0.f;
    float a2 = roughness * roughness;
    return 2.f / (1.f + native_sqrt(1.f + a2 * tannv * tannv));
}

// Shadowing function also depends on microfacet distribution
float UberV2_MicrofacetDistribution_GGX_G(float roughness, float3 wi, float3 wo, float3 wh)
{
    return UberV2_MicrofacetDistribution_GGX_G1(roughness, wi, wh) * UberV2_MicrofacetDistribution_GGX_G1(roughness, wo, wh);
}

float3 UberV2_MicrofacetGGX_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST,
    float3 ks
)
{
    // Incident and reflected zenith angles
    float costhetao = fabs(wo.y);
    float costhetai = fabs(wi.y);

    // Calc halfway vector
    float3 wh = normalize(wi + wo);

    float denom = (4.f * costhetao * costhetai);

    return denom > DENOM_EPS ? ks * UberV2_MicrofacetDistribution_GGX_G(shader_data->reflection_roughness, wi, wo, wh) * UberV2_MicrofacetDistribution_GGX_D(shader_data->reflection_roughness, wh) / denom : 0.f;
}


float UberV2_MicrofacetGGX_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    float3 wh = normalize(wo + wi);

    return UberV2_MicrofacetDistribution_GGX_GetPdf(wh, shader_data->reflection_roughness, shader_data, wi, wo, TEXTURE_ARGS);
}

float3 UberV2_MicrofacetGGX_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf,
    float3 ks
)
{
    float3 wh;
    UberV2_MicrofacetDistribution_GGX_SampleNormal(shader_data->reflection_roughness, shader_data, TEXTURE_ARGS, sample, &wh);

    *wo = -wi + 2.f*fabs(dot(wi, wh)) * wh;

    *pdf = UberV2_MicrofacetDistribution_GGX_GetPdf(wh, shader_data->reflection_roughness, shader_data, wi, *wo, TEXTURE_ARGS);

    return UberV2_MicrofacetGGX_Evaluate(shader_data, wi, *wo, TEXTURE_ARGS, ks);
}

/*
Ideal reflection BRDF
*/
float3 UberV2_IdealReflect_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

float UberV2_IdealReflect_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

float3 UberV2_IdealReflect_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf,
    float3 ks)
{
    // Mirror reflect wi
    *wo = normalize(make_float3(-wi.x, wi.y, -wi.z));

    // PDF is infinite at that point, but deltas are going to cancel out while evaluating
    // so set it to 1.f
    *pdf = 1.f;

    float coswo = fabs((*wo).y);

    // Return reflectance value
    return coswo > DENOM_EPS ? (ks * (1.f / coswo)) : 0.f;
}

// Refraction
/*
Ideal refraction BTDF
*/

float3 UberV2_IdealRefract_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

float UberV2_IdealRefract_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

float3 UberV2_IdealRefract_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const float3 ks = shader_data->refraction_color.xyz;

    float etai = 1.f;
    float etat = shader_data->refraction_ior;
    float cosi = wi.y;

    bool entering = cosi > 0.f;

    // Revert normal and eta if needed
    if (!entering)
    {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    float eta = etai / etat;
    float sini2 = 1.f - cosi * cosi;
    float sint2 = eta * eta * sini2;

    if (sint2 >= 1.f)
    {
        *pdf = 0.f;
        return 0.f;
    }

    float cost = native_sqrt(max(0.f, 1.f - sint2));

    // Transmitted ray
    *wo = normalize(make_float3(eta * -wi.x, entering ? -cost : cost, eta * -wi.z));

    *pdf = 1.f;

    return cost > DENOM_EPS ? (eta * eta * ks / cost) : 0.f;
}


float3 UberV2_MicrofacetRefractionGGX_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const float3 ks = shader_data->refraction_color.xyz;
    const float roughness = max(shader_data->refraction_roughness, ROUGHNESS_EPS);

    float ndotwi = wi.y;
    float ndotwo = wo.y;

    if (ndotwi * ndotwo >= 0.f)
    {
        return 0.f;
    }

    float etai = 1.f;
    float etat = shader_data->refraction_ior;

    // Revert normal and eta if needed
    if (ndotwi < 0.f)
    {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    // Calc halfway vector
    float3 ht = -(etai * wi + etat * wo);
    float3 wh = normalize(ht);

    float widotwh = fabs(dot(wh, wi));
    float wodotwh = fabs(dot(wh, wo));

    float denom = dot(ht, ht);
    denom *= (fabs(ndotwi) * fabs(ndotwo));

    return denom > DENOM_EPS ? (ks * (widotwh * wodotwh)  * (etat)* (etat)*
        UberV2_MicrofacetDistribution_GGX_G(roughness, wi, wo, wh) * UberV2_MicrofacetDistribution_GGX_D(roughness, wh) / denom) : 0.f;
}

float UberV2_MicrofacetRefractionGGX_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const float roughness = max(shader_data->refraction_roughness, ROUGHNESS_EPS);

    float ndotwi = wi.y;
    float ndotwo = wo.y;

    if (ndotwi * ndotwo >= 0.f)
    {
        return 0.f;
    }

    float etai = 1.f;
    float etat = shader_data->refraction_ior;

    // Revert normal and eta if needed
    if (ndotwi < 0.f)
    {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    // Calc halfway vector
    float3 ht = -(etai * wi + etat * wo);

    float3 wh = normalize(ht);

    float wodotwh = fabs(dot(wo, wh));

    float whpdf = UberV2_MicrofacetDistribution_GGX_D(roughness, wh) * fabs(wh.y);

    float whwo = wodotwh * etat * etat;

    float denom = dot(ht, ht);

    return denom > DENOM_EPS ? whpdf * whwo / denom : 0.f;
}

float3 UberV2_MicrofacetRefractionGGX_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const float3 ks = shader_data->refraction_color.xyz;
    const float roughness = max(shader_data->refraction_roughness, ROUGHNESS_EPS);

    float ndotwi = wi.y;

    if (ndotwi == 0.f)
    {
        *pdf = 0.f;
        return 0.f;
    }

    float etai = 1.f;
    float etat = shader_data->refraction_ior;
    float s = 1.f;

    // Revert normal and eta if needed
    if (ndotwi < 0.f)
    {
        float tmp = etai;
        etai = etat;
        etat = tmp;
        s = -s;
    }

    float3 wh;
    UberV2_MicrofacetDistribution_GGX_SampleNormal(roughness, shader_data, TEXTURE_ARGS, sample, &wh);

    float c = dot(wi, wh);
    float eta = etai / etat;

    float d = 1 + eta * (c * c - 1);

    if (d <= 0.f)
    {
        *pdf = 0.f;
        return 0.f;
    }

    *wo = normalize((eta * c - s * native_sqrt(d)) * wh - eta * wi);

    *pdf = UberV2_MicrofacetRefractionGGX_GetPdf(shader_data, wi, *wo, TEXTURE_ARGS);

    return UberV2_MicrofacetRefractionGGX_Evaluate(shader_data, wi, *wo, TEXTURE_ARGS);
}

float3 UberV2_Passthrough_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    *wo = -wi;
    float coswo = fabs((*wo).y);

    // PDF is infinite at that point, but deltas are going to cancel out while evaluating
    // so set it to 1.f
    *pdf = 1.f;

    return coswo > 1e-5f ? (1.f / coswo) : 0.f;
}

float3 UberV2_Reflection_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const bool is_singular = (shader_data->reflection_roughness < ROUGHNESS_EPS);
    const float metalness = shader_data->reflection_metalness;

    const float3 ks = shader_data->reflection_color.xyz;

    float3 color = mix((float3)(1.0f, 1.0f, 1.0f), ks, metalness);

    return is_singular ?
        UberV2_IdealReflect_Evaluate(shader_data, wi, wo, TEXTURE_ARGS) :
        UberV2_MicrofacetGGX_Evaluate(shader_data, wi, wo, TEXTURE_ARGS, color);
}

float3 UberV2_Refraction_Evaluate(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const bool is_singular = (shader_data->refraction_roughness < ROUGHNESS_EPS);

    return is_singular ?
        UberV2_IdealRefract_Evaluate(shader_data, wi, wo, TEXTURE_ARGS) :
        UberV2_MicrofacetRefractionGGX_Evaluate(shader_data, wi, wo, TEXTURE_ARGS);
}

float UberV2_Reflection_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const bool is_singular = (shader_data->reflection_roughness < ROUGHNESS_EPS);

    return is_singular ?
        UberV2_IdealReflect_GetPdf(shader_data, wi, wo, TEXTURE_ARGS) :
        UberV2_MicrofacetGGX_GetPdf(shader_data, wi, wo, TEXTURE_ARGS);
}

float UberV2_Refraction_GetPdf(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Outgoing direction
    float3 wo,
    // Texture args
    TEXTURE_ARG_LIST
)
{
    const bool is_singular = (shader_data->refraction_roughness < ROUGHNESS_EPS);

    return is_singular ?
        UberV2_IdealRefract_GetPdf(shader_data, wi, wo, TEXTURE_ARGS) :
        UberV2_MicrofacetRefractionGGX_GetPdf(shader_data, wi, wo, TEXTURE_ARGS);
}

float3 UberV2_Reflection_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const float3 ks = shader_data->reflection_color.xyz;
    const bool is_singular = (shader_data->reflection_roughness < ROUGHNESS_EPS);
    const float metalness = shader_data->reflection_metalness;

    float3 color = mix((float3)(1.0f, 1.0f, 1.0f), ks, metalness);

    return is_singular ?
        UberV2_IdealReflect_Sample(shader_data, wi, TEXTURE_ARGS, wo, pdf, color) :
        UberV2_MicrofacetGGX_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf, color);
}

float3 UberV2_Refraction_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const bool is_singular = (shader_data->refraction_roughness < ROUGHNESS_EPS);

    return is_singular ?
        UberV2_IdealRefract_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf) :
        UberV2_MicrofacetRefractionGGX_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
}

float3 UberV2_Coating_Sample(
    // Preprocessed shader input data
    UberV2ShaderData const* shader_data,
    // Incoming direction
    float3 wi,
    // Texture args
    TEXTURE_ARG_LIST,
    // Outgoing  direction
    float3* wo,
    // PDF at wo
    float* pdf
)
{
    const float3 ks = shader_data->coating_color.xyz;

    return UberV2_IdealReflect_Sample(shader_data, wi, TEXTURE_ARGS, wo, pdf, ks);
}

#endif


void UberV2_ApplyShadingNormal(
    // Geometry
    DifferentialGeometry* dg,
    // Prepared UberV2 shader inputs
    UberV2ShaderData const* shader_data
)
{
    const int layers = dg->mat.layers;

    if ((layers & kShadingNormalLayer) == kShadingNormalLayer)
    {
        dg->n = normalize(shader_data->shading_normal.z * dg->n + shader_data->shading_normal.x * dg->dpdu + shader_data->shading_normal.y * dg->dpdv);
        dg->dpdv = normalize(cross(dg->n, dg->dpdu));
        dg->dpdu = normalize(cross(dg->dpdv, dg->n));
    }
}

#endif


/// Emissive BRDF sampling
float3 Emissive_GetLe(
    // Geometry
    DifferentialGeometry const* dg,
    // Texture args
    TEXTURE_ARG_LIST,
    UberV2ShaderData const* shader_data)
{
    return shader_data->emission_color.xyz;
}

/// BxDF singularity check
bool Bxdf_IsSingular(DifferentialGeometry const* dg)
{
    return (dg->mat.flags & kBxdfFlagsSingular) == kBxdfFlagsSingular;
}

/// BxDF emission check
bool Bxdf_IsEmissive(DifferentialGeometry const* dg)
{
    return (dg->mat.flags & kBxdfFlagsEmissive) == kBxdfFlagsEmissive;
}

bool Bxdf_IsBtdf(DifferentialGeometry const* dg)
{
    return (dg->mat.flags & kBxdfFlagsBrdf) == 0;
}

bool Bxdf_IsReflection(DifferentialGeometry const* dg)
{
    return ((dg->mat.flags & kBxdfFlagsBrdf) == kBxdfFlagsBrdf) && ((dg->mat.flags & kBxdfFlagsDiffuse) == kBxdfFlagsDiffuse);
}

bool Bxdf_IsTransparency(DifferentialGeometry const* dg)
{
    return (dg->mat.flags & kBxdfFlagsTransparency) == kBxdfFlagsTransparency;
}

bool Bxdf_IsRefraction(DifferentialGeometry const* dg)
{
    return Bxdf_IsBtdf(dg) && !Bxdf_IsTransparency(dg);
}
void GetMaterialBxDFType8(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
		if (shader_data->reflection_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleReflection);
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate8(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->reflection_ior, UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

float UberV2_GetPdf8(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->reflection_ior, UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

void UberV2PrepareInputs8(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->reflection_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy_rotation = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_metalness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample8(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleReflection: result = UberV2_Reflection_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf8(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate8(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType16(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate16(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS));
}

float UberV2_GetPdf16(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS));
}

void UberV2PrepareInputs16(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample16(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf16(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate16(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType24(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
	const float fresnel2 = CalculateFresnel(top_ior, shader_data->reflection_ior, ndotwi);
	const float sample4 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	if (sample4 < fresnel2)
	{
		if (shader_data->reflection_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleReflection);
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate24(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->reflection_ior, UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

float UberV2_GetPdf24(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->reflection_ior, UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

void UberV2PrepareInputs24(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->reflection_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy_rotation = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_metalness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample24(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleReflection: result = UberV2_Reflection_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf24(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate24(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType40(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	const float sample1 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	const float fresnel = CalculateFresnel(1.0f, shader_data->refraction_ior, ndotwi);
	if (sample1 >= fresnel)
	{
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleRefraction);
		if (shader_data->refraction_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
		if (shader_data->reflection_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleReflection);
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate40(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->refraction_ior, (Fresnel_Blend(1.0f, shader_data->reflection_ior, UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), wi)), UberV2_Refraction_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

float UberV2_GetPdf40(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->refraction_ior, (Fresnel_Blend_F(1.0f, shader_data->reflection_ior, UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), wi)), UberV2_Refraction_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

void UberV2PrepareInputs40(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->reflection_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy_rotation = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_metalness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample40(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleReflection: result = UberV2_Reflection_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleRefraction: result = UberV2_Refraction_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf40(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate40(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType48(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	const float sample1 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	const float fresnel = CalculateFresnel(1.0f, shader_data->refraction_ior, ndotwi);
	if (sample1 >= fresnel)
	{
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleRefraction);
		if (shader_data->refraction_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate48(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->refraction_ior, (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), UberV2_Refraction_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

float UberV2_GetPdf48(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->refraction_ior, (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), UberV2_Refraction_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

void UberV2PrepareInputs48(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample48(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleRefraction: result = UberV2_Refraction_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf48(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate48(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType56(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	const float sample1 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	const float fresnel = CalculateFresnel(1.0f, shader_data->refraction_ior, ndotwi);
	if (sample1 >= fresnel)
	{
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleRefraction);
		if (shader_data->refraction_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
	const float fresnel2 = CalculateFresnel(top_ior, shader_data->reflection_ior, ndotwi);
	const float sample4 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	if (sample4 < fresnel2)
	{
		if (shader_data->reflection_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleReflection);
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate56(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->refraction_ior, (Fresnel_Blend(1.0f, shader_data->reflection_ior, UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), wi)), UberV2_Refraction_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

float UberV2_GetPdf56(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->refraction_ior, (Fresnel_Blend_F(1.0f, shader_data->reflection_ior, UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), wi)), UberV2_Refraction_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), wi));
}

void UberV2PrepareInputs56(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->reflection_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy_rotation = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_metalness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->refraction_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample56(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleReflection: result = UberV2_Reflection_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleRefraction: result = UberV2_Refraction_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf56(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate56(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType144(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate144(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS));
}

float UberV2_GetPdf144(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS));
}

void UberV2PrepareInputs144(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->shading_normal = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample144(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf144(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate144(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void GetMaterialBxDFType152(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data)
{
	int bxdf_flags = 0;
	const float ndotwi = dot(dg->n, wi);
	bxdf_flags |= kBxdfFlagsBrdf;
	float top_ior = 1.0f;
	const float fresnel2 = CalculateFresnel(top_ior, shader_data->reflection_ior, ndotwi);
	const float sample4 = Sampler_Sample1D(sampler, SAMPLER_ARGS);
	if (sample4 < fresnel2)
	{
		if (shader_data->reflection_roughness < ROUGHNESS_EPS)
		{
			bxdf_flags |= kBxdfFlagsSingular;
		}
		Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleReflection);
		Bxdf_SetFlags(dg, bxdf_flags);
		return;
	}
	Bxdf_UberV2_SetSampledComponent(dg, kBxdfUberV2SampleDiffuse);
	bxdf_flags |= kBxdfFlagsBrdf;
	Bxdf_SetFlags(dg, bxdf_flags);
return;
}

float3 UberV2_Evaluate152(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend(1.0f, shader_data->reflection_ior, UberV2_Reflection_Evaluate(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_Evaluate(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

float UberV2_GetPdf152(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	return (Fresnel_Blend_F(1.0f, shader_data->reflection_ior, UberV2_Reflection_GetPdf(shader_data, wi, wo, TEXTURE_ARGS), (UberV2_Lambert_GetPdf(shader_data, wi, wo, TEXTURE_ARGS)), wi));
}

void UberV2PrepareInputs152(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *data)
{
	int offset = dg->mat.offset + 1;
	data->reflection_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_roughness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_anisotropy_rotation = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_ior = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->reflection_metalness = GetInputMapFloat(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->diffuse_color = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);
	data->shading_normal = GetInputMapFloat4(material_attributes[offset++], dg, input_map_values, TEXTURE_ARGS);

}
float3 UberV2_Sample152(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3* wo, float* pdf,UberV2ShaderData const* shader_data)
{
	const int sampledComponent = Bxdf_UberV2_GetSampledComponent(dg);
	float3 result;
	switch(sampledComponent)
	{
		case kBxdfUberV2SampleReflection: result = UberV2_Reflection_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
		case kBxdfUberV2SampleDiffuse: result = UberV2_Lambert_Sample(shader_data, wi, TEXTURE_ARGS, sample, wo, pdf);
			break;
	}
	if (false)
	{
		*pdf = UberV2_GetPdf152(dg, wi, *wo, TEXTURE_ARGS, shader_data);
		return UberV2_Evaluate152(dg, wi, *wo, TEXTURE_ARGS, shader_data);
	}
	return result;
}

void UberV2PrepareInputs(DifferentialGeometry const* dg, GLOBAL InputMapData const* restrict input_map_values,GLOBAL int const* restrict material_attributes, TEXTURE_ARG_LIST, UberV2ShaderData *shader_data)
{
	switch(dg->mat.layers)
	{
		case 8:
			return UberV2PrepareInputs8(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 16:
			return UberV2PrepareInputs16(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 24:
			return UberV2PrepareInputs24(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 40:
			return UberV2PrepareInputs40(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 48:
			return UberV2PrepareInputs48(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 56:
			return UberV2PrepareInputs56(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 144:
			return UberV2PrepareInputs144(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
		case 152:
			return UberV2PrepareInputs152(dg, input_map_values, material_attributes, TEXTURE_ARGS, shader_data);
	}
}
void GetMaterialBxDFType(float3 wi, Sampler* sampler, SAMPLER_ARG_LIST, DifferentialGeometry* dg, UberV2ShaderData const* shader_data){
	switch(dg->mat.layers)
	{
		case 8:
			return GetMaterialBxDFType8(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 16:
			return GetMaterialBxDFType16(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 24:
			return GetMaterialBxDFType24(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 40:
			return GetMaterialBxDFType40(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 48:
			return GetMaterialBxDFType48(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 56:
			return GetMaterialBxDFType56(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 144:
			return GetMaterialBxDFType144(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
		case 152:
			return GetMaterialBxDFType152(wi, sampler, SAMPLER_ARGS, dg, shader_data);
			break;
	}
}
float UberV2_GetPdf(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	float3 wi_t = matrix_mul_vector3(dg->world_to_tangent, wi);
	float3 wo_t = matrix_mul_vector3(dg->world_to_tangent, wo);
	switch (dg->mat.layers)
	{
		case 8:
			return UberV2_GetPdf8(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 16:
			return UberV2_GetPdf16(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 24:
			return UberV2_GetPdf24(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 40:
			return UberV2_GetPdf40(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 48:
			return UberV2_GetPdf48(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 56:
			return UberV2_GetPdf56(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 144:
			return UberV2_GetPdf144(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 152:
			return UberV2_GetPdf152(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
	}
	return 0.0f;
}
float3 UberV2_Evaluate(DifferentialGeometry const* dg, float3 wi, float3 wo, TEXTURE_ARG_LIST, UberV2ShaderData const* shader_data)
{
	float3 wi_t = matrix_mul_vector3(dg->world_to_tangent, wi);
	float3 wo_t = matrix_mul_vector3(dg->world_to_tangent, wo);
	switch (dg->mat.layers)
	{
		case 8:
			return UberV2_Evaluate8(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 16:
			return UberV2_Evaluate16(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 24:
			return UberV2_Evaluate24(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 40:
			return UberV2_Evaluate40(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 48:
			return UberV2_Evaluate48(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 56:
			return UberV2_Evaluate56(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 144:
			return UberV2_Evaluate144(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
		case 152:
			return UberV2_Evaluate152(dg, wi_t, wo_t, TEXTURE_ARGS, shader_data);
	}
	return (float3)(0.0f);
}
float3 UberV2_Sample(DifferentialGeometry const* dg, float3 wi, TEXTURE_ARG_LIST, float2 sample, float3 *wo, float *pdf, UberV2ShaderData const* shader_data)
{
	float3 wi_t = matrix_mul_vector3(dg->world_to_tangent, wi);
	float3 wo_t;
	float3 res = 0.f;
	switch (dg->mat.layers)
	{
		case 8:
			res = UberV2_Sample8(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 16:
			res = UberV2_Sample16(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 24:
			res = UberV2_Sample24(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 40:
			res = UberV2_Sample40(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 48:
			res = UberV2_Sample48(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 56:
			res = UberV2_Sample56(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 144:
			res = UberV2_Sample144(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
		case 152:
			res = UberV2_Sample152(dg, wi_t, TEXTURE_ARGS, sample, &wo_t, pdf, shader_data);
			break;
	}
	*wo = matrix_mul_vector3(dg->tangent_to_world, wo_t);	return res;
}


#endif // BXDF_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef LIGHT_CL
#define LIGHT_CL
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef SCENE_CL
#define SCENE_CL


typedef struct
{
    // Vertices
    GLOBAL float3 const* restrict vertices;
    // Normals
    GLOBAL float3 const* restrict normals;
    // UVs
    GLOBAL float2 const* restrict uvs;
    // Indices
    GLOBAL int const* restrict indices;
    // Shapes
    GLOBAL Shape const* restrict shapes;
    // Material attributes
    GLOBAL int const* restrict material_attributes;
    // Input map values
    GLOBAL InputMapData const* restrict input_map_values;
    // Emissive objects
    GLOBAL Light const* restrict lights;
    // Envmap idx
    int env_light_idx;
    // Number of emissive objects
    int num_lights;
    // Light distribution 
    GLOBAL int const* restrict light_distribution;
} Scene;

// Get triangle vertices given scene, shape index and prim index
INLINE void Scene_GetTriangleVertices(Scene const* scene, int shape_idx, int prim_idx, float3* v0, float3* v1, float3* v2)
{
    // Extract shape data
    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch positions and transform to world space
    *v0 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i0]);
    *v1 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i1]);
    *v2 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i2]);
}

// Get triangle uvs given scene, shape index and prim index
INLINE void Scene_GetTriangleUVs(Scene const* scene, int shape_idx, int prim_idx, float2* uv0, float2* uv1, float2* uv2)
{
    // Extract shape data
    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch positions and transform to world space
    *uv0 = scene->uvs[shape.startvtx + i0];
    *uv1 = scene->uvs[shape.startvtx + i1];
    *uv2 = scene->uvs[shape.startvtx + i2];
}


// Interpolate position, normal and uv
INLINE void Scene_InterpolateAttributes(Scene const* scene, int shape_idx, int prim_idx, float2 barycentrics, float3* p, float3* n, float2* uv, float* area)
{
    // Extract shape data
    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch normals
    float3 n0 = scene->normals[shape.startvtx + i0];
    float3 n1 = scene->normals[shape.startvtx + i1];
    float3 n2 = scene->normals[shape.startvtx + i2];

    // Fetch positions and transform to world space
    float3 v0 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i0]);
    float3 v1 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i1]);
    float3 v2 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i2]);

    // Fetch UVs
    float2 uv0 = scene->uvs[shape.startvtx + i0];
    float2 uv1 = scene->uvs[shape.startvtx + i1];
    float2 uv2 = scene->uvs[shape.startvtx + i2];

    // Calculate barycentric position and normal
    *p = (1.f - barycentrics.x - barycentrics.y) * v0 + barycentrics.x * v1 + barycentrics.y * v2;
    *n = normalize(matrix_mul_vector3(shape.transform, (1.f - barycentrics.x - barycentrics.y) * n0 + barycentrics.x * n1 + barycentrics.y * n2));
    *uv = (1.f - barycentrics.x - barycentrics.y) * uv0 + barycentrics.x * uv1 + barycentrics.y * uv2;
    *area = 0.5f * length(cross(v2 - v0, v1 - v0));
}

// Interpolate position, normal and uv
INLINE void Scene_InterpolateVertices(Scene const* scene, int shape_idx, int prim_idx, float2 barycentrics, float3* p)
{
    // Extract shape data
    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch positions and transform to world space
    float3 v0 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i0]);
    float3 v1 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i1]);
    float3 v2 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i2]);

    // Calculate barycentric position and normal
    *p = (1.f - barycentrics.x - barycentrics.y) * v0 + barycentrics.x * v1 + barycentrics.y * v2;
}

// Interpolate position, normal and uv
INLINE void Scene_InterpolateVerticesFromIntersection(Scene const* scene, Intersection const* isect, float3* p)
{
    // Extract shape data
    int shape_idx = isect->shapeid - 1;
    int prim_idx = isect->primid;
    float2 barycentrics = isect->uvwt.xy;

    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch positions and transform to world space
    float3 v0 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i0]);
    float3 v1 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i1]);
    float3 v2 = matrix_mul_point3(shape.transform, scene->vertices[shape.startvtx + i2]);

    // Calculate barycentric position and normal
    *p = (1.f - barycentrics.x - barycentrics.y) * v0 + barycentrics.x * v1 + barycentrics.y * v2;
}

// Interpolate position, normal and uv
INLINE void Scene_InterpolateNormalsFromIntersection(Scene const* scene, Intersection const* isect, float3* n)
{
    // Extract shape data
    int shape_idx = isect->shapeid - 1;
    int prim_idx = isect->primid;
    float2 barycentrics = isect->uvwt.xy;

    Shape shape = scene->shapes[shape_idx];

    // Fetch indices starting from startidx and offset by prim_idx
    int i0 = scene->indices[shape.startidx + 3 * prim_idx];
    int i1 = scene->indices[shape.startidx + 3 * prim_idx + 1];
    int i2 = scene->indices[shape.startidx + 3 * prim_idx + 2];

    // Fetch normals
    float3 n0 = scene->normals[shape.startvtx + i0];
    float3 n1 = scene->normals[shape.startvtx + i1];
    float3 n2 = scene->normals[shape.startvtx + i2];

    // Calculate barycentric position and normal
    *n = normalize(matrix_mul_vector3(shape.transform, (1.f - barycentrics.x - barycentrics.y) * n0 + barycentrics.x * n1 + barycentrics.y * n2));
}

INLINE int Scene_GetVolumeIndex(Scene const* scene, int shape_idx)
{
    Shape shape = scene->shapes[shape_idx];
    return shape.volume_idx;
}

/// Fill DifferentialGeometry structure based on intersection info from RadeonRays
void Scene_FillDifferentialGeometry(// Scene
                              Scene const* scene,
                              // RadeonRays intersection
                              Intersection const* isect,
                              // Differential geometry
                              DifferentialGeometry* diffgeo
                              )
{
    // Determine shape and polygon
    int shape_idx = isect->shapeid - 1;
    int prim_idx = isect->primid;

    // Get barycentrics
    float2 barycentrics = isect->uvwt.xy;

    // Extract shape data
    Shape shape = scene->shapes[shape_idx];

    // Interpolate attributes
    float3 p;
    float3 n;
    float2 uv;
    float area;
    Scene_InterpolateAttributes(scene, shape_idx, prim_idx, barycentrics, &p, &n, &uv, &area);
    // Triangle area (for area lighting)
    diffgeo->area = area;

    // Calculate barycentric position and normal
    diffgeo->n = n;
    diffgeo->p = p;
    diffgeo->uv = uv;

    // Get vertices
    float3 v0, v1, v2;
    Scene_GetTriangleVertices(scene, shape_idx, prim_idx, &v0, &v1, &v2);

    // Calculate true normal
    diffgeo->ng = normalize(cross(v1 - v0, v2 - v0));

    // Get material at shading point
    diffgeo->mat = shape.material;

    // Get UVs
    float2 uv0, uv1, uv2;
    Scene_GetTriangleUVs(scene, shape_idx, prim_idx, &uv0, &uv1, &uv2);

    // Reverse geometric normal if shading normal points to different side
    if (dot(diffgeo->ng, diffgeo->n) < 0.f)
    {
        diffgeo->ng = -diffgeo->ng;
    }

    /// Calculate tangent basis
    /// From PBRT book
    float du1 = uv0.x - uv2.x;
    float du2 = uv1.x - uv2.x;
    float dv1 = uv0.y - uv2.y;
    float dv2 = uv1.y - uv2.y;
    float3 dp1 = v0 - v2;
    float3 dp2 = v1 - v2;
    float det = du1 * dv2 - dv1 * du2;

    if (0 && det != 0.f)
    {
        float invdet = 1.f / det;
        diffgeo->dpdu = normalize( (dv2 * dp1 - dv1 * dp2) * invdet );
        diffgeo->dpdv = normalize( (-du2 * dp1 + du1 * dp2) * invdet );
        diffgeo->dpdu -= dot(diffgeo->n, diffgeo->dpdu) * diffgeo->n;
        diffgeo->dpdu = normalize(diffgeo->dpdu);
        
        diffgeo->dpdv -= dot(diffgeo->n, diffgeo->dpdv) * diffgeo->n;
        diffgeo->dpdv -= dot(diffgeo->dpdu, diffgeo->dpdv) * diffgeo->dpdu;
        diffgeo->dpdv = normalize(diffgeo->dpdv);
    }
    else
    {
        diffgeo->dpdu = normalize(GetOrthoVector(diffgeo->n));
        diffgeo->dpdv = normalize(cross(diffgeo->n, diffgeo->dpdu));
    }
}


// Calculate tangent transform matrices inside differential geometry
INLINE void DifferentialGeometry_CalculateTangentTransforms(DifferentialGeometry* diffgeo)
{
    diffgeo->world_to_tangent = matrix_from_rows3(diffgeo->dpdu, diffgeo->n, diffgeo->dpdv);

    diffgeo->world_to_tangent.m0.w = -dot(diffgeo->dpdu, diffgeo->p);
    diffgeo->world_to_tangent.m1.w = -dot(diffgeo->n, diffgeo->p);
    diffgeo->world_to_tangent.m2.w = -dot(diffgeo->dpdv, diffgeo->p);

    diffgeo->tangent_to_world = matrix_from_cols3(diffgeo->world_to_tangent.m0.xyz, 
        diffgeo->world_to_tangent.m1.xyz, diffgeo->world_to_tangent.m2.xyz);

    diffgeo->tangent_to_world.m0.w = diffgeo->p.x;
    diffgeo->tangent_to_world.m1.w = diffgeo->p.y;
    diffgeo->tangent_to_world.m2.w = diffgeo->p.z;
}

#define POWER_SAMPLING

// Sample light index
INLINE int Scene_SampleLight(Scene const* scene, float sample, float* pdf)
{
#ifndef POWER_SAMPLING
    int num_lights = scene->num_lights;
    int light_idx = clamp((int)(sample * num_lights), 0, num_lights - 1);
    *pdf = 1.f / num_lights;
    return light_idx;
#else
    int num_lights = scene->num_lights;
    int light_idx = Distribution1D_SampleDiscrete(sample, scene->light_distribution, pdf);
    return light_idx;
#endif
}

#endif
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef PATH_CL
#define PATH_CL


typedef struct _Path
{
    float3 throughput;
    int volume;
    int flags;
    int active;
    int extra1;
} Path;

typedef enum _PathFlags
{
    kNone = 0x0,
    kKilled = 0x1,
    kScattered = 0x2,
    kOpaque = 0x4
} PathFlags;

INLINE bool Path_IsScattered(__global Path const* path)
{
    return path->flags & kScattered;
}

INLINE bool Path_IsAlive(__global Path const* path)
{
    return ((path->flags & kKilled) == 0);
}

INLINE bool Path_ContainsOpacity(__global Path const* path)
{
    return path->flags & kOpaque;
}

INLINE void Path_ClearScatterFlag(__global Path* path)
{
    path->flags &= ~kScattered;
}

INLINE void Path_SetScatterFlag(__global Path* path)
{
    path->flags |= kScattered;
}

INLINE void Path_SetOpacityFlag(__global Path* path)
{
    path->flags |= kOpaque;
}

INLINE void Path_ClearBxdfFlags(__global Path* path)
{
    path->flags &= (kKilled | kScattered | kOpaque);
}

INLINE int Path_GetBxdfFlags(__global Path const* path)
{
    return path->flags >> 8;
}

INLINE int Path_SetBxdfFlags(__global Path* path, int flags)
{
    return path->flags |= (flags << 8);
}

INLINE void Path_Restart(__global Path* path)
{
    path->flags = 0;
}

INLINE int Path_GetVolumeIdx(__global Path const* path)
{
    return path->volume;
}

INLINE void Path_SetVolumeIdx(__global Path* path, int volume_idx)
{
    path->volume = volume_idx;
}

INLINE float3 Path_GetThroughput(__global Path const* path)
{
    float3 t = path->throughput;
    return t;
}

INLINE void Path_MulThroughput(__global Path* path, float3 mul)
{
    path->throughput *= mul;
}

INLINE void Path_Kill(__global Path* path)
{
    path->flags |= kKilled;
}

INLINE void Path_AddContribution(__global Path* path, __global float3* output, int idx, float3 val)
{
    output[idx] += Path_GetThroughput(path) * val;
}

INLINE bool Path_IsSpecular(__global Path const* path)
{
    int flags = Path_GetBxdfFlags(path);
    return (flags & kBxdfFlagsSingular) == kBxdfFlagsSingular;
}

INLINE void Path_SetFlags(DifferentialGeometry* diffgeo, GLOBAL Path* restrict path)
{
    Path_ClearBxdfFlags(path);
    Path_SetBxdfFlags(path, Bxdf_GetFlags(diffgeo));
}

#endif


enum LightInteractionType
{
    kLightInteractionUnknown,
    kLightInteractionSurface,
    kLightInteractionVolume
};

INLINE
bool IntersectTriangle(ray const* r, float3 v1, float3 v2, float3 v3, float* a, float* b)
{
    const float3 e1 = v2 - v1;
    const float3 e2 = v3 - v1;
    const float3 s1 = cross(r->d.xyz, e2);
    const float  invd = native_recip(dot(s1, e1));
    const float3 d = r->o.xyz - v1;
    const float  b1 = dot(d, s1) * invd;
    const float3 s2 = cross(d, e1);
    const float  b2 = dot(r->d.xyz, s2) * invd;
    const float temp = dot(e2, s2) * invd;

    if (b1 < 0.f || b1 > 1.f || b2 < 0.f || b1 + b2 > 1.f)
    {
        return false;
    }
    else
    {
        *a = b1;
        *b = b2;
        return true;
    }
}

INLINE int EnvironmentLight_GetTexture(Light const* light, int bxdf_flags)
{
    int tex = light->tex;

    if ((bxdf_flags & kBxdfFlagsBrdf) && (light->tex_reflection != -1) && ((bxdf_flags & kBxdfFlagsDiffuse) != kBxdfFlagsDiffuse))
        tex = light->tex_reflection;

    if (((bxdf_flags & kBxdfFlagsBrdf) == 0) && light->tex_refraction != -1)
        tex = light->tex_refraction;

    if ((bxdf_flags & kBxdfFlagsTransparency) && light->tex_transparency != -1)
        tex = light->tex_transparency;

    return tex;
}

INLINE int EnvironmentLight_GetBackgroundTexture(Light const* light)
{
    return light->tex_background == -1 ? light->tex : light->tex_background;
}

/*
 Environment light
 */
/// Get intensity for a given direction
float3 EnvironmentLight_GetLe(// Light
                              Light const* light,
                              // Scene
                              Scene const* scene,
                              // Geometry
                              DifferentialGeometry const* dg,
                              // Path flags
                              int bxdf_flags,
                              // Light inteaction type
                              int interaction_type,
                              // Direction to light source
                              float3* wo,
                              // Textures
                              TEXTURE_ARG_LIST
                              )
{
    // Sample envmap
    *wo *= CRAZY_HIGH_DISTANCE;

    int tex = EnvironmentLight_GetTexture(light, bxdf_flags);

    if (tex == -1)
    {
        return 0.f;
    }

    return light->multiplier * Texture_SampleEnvMap(normalize(*wo), TEXTURE_ARGS_IDX(tex), light->ibl_mirror_x);
}

/// Sample direction to the light
float3 EnvironmentLight_Sample(// Light
                               Light const* light,
                               // Scene
                               Scene const* scene,
                               // Geometry
                               DifferentialGeometry const* dg,
                               // Textures
                               TEXTURE_ARG_LIST,
                               // Sample
                               float2 sample,
                               // Path flags
                               int bxdf_flags,
                               // Light inteaction type
                               int interaction_type,
                               // Direction to light source
                               float3* wo,
                               // PDF
                               float* pdf
                              )
{
    float3 d;

    if (interaction_type != kLightInteractionVolume)
    {
        d = Sample_MapToHemisphere(sample, dg->n, 0.f);
        *pdf = 1.f / (2.f * PI);
    }
    else
    {
        d = Sample_MapToSphere(sample);
        *pdf = 1.f / (4.f * PI);
    }

    // Generate direction
    *wo = CRAZY_HIGH_DISTANCE * d;

    int tex = EnvironmentLight_GetTexture(light, bxdf_flags);

    if (tex == -1)
    {
        *pdf = 0.f;
        return 0.f;
    }

    // Sample envmap
    return light->multiplier * Texture_SampleEnvMap(d, TEXTURE_ARGS_IDX(tex), light->ibl_mirror_x);
}

/// Get PDF for a given direction
float EnvironmentLight_GetPdf(
                              // Light
                              Light const* light,
                              // Scene
                              Scene const* scene,
                              // Geometry
                              DifferentialGeometry const* dg,
                              // Path flags
                              int bxdf_flags,
                              // Light inteaction type
                              int interaction_type,
                              // Direction to light source
                              float3 wo,
                              // Textures
                              TEXTURE_ARG_LIST
                              )
{
    if (interaction_type != kLightInteractionVolume)
    {
        return 1.f / (2.f * PI);
    }
    else
    {
        return 1.f / (4.f * PI);
    }
}


/*
 Area light
 */
// Get intensity for a given direction
float3 AreaLight_GetLe(// Emissive object
                       Light const* light,
                       // Scene
                       Scene const* scene,
                       // Geometry
                       DifferentialGeometry const* dg,
                       // Direction to light source
                       float3* wo,
                       // Textures
                       TEXTURE_ARG_LIST
                       )
{
    ray r;
    r.o.xyz = dg->p;
    r.d.xyz = *wo;

    int shapeidx = light->shapeidx;
    int primidx = light->primidx;

    float3 v0, v1, v2;
    Scene_GetTriangleVertices(scene, shapeidx, primidx, &v0, &v1, &v2);

    float a, b;
    if (IntersectTriangle(&r, v0, v1, v2, &a, &b))
    {
        float3 n;
        float3 p;
        float2 tx;
        float area;
        Scene_InterpolateAttributes(scene, shapeidx, primidx, make_float2(a, b), &p, &n, &tx, &area);

        float3 d = p - dg->p;
        *wo = d;

        int material_offset = scene->shapes[shapeidx].material.offset;

        const float3 ke = GetUberV2EmissionColor(material_offset, dg, scene->input_map_values, scene->material_attributes, TEXTURE_ARGS).xyz;
        
        return ke;
    }
    else
    {
        return make_float3(0.f, 0.f, 0.f);
    }
}

/// Sample direction to the light
float3 AreaLight_Sample(// Emissive object
                        Light const* light,
                        // Scene
                        Scene const* scene,
                        // Geometry
                        DifferentialGeometry const* dg,
                        // Textures
                        TEXTURE_ARG_LIST,
                        // Sample
                        float2 sample,
                        // Direction to light source
                        float3* wo,
                        // PDF
                        float* pdf)
{
    int shapeidx = light->shapeidx;
    int primidx = light->primidx;

    // Generate sample on triangle
    float r0 = sample.x;
    float r1 = sample.y;

    // Convert random to barycentric coords
    float2 uv;

    uv.x = 1.f - native_sqrt(r0);
    uv.y = native_sqrt(r0) * r1;

    float3 n;
    float3 p;
    float2 tx;
    float area;
    Scene_InterpolateAttributes(scene, shapeidx, primidx, uv, &p, &n, &tx, &area);

    *wo = p - dg->p;

    int material_offset = scene->shapes[shapeidx].material.offset;

    const float3 ke = GetUberV2EmissionColor(material_offset, dg, scene->input_map_values, scene->material_attributes, TEXTURE_ARGS).xyz;
    float3 v = -normalize(*wo);

    float ndotv = dot(n, v);

    if (ndotv > 0.f)
    {
        float dist2 = dot(*wo, *wo);
        float denom = fabs(ndotv) * area;
        *pdf = denom > 0.f ? dist2 / denom : 0.f;
        return ke;
    }
    else
    {
        *pdf = 0.f;
        return 0.f;
    }
}

/// Get PDF for a given direction
float AreaLight_GetPdf(// Emissive object
                       Light const* light,
                       // Scene
                       Scene const* scene,
                       // Geometry
                       DifferentialGeometry const* dg,
                       // Direction to light source
                       float3 wo,
                       // Textures
                       TEXTURE_ARG_LIST
                       )
{
    ray r;
    r.o.xyz = dg->p;
    r.d.xyz = wo;

    int shapeidx = light->shapeidx;
    int primidx = light->primidx;

    float3 v0, v1, v2;
    Scene_GetTriangleVertices(scene, shapeidx, primidx, &v0, &v1, &v2);

    // Intersect ray against this area light
    float a, b;
    if (IntersectTriangle(&r, v0, v1, v2, &a, &b))
    {
        float3 n;
        float3 p;
        float2 tx;
        float area;
        Scene_InterpolateAttributes(scene, shapeidx, primidx, make_float2(a, b), &p, &n, &tx, &area);

        float3 d = p - dg->p;
        float dist2 = dot(d, d) ;
        float denom = (fabs(dot(-normalize(d), n)) * area);

        return denom > 0.f ? dist2 / denom : 0.f;
    }
    else
    {
        return 0.f;
    }
}

float3 AreaLight_SampleVertex(
    // Emissive object
    Light const* light,
    // Scene
    Scene const* scene,
    // Textures
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample0,
    float2 sample1,
    // Direction to light source
    float3* p,
    float3* n,
    float3* wo,
    // PDF
    float* pdf)
{
    int shapeidx = light->shapeidx;
    int primidx = light->primidx;

    // Generate sample on triangle
    float r0 = sample0.x;
    float r1 = sample0.y;

    // Convert random to barycentric coords
    float2 uv;
    uv.x = native_sqrt(r0) * (1.f - r1);
    uv.y = native_sqrt(r0) * r1;

    float2 tx;
    float area;
    Scene_InterpolateAttributes(scene, shapeidx, primidx, uv, p, n, &tx, &area);

    int material_offset = scene->shapes[shapeidx].material.offset;

    /*const float3 ke = GetUberV2EmissionColor(material_offset, dg, scene->input_map_values, scene->material_attributes, TEXTURE_ARGS).xyz;*/
    const float3 ke = make_float3(0.f, 0.f, 0.f);
    *wo = Sample_MapToHemisphere(sample1, *n, 1.f);
    *pdf = (1.f / area) * fabs(dot(*n, *wo)) / PI;

    return ke;
}

/*
Directional light
*/
// Get intensity for a given direction
float3 DirectionalLight_GetLe(// Emissive object
    Light const* light,
    // Scene
    Scene const* scene,
    // Geometry
    DifferentialGeometry const* dg,
    // Direction to light source
    float3* wo,
    // Textures
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

/// Sample direction to the light
float3 DirectionalLight_Sample(// Emissive object
    Light const* light,
    // Scene
    Scene const* scene,
    // Geometry
    DifferentialGeometry const* dg,
    // Textures
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample,
    // Direction to light source
    float3* wo,
    // PDF
    float* pdf)
{
    *wo = CRAZY_HIGH_DISTANCE * -light->d;
    *pdf = 1.f;
    return light->intensity;
}

/// Get PDF for a given direction
float DirectionalLight_GetPdf(// Emissive object
    Light const* light,
    // Scene
    Scene const* scene,
    // Geometry
    DifferentialGeometry const* dg,
    // Direction to light source
    float3 wo,
    // Textures
    TEXTURE_ARG_LIST
)
{
    return 0.f;
}

/*
 Point light
 */
// Get intensity for a given direction
float3 PointLight_GetLe(// Emissive object
                              Light const* light,
                              // Scene
                              Scene const* scene,
                              // Geometry
                              DifferentialGeometry const* dg,
                              // Direction to light source
                              float3* wo,
                              // Textures
                              TEXTURE_ARG_LIST
                              )
{
    return 0.f;
}

/// Sample direction to the light
float3 PointLight_Sample(// Emissive object
                               Light const* light,
                               // Scene
                               Scene const* scene,
                               // Geometry
                               DifferentialGeometry const* dg,
                               // Textures
                               TEXTURE_ARG_LIST,
                               // Sample
                               float2 sample,
                               // Direction to light source
                               float3* wo,
                               // PDF
                               float* pdf)
{
    *wo = light->p - dg->p;
    *pdf = 1.f;
    return light->intensity / dot(*wo, *wo);
}

/// Get PDF for a given direction
float PointLight_GetPdf(// Emissive object
                              Light const* light,
                              // Scene
                              Scene const* scene,
                              // Geometry
                              DifferentialGeometry const* dg,
                              // Direction to light source
                              float3 wo,
                              // Textures
                              TEXTURE_ARG_LIST
                              )
{
    return 0.f;
}

/// Sample vertex on the light
float3 PointLight_SampleVertex(
    // Light object
    Light const* light,
    // Scene
    Scene const* scene,
    // Textures
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample0,
    float2 sample1,
    // Direction to light source
    float3* p,
    float3* n,
    float3* wo,
    // PDF
    float* pdf)
{
    *p = light->p;
    *n = make_float3(0.f, 1.f, 0.f);
    *wo = Sample_MapToSphere(sample0);
    *pdf = 1.f / (4.f * PI);
    return light->intensity;
}

/*
 Spot light
 */
// Get intensity for a given direction
float3 SpotLight_GetLe(// Emissive object
                        Light const* light,
                        // Scene
                        Scene const* scene,
                        // Geometry
                        DifferentialGeometry const* dg,
                        // Direction to light source
                        float3* wo,
                        // Textures
                        TEXTURE_ARG_LIST
                        )
{
    return 0.f;
}

/// Sample direction to the light
float3 SpotLight_Sample(// Emissive object
                         Light const* light,
                         // Scene
                         Scene const* scene,
                         // Geometry
                         DifferentialGeometry const* dg,
                         // Textures
                         TEXTURE_ARG_LIST,
                         // Sample
                         float2 sample,
                         // Direction to light source
                         float3* wo,
                         // PDF
                         float* pdf)
{
    *wo = light->p - dg->p;
    float ddotwo = dot(-normalize(*wo), light->d);
    
    if (ddotwo > light->oa)
    {
        float3 intensity = light->intensity / dot(*wo, *wo);
        *pdf = 1.f;
        return ddotwo > light->ia ? intensity : intensity * (1.f - (light->ia - ddotwo) / (light->ia - light->oa));
    }
    else
    {
        *pdf = 0.f;
        return 0.f;
    }
}

/// Get PDF for a given direction
float SpotLight_GetPdf(// Emissive object
                        Light const* light,
                        // Scene
                        Scene const* scene,
                        // Geometry
                        DifferentialGeometry const* dg,
                        // Direction to light source
                        float3 wo,
                        // Textures
                        TEXTURE_ARG_LIST
                        )
{
    return 0.f;
}


/*
 Dispatch calls
 */

/// Get intensity for a given direction
float3 Light_GetLe(// Light index
                   int idx,
                   // Scene
                   Scene const* scene,
                   // Geometry
                   DifferentialGeometry const* dg,
                   // Path flags
                    int bxdf_flags,
                   // Light inteaction type
                   int interaction_type,
                   // Direction to light source
                   float3* wo,
                   // Textures
                   TEXTURE_ARG_LIST
                   )
{
    Light light = scene->lights[idx];

    switch(light.type)
    {
        case kIbl:
            return EnvironmentLight_GetLe(&light, scene, dg, bxdf_flags, interaction_type, wo, TEXTURE_ARGS);
        case kArea:
            return AreaLight_GetLe(&light, scene, dg, wo, TEXTURE_ARGS);
        case kDirectional:
            return DirectionalLight_GetLe(&light, scene, dg, wo, TEXTURE_ARGS);
        case kPoint:
            return PointLight_GetLe(&light, scene, dg, wo, TEXTURE_ARGS);
        case kSpot:
            return SpotLight_GetLe(&light, scene, dg, wo, TEXTURE_ARGS);
    }

    return make_float3(0.f, 0.f, 0.f);
}

/// Sample direction to the light
float3 Light_Sample(// Light index
                    int idx,
                    // Scene
                    Scene const* scene,
                    // Geometry
                    DifferentialGeometry const* dg,
                    // Textures
                    TEXTURE_ARG_LIST,
                    // Sample
                    float2 sample,
                    // Path flags
                    int bxdf_flags,
                    // Light inteaction type
                    int interaction_type,
                    // Direction to light source
                    float3* wo,
                    // PDF
                    float* pdf)
{
    Light light = scene->lights[idx];

    switch(light.type)
    {
        case kIbl:
            return EnvironmentLight_Sample(&light, scene, dg, TEXTURE_ARGS, sample, bxdf_flags, interaction_type, wo, pdf);
        case kArea:
            return AreaLight_Sample(&light, scene, dg, TEXTURE_ARGS, sample, wo, pdf);
        case kDirectional:
            return DirectionalLight_Sample(&light, scene, dg, TEXTURE_ARGS, sample, wo, pdf);
        case kPoint:
            return PointLight_Sample(&light, scene, dg, TEXTURE_ARGS, sample, wo, pdf);
        case kSpot:
            return SpotLight_Sample(&light, scene, dg, TEXTURE_ARGS, sample, wo, pdf);
    }

    *pdf = 0.f;
    return make_float3(0.f, 0.f, 0.f);
}

/// Get PDF for a given direction
float Light_GetPdf(// Light index
                   int idx,
                   // Scene
                   Scene const* scene,
                   // Geometry
                   DifferentialGeometry const* dg,
                    // Path flags
                    int bxdf_flags,
                    // Light inteaction type
                    int interaction_type,
                   // Direction to light source
                   float3 wo,
                   // Textures
                   TEXTURE_ARG_LIST
                   )
{
    Light light = scene->lights[idx];

    switch(light.type)
    {
        case kIbl:
            return EnvironmentLight_GetPdf(&light, scene, dg, bxdf_flags, interaction_type, wo, TEXTURE_ARGS);
        case kArea:
            return AreaLight_GetPdf(&light, scene, dg, wo, TEXTURE_ARGS);
        case kDirectional:
            return DirectionalLight_GetPdf(&light, scene, dg, wo, TEXTURE_ARGS);
        case kPoint:
            return PointLight_GetPdf(&light, scene, dg, wo, TEXTURE_ARGS);
        case kSpot:
            return SpotLight_GetPdf(&light, scene, dg, wo, TEXTURE_ARGS);
    }

    return 0.f;
}

/// Sample vertex on the light
float3 Light_SampleVertex(
    // Light index
    int idx,
    // Scene
    Scene const* scene,
    // Textures
    TEXTURE_ARG_LIST,
    // Sample
    float2 sample0,
    float2 sample1,
    // Point on the light
    float3* p,
    // Normal at light vertex
    float3* n,
    // Direction
    float3* wo,
    // PDF
    float* pdf)
{
    Light light = scene->lights[idx];

    switch (light.type)
    {
        case kArea:
            return AreaLight_SampleVertex(&light, scene, TEXTURE_ARGS, sample0, sample1, p, n, wo, pdf);
        case kPoint:
            return PointLight_SampleVertex(&light, scene, TEXTURE_ARGS, sample0, sample1, p, n, wo, pdf);
    }

    *pdf = 0.f;
    return make_float3(0.f, 0.f, 0.f);
}

/// Check if the light is singular
bool Light_IsSingular(__global Light const* light)
{
    return light->type == kPoint ||
        light->type == kSpot ||
        light->type == kDirectional;
}

#endif // LIGHT_CLnv
/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef VOLUMETRICS_CL
#define VOLUMETRICS_CL


#define FAKE_SHAPE_SENTINEL 0xFFFFFF

float PhaseFunctionHG(float3 wi, float3 wo, float g)
{
    float costheta = dot(wi, wo);
    return 1.f / (4.f * PI) *
        (1.f - g*g) / native_powr(1.f + g*g - 2.f * g * costheta, 1.5f);
}

// See PBRT for derivation
float PhaseFunctionHG_Sample(float3 wi, float g, float2 sample, float3* wo)
{
    float costheta = 0.f;
    if (fabs(g) < 1e-5)
    {
        costheta = 1.f - 2.f * sample.x;
    }
    else
    {
        float temp = (1.f - g * g) / (1.f - g + 2.f * g * sample.x);
        costheta = (1 + g * g - temp * temp) / (2.f * g);
    }

    float phi = 2.f * PI * sample.y;

    float3 u = GetOrthoVector(-wi);
    float3 v = normalize(cross(-wi, u));
    *wo = u * native_cos(phi) + v * native_sin(phi) - wi * costheta;

    return PhaseFunctionHG(wi, *wo, g);
}

// Evaluate volume transmittance along the ray [0, dist] segment
float3 Volume_Transmittance(GLOBAL Volume const* volume, GLOBAL ray const* ray, float dist)
{
    switch (volume->type)
    {
        case kHomogeneous:
        {
            // For homogeneous it is e(-sigma * dist)
            float3 sigma_t = TEXTURED_INPUT_GET_COLOR(volume->sigma_a) +
                             TEXTURED_INPUT_GET_COLOR(volume->sigma_s);
            return native_exp(-sigma_t * dist);
        }
    }
    
    return 1.f;
}

// Evaluate volume selfemission along the ray [0, dist] segment
float3 Volume_Emission(GLOBAL Volume const* volume, GLOBAL ray const* ray, float dist)
{
    switch (volume->type)
    {
        case kHomogeneous:
        {
            return TEXTURED_INPUT_GET_COLOR(volume->sigma_e) * dist;
        }
    }
    
    return 0.f;
}

// Sample volume in order to find next scattering event
float Volume_SampleDistance(GLOBAL Volume const* volume, GLOBAL ray const* ray, float maxdist, float2 sample, float* pdf)
{
    // Sample component
    float3 sigma_s = TEXTURED_INPUT_GET_COLOR(volume->sigma_s);
    float sigma = sample.x < 0.33f ? sigma_s.x :
                  sample.x < 0.66f ? sigma_s.y : sigma_s.z;

    switch (volume->type)
    {
        case kHomogeneous:
        {
            
            float d = sigma > 0.f ? (-native_log(sample.y) / sigma) : -1.f;
            float temp = (1.f / 3.f) * (sigma_s.x * native_exp(-sigma_s.x * d)
                + sigma_s.y * native_exp(-sigma_s.y * d)
                + sigma_s.z * native_exp(-sigma_s.z * d));
            *pdf = sigma > 0.f ? temp : 0.f;
            return d;
        }
    }
    
    return -1.f;
}

// Sample volume in order to find next scattering event
float Volume_GetDistancePdf(GLOBAL Volume const* volume, float dist)
{
    switch (volume->type)
    {
    case kHomogeneous:
    {
        float3 sigma_s = TEXTURED_INPUT_GET_COLOR(volume->sigma_s);
        return (1.f / 3.f) * (native_exp(-sigma_s.x * dist)
                            + native_exp(-sigma_s.y * dist)
                            + native_exp(-sigma_s.z * dist));
    }
    }

    return 0.f;
}

// Apply volume effects (absorbtion and emission) and scatter if needed.
// The rays we handling here might intersect something or miss, 
// since scattering can happen even for missed rays.
// That's why this function is called prior to ray compaction.
// In case ray has missed geometry (has shapeid < 0) and has been scattered,
// we put FAKE_SHAPE_SENTINEL into shapeid to prevent ray from being compacted away.
//
KERNEL void SampleVolume(
    // Ray batch
    GLOBAL ray const* rays,
    // Pixel indices
    GLOBAL int const* pixelindices,
    // Output indices
    GLOBAL int const* output_indices,
    // Number of rays
    GLOBAL int const* numrays,
    // Volumes
    GLOBAL Volume const* volumes,
    // Textures
    TEXTURE_ARG_LIST,
    // RNG seed
    uint rngseed,
    // Sampler state
    GLOBAL uint* random,
    // Sobol matrices
    GLOBAL uint const* sobol_mat,
    // Current bounce 
    int bounce,
    // Current frame
    int frame,
    // Intersection data
    GLOBAL Intersection* isects,
    // Current paths
    GLOBAL Path* paths,
    // Output
    GLOBAL float3* output
    )
{
    int globalid = get_global_id(0);

    // Only handle active rays
    if (globalid < *numrays)
    {
        int pixelidx = pixelindices[globalid];
        
        GLOBAL Path* path = paths + pixelidx;

        // Path can be dead here since compaction step has not 
        // yet been applied
        if (!Path_IsAlive(path))
            return;

        int volidx = Path_GetVolumeIdx(path);

        // Check if we are inside some volume
        if (volidx != -1)
        {
            Sampler sampler;
#if SAMPLER == SOBOL
            uint scramble = random[pixelidx] * 0x1fe3434f;
            Sampler_Init(&sampler, frame, SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE + SAMPLE_DIM_VOLUME_APPLY_OFFSET, scramble);
#elif SAMPLER == RANDOM
            uint scramble = pixelidx * rngseed;
            Sampler_Init(&sampler, scramble);
#elif SAMPLER == CMJ
            uint rnd = random[pixelidx];
            uint scramble = rnd * 0x1fe3434f * ((frame + 71 * rnd) / (CMJ_DIM * CMJ_DIM));
            Sampler_Init(&sampler, frame % (CMJ_DIM * CMJ_DIM), SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE + SAMPLE_DIM_VOLUME_APPLY_OFFSET, scramble);
#endif

            // Try sampling volume for a next scattering event
            float pdf = 0.f;
            float maxdist = Intersection_GetDistance(isects + globalid);
            float2 sample = Sampler_Sample2D(&sampler, SAMPLER_ARGS);
            float2 sample1 = Sampler_Sample2D(&sampler, SAMPLER_ARGS);
            float d = Volume_SampleDistance(&volumes[volidx], &rays[globalid], maxdist, make_float2(sample.x, sample1.y), &pdf);
            
            // Check if we shall skip the event (it is either outside of a volume or not happened at all)
            bool skip = d < 0 || d > maxdist || pdf <= 0.f;

            if (skip)
            {
                // In case we skip we just need to apply volume absorbtion and emission for the segment we went through
                // and clear scatter flag
                Path_ClearScatterFlag(path);
                // And finally update the throughput
                Path_MulThroughput(path, Volume_Transmittance(&volumes[volidx], &rays[globalid], maxdist) * Volume_GetDistancePdf(&volumes[volidx], maxdist));
                // Emission contribution accounting for a throughput we have so far
                Path_AddContribution(path, output, output_indices[pixelidx], Volume_Emission(&volumes[volidx], &rays[globalid], maxdist));
            }
            else
            {
                // Set scattering flag to notify ShadeVolume kernel to handle this path
                Path_SetScatterFlag(path);
                // Update the throughput
                float3 sigma_s = TEXTURED_INPUT_GET_COLOR(volumes[volidx].sigma_s);
                Path_MulThroughput(path, sigma_s * (Volume_Transmittance(&volumes[volidx], &rays[globalid], d) / pdf));
                // Emission contribution accounting for a throughput we have so far
                Path_AddContribution(path, output, output_indices[pixelidx], Volume_Emission(&volumes[volidx], &rays[globalid], d) / pdf);
                // Put fake shape to prevent from being compacted away
                isects[globalid].shapeid = FAKE_SHAPE_SENTINEL;
                // And keep scattering distance around as well
                isects[globalid].uvwt.w = d;
            }
        }
    }
}

#endif // VOLUMETRICS_CL


// This kernel only handles scattered paths.
// It applies direct illumination and generates
// path continuation if multiscattering is enabled.
KERNEL void ShadeVolumeUberV2(
    // Ray batch
    GLOBAL ray const* restrict rays,
    // Intersection data
    GLOBAL Intersection const* restrict isects,
    // Hit indices
    GLOBAL int const* restrict hit_indices,
    // Pixel indices
    GLOBAL int const*  restrict pixel_indices,
    // Output indices
    GLOBAL int const*  restrict output_indices,
    // Number of rays
    GLOBAL int const*  restrict num_hits,
    // Vertices
    GLOBAL float3 const* restrict vertices,
    // Normals
    GLOBAL float3 const* restrict normals,
    // UVs
    GLOBAL float2 const* restrict uvs,
    // Indices
    GLOBAL int const* restrict indices,
    // Shapes
    GLOBAL Shape const* restrict shapes,
    // Material parameters
    GLOBAL int const* restrict material_attributes,
    // Textures
    TEXTURE_ARG_LIST,
    // Environment texture index
    int env_light_idx,
    // Emissives
    GLOBAL Light const* restrict lights,
    // Light distribution
    GLOBAL int const* restrict light_distribution,
    // Number of emissive objects
    int num_lights,
    // RNG seed
    uint rng_seed,
    // Sampler state
    GLOBAL uint* restrict random,
    // Sobol matrices
    GLOBAL uint const* restrict sobol_mat,
    // Current bounce
    int bounce,
    // Current frame
    int frame,
    // Volume data
    GLOBAL Volume const* restrict volumes,
    // Shadow rays
    GLOBAL ray* restrict shadow_rays,
    // Light samples
    GLOBAL float3* restrict light_samples,
    // Path throughput
    GLOBAL Path* restrict paths,
    // Indirect rays (next path segment)
    GLOBAL ray* restrict indirect_rays,
    // Radiance
    GLOBAL float3* restrict output,
    GLOBAL InputMapData const* restrict input_map_values
)
{
    int global_id = get_global_id(0);

    Scene scene =
    {
        vertices,
        normals,
        uvs,
        indices,
        shapes,
        material_attributes,
        input_map_values,
        lights,
        env_light_idx,
        num_lights,
        light_distribution
    };

    if (global_id < *num_hits)
    {
        // Fetch index
        int hit_idx = hit_indices[global_id];
        int pixel_idx = pixel_indices[global_id];
        Intersection isect = isects[hit_idx];

        GLOBAL Path* path = paths + pixel_idx;

        // Only apply to scattered paths
        if (!Path_IsScattered(path))
        {
            return;
        }

        // Fetch incoming ray
        float3 o = rays[hit_idx].o.xyz;
        float3 wi = -rays[hit_idx].d.xyz;

        Sampler sampler;
#if SAMPLER == SOBOL
        uint scramble = random[pixel_idx] * 0x1fe3434f;
        Sampler_Init(&sampler, frame, SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE + SAMPLE_DIM_VOLUME_EVALUATE_OFFSET, scramble);
#elif SAMPLER == RANDOM
        uint scramble = pixel_idx * rng_seed;
        Sampler_Init(&sampler, scramble);
#elif SAMPLER == CMJ
        uint rnd = random[pixel_idx];
        uint scramble = rnd * 0x1fe3434f * ((frame + 13 * rnd) / (CMJ_DIM * CMJ_DIM));
        Sampler_Init(&sampler, frame % (CMJ_DIM * CMJ_DIM), SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE + SAMPLE_DIM_VOLUME_EVALUATE_OFFSET, scramble);
#endif


        // Here we know that volume_idx != -1 since this is a precondition
        // for scattering event
        int volume_idx = Path_GetVolumeIdx(path);

        // Sample light source
        float pdf = 0.f;
        float selection_pdf = 0.f;
        float3 wo;

        int light_idx = Scene_SampleLight(&scene, Sampler_Sample1D(&sampler, SAMPLER_ARGS), &selection_pdf);

        // Here we need fake differential geometry for light sampling procedure
        DifferentialGeometry dg;
        // put scattering position in there (it is along the current ray at isect.distance
        // since EvaluateVolume has put it there
        dg.p = o - wi * Intersection_GetDistance(isects + hit_idx);
        // Get light sample intencity
        int bxdf_flags = Path_GetBxdfFlags(path); 
        float3 le = Light_Sample(light_idx, &scene, &dg, TEXTURE_ARGS, Sampler_Sample2D(&sampler, SAMPLER_ARGS), bxdf_flags, kLightInteractionVolume, &wo, &pdf);

        // Generate shadow ray
        float shadow_ray_length = length(wo); 
        Ray_Init(shadow_rays + global_id, dg.p, normalize(wo), shadow_ray_length, 0.f, 0xFFFFFFFF);
        Ray_SetExtra(shadow_rays + global_id, make_float2(1.f, 0.f));

        // Evaluate volume transmittion along the shadow ray (it is incorrect if the light source is outside of the
        // current volume, but in this case it will be discarded anyway since the intersection at the outer bound
        // of a current volume), so the result is fully correct.
        float3 tr = 1.f;// Volume_Transmittance(&volumes[volume_idx], &shadow_rays[global_id], shadow_ray_length);
        float3 emission = 0.f;// Volume_Emission(&volumes[volume_idx], &shadow_rays[global_id], shadow_ray_length);

        // Volume emission is applied only if the light source is in the current volume(this is incorrect since the light source might be
        // outside of a volume and we have to compute fraction of ray in this case, but need to figure out how)
        // float3 r = Volume_Emission(&volumes[volume_idx], &shadow_rays[global_id], shadow_ray_length);
        float3 r = 0.f;
        float g = volumes[volume_idx].g;
        // This is the estimate coming from a light source
        // TODO: remove hardcoded phase func and sigma 
        r += tr * le  * PhaseFunctionHG(wi, normalize(wo), g) / pdf / selection_pdf; 
        r += tr * emission;

        // Only if we have some radiance compute the visibility ray  
        if (NON_BLACK(tr) && NON_BLACK(r) && pdf > 0.f) 
        {
            // Put lightsample result
            light_samples[global_id] = REASONABLE_RADIANCE(r * Path_GetThroughput(path));
        }
        else
        { 
            // Nothing to compute
            light_samples[global_id] = 0.f;
            // Otherwise make it incative to save intersector cycles (hopefully) 
            Ray_SetInactive(shadow_rays + global_id);
        }

#ifdef MULTISCATTER
        // This is highly brute-force
        float phase = PhaseFunctionHG_Sample(wi, g, Sampler_Sample2D(&sampler, SAMPLER_ARGS), &wo);

        // Generate new path segment
        Ray_Init(indirect_rays + global_id, dg.p, normalize(wo), CRAZY_HIGH_DISTANCE, 0.f, 0xFFFFFFFF);


        // Update path throughput multiplying by phase function.
        Path_MulThroughput(path, phase);
#else
        // Single-scattering mode only,
        // kill the path and compact away on next iteration
        Path_Kill(path);
        Ray_SetInactive(indirect_rays + global_id);
#endif
    }
}


// Handle ray-surface interaction possibly generating path continuation.
// This is only applied to non-scattered paths.
KERNEL void ShadeSurfaceUberV2(
    // Ray batch
    GLOBAL ray const* restrict rays,
    // Intersection data
    GLOBAL Intersection const* restrict isects,
    // Hit indices
    GLOBAL int const* restrict hit_indices,
    // Pixel indices
    GLOBAL int const* restrict pixel_indices,
    // Output indices
    GLOBAL int const*  restrict output_indices,
    // Number of rays
    GLOBAL int const* restrict num_hits,
    // Vertices
    GLOBAL float3 const* restrict vertices,
    // Normals
    GLOBAL float3 const* restrict normals,
    // UVs
    GLOBAL float2 const* restrict uvs,
    // Indices
    GLOBAL int const* restrict indices,
    // Shapes
    GLOBAL Shape const* restrict shapes,
    // Materials
    GLOBAL int const* restrict material_attributes,
    // Textures
    TEXTURE_ARG_LIST,
    // Environment texture index
    int env_light_idx,
    // Emissives
    GLOBAL Light const* restrict lights,
    // Light distribution
    GLOBAL int const* restrict light_distribution,
    // Number of emissive objects
    int num_lights,
    // RNG seed
    uint rng_seed,
    // Sampler states
    GLOBAL uint* restrict random,
    // Sobol matrices
    GLOBAL uint const* restrict sobol_mat,
    // Current bounce
    int bounce,
    // Frame
    int frame,
    // Volume data
    GLOBAL Volume const* restrict volumes,
    // Shadow rays
    GLOBAL ray* restrict shadow_rays,
    // Light samples
    GLOBAL float3* restrict light_samples,
    // Path throughput
    GLOBAL Path* restrict paths,
    // Indirect rays
    GLOBAL ray* restrict indirect_rays,
    // Radiance
    GLOBAL float3* restrict output,
    GLOBAL InputMapData const* restrict input_map_values
)
{
    int global_id = get_global_id(0);

    Scene scene =
    {
        vertices,
        normals,
        uvs,
        indices,
        shapes,
        material_attributes,
        input_map_values,
        lights,
        env_light_idx,
        num_lights,
        light_distribution
    };

    // Only applied to active rays after compaction
    if (global_id < *num_hits)
    {
        // Fetch index
        int hit_idx = hit_indices[global_id];
        int pixel_idx = pixel_indices[global_id];
        Intersection isect = isects[hit_idx];

        GLOBAL Path* path = paths + pixel_idx;

        // Early exit for scattered paths
        if (Path_IsScattered(path))
        {
            return;
        }

        // Fetch incoming ray direction
        float3 wi = -normalize(rays[hit_idx].d.xyz);

        Sampler sampler;
#if SAMPLER == SOBOL
        uint scramble = random[pixel_idx] * 0x1fe3434f;
        Sampler_Init(&sampler, frame, SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE, scramble);
#elif SAMPLER == RANDOM
        uint scramble = pixel_idx * rng_seed;
        Sampler_Init(&sampler, scramble);
#elif SAMPLER == CMJ
        uint rnd = random[pixel_idx];
        uint scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (CMJ_DIM * CMJ_DIM));
        Sampler_Init(&sampler, frame % (CMJ_DIM * CMJ_DIM), SAMPLE_DIM_SURFACE_OFFSET + bounce * SAMPLE_DIMS_PER_BOUNCE, scramble);
#endif

        // Fill surface data
        DifferentialGeometry diffgeo;
        Scene_FillDifferentialGeometry(&scene, &isect, &diffgeo);

        // Check if we are hitting from the inside
        float ngdotwi = dot(diffgeo.ng, wi);
        bool backfacing = ngdotwi < 0.f;

        // Select BxDF
        UberV2ShaderData uber_shader_data;
        UberV2PrepareInputs(&diffgeo, input_map_values, material_attributes, TEXTURE_ARGS, &uber_shader_data);

        UberV2_ApplyShadingNormal(&diffgeo, &uber_shader_data);
        DifferentialGeometry_CalculateTangentTransforms(&diffgeo);

        GetMaterialBxDFType(wi, &sampler, SAMPLER_ARGS, &diffgeo, &uber_shader_data);

        // Set surface interaction flags
        Path_SetFlags(&diffgeo, path);

        // Opacity flag for opacity AOV
        if (!Bxdf_IsTransparency(&diffgeo))
        {
            Path_SetOpacityFlag(path);
        }

        // Terminate if emissive
        if (Bxdf_IsEmissive(&diffgeo))
        {
            if (!backfacing)
            {
                float weight = 1.f;

                if (bounce > 0 && !Path_IsSpecular(path))
                {
                    float2 extra = Ray_GetExtra(&rays[hit_idx]);
                    float ld = isect.uvwt.w;
                    float denom = fabs(dot(diffgeo.n, wi)) * diffgeo.area;
                    // TODO: num_lights should be num_emissies instead, presence of analytical lights breaks this code
                    float bxdf_light_pdf = denom > 0.f ? (ld * ld / denom / num_lights) : 0.f;
                    weight = extra.x > 0.f ? BalanceHeuristic(1, extra.x, 1, bxdf_light_pdf) : 1.f;
                }

                // In this case we hit after an application of MIS process at previous step.
                // That means BRDF weight has been already applied.
                float3 v = REASONABLE_RADIANCE(Path_GetThroughput(path) * Emissive_GetLe(&diffgeo, TEXTURE_ARGS, &uber_shader_data) * weight);

                int output_index = output_indices[pixel_idx];
                ADD_FLOAT3(&output[output_index], v);
            }

            Path_Kill(path);
            Ray_SetInactive(shadow_rays + global_id);
            Ray_SetInactive(indirect_rays + global_id);

            light_samples[global_id] = 0.f;
            return;
        }

        float s = Bxdf_IsBtdf(&diffgeo) ? (-sign(ngdotwi)) : 1.f;
        if (backfacing && !Bxdf_IsBtdf(&diffgeo))
        {
            //Reverse normal and tangents in this case
            //but not for BTDFs, since BTDFs rely
            //on normal direction in order to arrange
            //indices of refraction
            diffgeo.n = -diffgeo.n;
            diffgeo.dpdu = -diffgeo.dpdu;
            diffgeo.dpdv = -diffgeo.dpdv;
            s = -s;
        }

        float ndotwi = fabs(dot(diffgeo.n, wi));

        float light_pdf = 0.f;
        float bxdf_light_pdf = 0.f;
        float bxdf_pdf = 0.f;
        float light_bxdf_pdf = 0.f;
        float selection_pdf = 0.f;
        float3 radiance = 0.f;
        float3 lightwo;
        float3 bxdfwo;
        float3 wo;
        float bxdf_weight = 1.f;
        float light_weight = 1.f;

        int light_idx = Scene_SampleLight(&scene, Sampler_Sample1D(&sampler, SAMPLER_ARGS), &selection_pdf);

        float3 throughput = Path_GetThroughput(path);

        // Sample bxdf
        const float2 sample = Sampler_Sample2D(&sampler, SAMPLER_ARGS);
        float3 bxdf = UberV2_Sample(&diffgeo, wi, TEXTURE_ARGS, sample, &bxdfwo, &bxdf_pdf, &uber_shader_data);

        // If we have light to sample we can hopefully do mis
        if (light_idx > -1)
        {
            // Sample light
            int bxdf_flags = Path_GetBxdfFlags(path);
            float3 le = Light_Sample(light_idx, &scene, &diffgeo, TEXTURE_ARGS, Sampler_Sample2D(&sampler, SAMPLER_ARGS), bxdf_flags, kLightInteractionSurface, &lightwo, &light_pdf);
            light_bxdf_pdf = UberV2_GetPdf(&diffgeo, wi, normalize(lightwo), TEXTURE_ARGS, &uber_shader_data);
            light_weight = Light_IsSingular(&scene.lights[light_idx]) ? 1.f : BalanceHeuristic(1, light_pdf * selection_pdf, 1, light_bxdf_pdf);

            // Apply MIS to account for both
            if (NON_BLACK(le) && (light_pdf > 0.0f) && (selection_pdf > 0.0f) && !Bxdf_IsSingular(&diffgeo))
            {
                wo = lightwo;
                float ndotwo = fabs(dot(diffgeo.n, normalize(wo)));
                radiance = le * ndotwo * UberV2_Evaluate(&diffgeo, wi, normalize(wo), TEXTURE_ARGS, &uber_shader_data) * throughput * light_weight / light_pdf / selection_pdf;
            }
        }

        // If we have some light here generate a shadow ray
        if (NON_BLACK(radiance))
        {
            // Generate shadow ray
            float3 shadow_ray_o = diffgeo.p + CRAZY_LOW_DISTANCE * s * diffgeo.ng;
            float3 temp = diffgeo.p + wo - shadow_ray_o;
            float3 shadow_ray_dir = normalize(temp);
            float shadow_ray_length = length(temp);
            int shadow_ray_mask = VISIBILITY_MASK_BOUNCE_SHADOW(bounce);

            Ray_Init(shadow_rays + global_id, shadow_ray_o, shadow_ray_dir, shadow_ray_length, 0.f, shadow_ray_mask);
            Ray_SetExtra(shadow_rays + global_id, make_float2(1.f, 0.f));

            light_samples[global_id] = REASONABLE_RADIANCE(radiance);
        }
        else
        {
            // Otherwise save some intersector cycles
            Ray_SetInactive(shadow_rays + global_id);
            light_samples[global_id] = 0;
        }

        // Apply Russian roulette
        float q = max(min(0.5f,
            // Luminance
            0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z), 0.01f);
        // Only if it is 3+ bounce
        bool rr_apply = bounce > 3;
        bool rr_stop = Sampler_Sample1D(&sampler, SAMPLER_ARGS) > q && rr_apply;

        if (rr_apply)
        {
            Path_MulThroughput(path, 1.f / q);
        }

        bxdfwo = normalize(bxdfwo);
        float3 t = bxdf * fabs(dot(diffgeo.n, bxdfwo));

        // Only continue if we have non-zero throughput & pdf
        if (NON_BLACK(t) && bxdf_pdf > 0.f && !rr_stop)
        {
            // Update the throughput
            Path_MulThroughput(path, t / bxdf_pdf);

            // Generate ray
            float3 indirect_ray_dir = bxdfwo;
            float3 indirect_ray_o = diffgeo.p + CRAZY_LOW_DISTANCE * s * diffgeo.ng;
            int indirect_ray_mask = VISIBILITY_MASK_BOUNCE(bounce + 1);

            Ray_Init(indirect_rays + global_id, indirect_ray_o, indirect_ray_dir, CRAZY_HIGH_DISTANCE, 0.f, indirect_ray_mask);
            Ray_SetExtra(indirect_rays + global_id, make_float2(Bxdf_IsSingular(&diffgeo) ? 0.f : bxdf_pdf, 0.f));

            if (Bxdf_IsBtdf(&diffgeo))
            {
                if (backfacing)
                {
                    Path_SetVolumeIdx(path, INVALID_IDX);
                }
                else
                {
                    Path_SetVolumeIdx(path, Scene_GetVolumeIndex(&scene, isect.shapeid - 1));
                }
            }
        }
        else
        {
            // Otherwise kill the path
            Path_Kill(path);
            Ray_SetInactive(indirect_rays + global_id);
        }
    }
}

///< Handle light samples and visibility info and add contribution to final buffer
KERNEL void ApplyVolumeTransmissionUberV2(
    // Pixel indices
    GLOBAL int const* restrict pixel_indices,
    // Output indices
    GLOBAL int const*  restrict output_indices,
    // Shadow rays batch
    GLOBAL ray* restrict shadow_rays,
    // Number of rays
    GLOBAL int* restrict num_rays,
    // Shadow rays hits
    GLOBAL Intersection const* restrict isects,
    // throughput
    GLOBAL Path const* restrict paths,
    // Vertices
    GLOBAL float3 const* restrict vertices,
    // Normals
    GLOBAL float3 const* restrict normals,
    // UVs
    GLOBAL float2 const* restrict uvs,
    // Indices
    GLOBAL int const* restrict indices,
    // Shapes
    GLOBAL Shape const* restrict shapes,
    // Materials
    GLOBAL int const* restrict material_attributes,
    // Volumes
    GLOBAL Volume const* restrict volumes,
    // Light samples
    GLOBAL float3* restrict light_samples,
    // Shadow predicates
    GLOBAL int* restrict shadow_hits,
    // Radiance sample buffer
    GLOBAL float4* restrict output,
    GLOBAL InputMapData const* restrict input_map_values
)
{
    int global_id = get_global_id(0);

    if (global_id < *num_rays)
    {
        int pixel_idx = pixel_indices[global_id];

        // Ray might be inactive, in this case we just 
        // fail an intersection test, nothing has been added for this ray.
        if (Ray_IsActive(&shadow_rays[global_id]))
        {
            Scene scene =
            {
                vertices,
                normals,
                uvs,
                indices,
                shapes,
                material_attributes,
                input_map_values,
                0,
                0,
                0,
                0
            };

            // Get pixel id for this sample set
            int pixel_idx = pixel_indices[global_id];
            GLOBAL Path* path = &paths[pixel_idx];
            int path_volume_idx = Path_GetVolumeIdx(path);

            // Here we do not have any intersections, 
            // so we mark the test passed.
            // OPTIMIZATION: this ray is going to be tested again
            // on the next iteration, we can make it inactive, but
            // in this case inactive rays need special handling and 
            // we can't fail the test for them like condition above does.
            if (isects[global_id].shapeid < 0)
            {
                Ray_SetInactive(&shadow_rays[global_id]);
                shadow_hits[global_id] = -1;
                return;
            }

            // Now we have a hit
            // FIXME: this should be scene functions
            Intersection isect = isects[global_id];
            int shape_idx = isect.shapeid - 1;
            int prim_idx = isect.primid;
            float t = isect.uvwt.w;

            int volume_idx = Scene_GetVolumeIndex(&scene, shape_idx);
            /// @FIXME need to get material params from material_attributes
            int layers = scene.shapes[shape_idx].material.layers;

            // If shape does not have volume, it is a surface intersection
            // and we fail a shadow test and bail out.
            if ((volume_idx == -1) || (!UberV2IsTransmissive(layers) && volume_idx != path_volume_idx))
            {
                shadow_hits[global_id] = 1;
                Ray_SetInactive(&shadow_rays[global_id]);
                return;
            }

            // Here we know volume intersection occured and we need to 
            // interpolate normal to figure out if we are entering or exiting volume
            float3 n;
            Scene_InterpolateNormalsFromIntersection(&scene, &isect, &n);

            ray shadow_ray = shadow_rays[global_id];
            float shadow_ray_throughput = Ray_GetExtra(&shadow_rays[global_id]).x;
            // Now we determine if we are exiting or entering. On exit 
            // we need to apply transmittance and emission, on enter we simply update the ray origin.
            if (dot(shadow_ray.d.xyz, n) > 0.f)
            {
                // Old target point is needed to update t_max
                float3 old_target = shadow_ray.o.xyz + (shadow_ray.o.w) * shadow_ray.d.xyz;
                // This is new ray origin after media boundary intersection
                float3 p = shadow_ray.o.xyz + (t + CRAZY_LOW_DISTANCE) * shadow_ray.d.xyz;

                // Calculate volume transmittance up to this point
                float3 tr = Volume_Transmittance(&volumes[volume_idx], &shadow_rays[global_id], t);
                // Calculat volume emission up to this point
                float3 emission = Volume_Emission(&volumes[volume_idx], &shadow_rays[global_id], t);

                // Multiply light sample by the transmittance of this segment
                light_samples[global_id] *= tr;

                // TODO: this goes directly to output, not affected by a shadow ray, fix me
                if (length(emission) > 0.f)
                {
                    int output_index = output_indices[pixel_idx];
                    float3 v = Path_GetThroughput(path) * emission * tr * shadow_ray_throughput;
                    ADD_FLOAT3(&output[output_index], v);
                }

                shadow_rays[global_id].o.xyz = p;
                shadow_rays[global_id].o.w = length(old_target - p);
                // TODO: we keep average throughput here since we do not have float3 available
                float tr_avg = (tr.x + tr.y + tr.z) / 3.f;
                Ray_SetExtra(&shadow_rays[global_id], make_float2(shadow_ray_throughput * tr_avg, 0.f));
            }
            else
            {
                float3 old_target = shadow_ray.o.xyz + (shadow_ray.o.w) * shadow_ray.d.xyz;
                float3 p = shadow_ray.o.xyz + (t + CRAZY_LOW_DISTANCE) * shadow_ray.d.xyz;

                shadow_rays[global_id].o.xyz = p;
                shadow_rays[global_id].o.w = length(old_target - p);
            }
        }
    }
}


#endif
