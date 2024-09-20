#ifndef CUDA_FLOAT3_OPERATIONS_H
#define CUDA_FLOAT3_OPERATIONS_H

#include <cuda_runtime.h>
#include <math.h>

// Addition
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtraction
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Scalar multiplication
__host__ __device__ inline float3 operator*(const float3 &a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &a)
{
    return a * s;
}

// Scalar division
__host__ __device__ inline float3 operator/(const float3 &a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

// Length
__host__ __device__ inline float length(const float3 &v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Normalization
__host__ __device__ inline float3 normalize(const float3 &v)
{
    float invLen = 1.0f / length(v);
    return v * invLen;
}

// Dot product
__host__ __device__ inline float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product
__host__ __device__ inline float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 1.0);
}

#endif // CUDA_FLOAT3_OPERATIONS_H

// Thank you Cluade Sonnet because I was not writing all that