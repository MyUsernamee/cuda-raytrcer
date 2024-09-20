#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "math.h"
#include "mat4.h"

#include <glm/glm.hpp>

enum ObjectType
{

    Sphere,
    Plane

};

struct Object
{

    glm::mat4 transform;
    glm::vec3 color;
    ObjectType type;

    glm::vec3 getPosition()
    {
        return transform[3];
    }
};

__host__ __device__ bool intersectSphere(glm::vec3 start, glm::vec3 direction, glm::vec3 position, float radius, float &t0, float &t1, glm::vec3 &normal, glm::vec3 &hit_position)
{
    glm::vec3 L = position - start;
    float tca = dot(L, direction);
    if (tca < 0)
        return false;
    float d2 = dot(L, L) - tca * tca;
    if (d2 > radius * radius)
        return false;
    float thc = sqrt(radius * radius - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    hit_position = position + direction * t0;
    normal = normalize(hit_position - position);

    return true;
}

__host__ __device__ bool intersectPlane(glm::vec3 start, glm::vec3 direction, glm::vec3 normal, float offset, float &t0, glm::vec3 &position)
{
    double L = dot(start, normal) + offset;
    double B = dot(normal, direction);

    if (B == 0)
        return false;

    t0 = L / B;
    position = start + direction * t0;
    return t0 > 0;
}

__host__ __device__ bool intersectObject(Object object, glm::vec3 start, glm::vec3 direction, float &t, glm::vec3 &normal, glm::vec3 &position)
{
    glm::vec4 local_start = (object.transform) * glm::vec4(start, 1.0);
    direction = glm::mat3(object.transform) * direction;

    float _t = 0;
    bool hit = false;
    switch (object.type)
    {

    case ObjectType::Sphere:

        hit = intersectSphere(local_start, direction, glm::vec3(0.0), 1.0, t, _t, normal, position);
        break;

    case ObjectType::Plane:

        hit = intersectPlane(local_start, direction, glm::vec3(0.0, 1.0, 0.0), 0.0, t, position);
        normal = glm::vec3(0.0, 1.0, 0.0);
        break;
    }

    normal = glm::inverse(glm::mat3(object.transform)) * normal;
    position = glm::vec3(glm::inverse(object.transform) * glm::vec4(position, 1.0));
    return hit;
}

__host__ __device__ bool intersectObject(Object object, glm::vec3 start, glm::vec3 direction)
{

    float t_;
    glm::vec3 normal_;
    glm::vec3 position_;

    return intersectObject(object, start, direction, t_, normal_, position_);
}

__host__ __device__ bool intersectMany(Object *objects, size_t count, glm::vec3 start, glm::vec3 direction, float &t, Object *&hit_object, glm::vec3 &normal, glm::vec3 &position)
{

    float closest = MAXFLOAT;
    Object *closest_object;
    bool hit = false;

    for (int i = 0; i < count; ++i)
    {

        glm::vec3 normal_;
        glm::vec3 position_;

        if (intersectObject(objects[i], start, direction, t, normal_, position_) && t < closest)
        {
            closest = t;
            closest_object = objects + i;
            normal = normal_;
            position = position_;
            hit = true;
        }
    }

    hit_object = closest_object;
    t = closest;
    return hit;
}

__host__ __device__ bool intersectMany(Object *objects, size_t count, glm::vec3 start, glm::vec3 direction)
{

    float t_;
    Object *ob;
    glm::vec3 n;
    glm::vec3 p;

    return intersectMany(objects, count, start, direction, t_, ob, n, p);
}