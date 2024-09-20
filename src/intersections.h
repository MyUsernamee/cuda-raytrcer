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
    bool is_light;
    ObjectType type;

    glm::vec3 getPosition()
    {
        return transform[3];
    }
};

struct IntersectionData
{
    float t0;
    float t1;
    glm::vec3 normal;
    glm::vec3 position;
    bool hit;
};

struct HitData
{
    float t;
    glm::vec3 normal;
    glm::vec3 position;
    Object *object;
    bool hit;
};

__host__ __device__ IntersectionData intersectSphere(glm::vec3 start, glm::vec3 direction)
{

    IntersectionData id;

    glm::vec3 L = -start;
    float tca = dot(L, direction);
    if (tca < 0)
    {
        id.hit = false;
        return id;
    }
    float d2 = dot(L, L) - tca * tca;
    if (d2 > 1)
    {
        id.hit = false;
        return id;
    }
    float thc = sqrt(1 - d2);
    id.t0 = tca - thc;
    id.position = start + direction * id.t0;
    id.normal = normalize(id.position);
    id.hit = true;

    return id;
}

__host__ __device__ IntersectionData intersectPlane(glm::vec3 start, glm::vec3 direction)
{

    IntersectionData id;

    double L = dot(start, glm::vec3(0.0, 1.0, 0.0));
    double B = dot(glm::vec3(0.0, 1.0, 0.0), direction);

    if (B == 0)
    {
        id.hit = false;
        return id;
    }

    id.normal = glm::vec3(0.0, 1.0, 0.0);
    id.hit = true;
    id.t0 = L / B;
    id.t1 = id.t0;
    id.position = start + direction * id.t0;
    id.hit = id.t0 > 0;
    return id;
}

__host__ __device__ IntersectionData intersectObject(Object object, glm::vec3 start, glm::vec3 direction)
{
    glm::vec4 local_start = (object.transform) * glm::vec4(start, 1.0);
    direction = glm::mat3(object.transform) * direction;

    IntersectionData hit;
    switch (object.type)
    {

    case ObjectType::Sphere:

        hit = intersectSphere(local_start, direction);
        break;

    case ObjectType::Plane:

        hit = intersectPlane(local_start, direction);
        break;
    }

    hit.normal = glm::inverse(glm::mat3(object.transform)) * hit.normal;
    hit.position = glm::vec3(glm::inverse(object.transform) * glm::vec4(hit.position, 1.0));
    return hit;
}

__host__ __device__ HitData intersectMany(Object *objects, size_t count, glm::vec3 start, glm::vec3 direction)
{

    HitData hd;

    float closest = MAXFLOAT;
    Object *closest_object;
    hd.hit = false;

    for (int i = 0; i < count; ++i)
    {

        auto id = intersectObject(objects[i], start, direction);

        if (id.hit && id.t0 < closest)
        {
            closest = id.t0;
            closest_object = objects + i;
            hd.position = id.position;
            hd.normal = id.normal;
            hd.hit = true;
        }
    }

    hd.object = closest_object;
    hd.t = closest;
    return hd;
}
