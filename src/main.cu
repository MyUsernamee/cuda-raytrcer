#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <GL/gl.h>
#include <math_functions.h>
#include "math.h"
#include "intersections.h"
#include <glm/gtc/matrix_transform.hpp>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

__device__ float rand(curandState *state)
{
    return curand_uniform(state);
}

__global__ void init_curand_state(curandState *states, int numBlocks, int seed)
{

    int id = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockDim.y * blockIdx.y) * numBlocks * blockDim.x;

    curand_init(seed, id, 0, &states[id]);
};

__device__ glm::vec3 rand_vec(curandState_t *state)
{
    return glm::vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
}

__device__ __host__ void setColor(unsigned char *image, glm::vec4 color, int x, int y, int width, int height)
{

    image[((y * width) + x) * 4] = max(min(color.x * 255.0, 255.0), 0.0);
    image[((y * width) + x) * 4 + 1] = max(min(color.y * 255.0, 255.0), 0.0);
    image[((y * width) + x) * 4 + 2] = max(min(color.z * 255.0, 255.0), 0.0);
    image[((y * width) + x) * 4 + 3] = max(min(color.w * 255.0, 255.0), 0.0);
}

__host__ __device__ glm::vec3 sample(float u1, float u2)
{

    float r = sqrt(u1);
    float theta = 2 * M_PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return glm::vec3(x, y, sqrt(max(0.0f, 1.0 - u1)));
}

__host__ __device__ glm::vec3 reflect(glm::vec3 incoming, glm::vec3 normal)
{
    return -dot(incoming, normal) * normal * 2.0f + incoming;
}

__host__ __device__ glm::vec3 rotate_towards(glm::vec3 reference, glm::vec3 target)
{

    if (reference.y == 1.0)
    {
        return glm::vec3(1.0, 0.0, 0.0) * target.y + glm::vec3(0.0, 1.0, 0.0) * target.x + reference * target.z;
    }

    glm::vec3 right = glm::cross(reference, glm::vec3(0.0, 1.0, 0.0));
    glm::vec3 up = glm::cross(right, reference);

    return right * target.x + up * target.y + reference * target.z;
}

__device__ glm::vec3 generateSample(Object *object, glm::vec3 incoming, glm::vec3 normal, curandState *state)
{

    glm::vec3 diffuse = rotate_towards(normal, sample(rand(state), rand(state)));
    glm::vec3 reflection = reflect(incoming, normal);

    return reflection * (1.0f - object->roughness) + diffuse * (object->roughness);
}

__device__ float pdf(glm::vec3 incoming, glm::vec3 normal, glm::vec3 outgoing, float roughness)
{

    glm::vec3 true_outgoing = reflect(incoming, normal);

    float area = roughness;
    float max_angle = (M_PI - M_PI * (1 - roughness));

    float angle = acos(dot(outgoing, true_outgoing));

    return (angle < max_angle) ? (1.0f / area) : 0.0f;
}

__device__ glm::vec3 trace(glm::vec3 start, glm::vec3 direction, Object *objects, size_t *lights, size_t num_lights, size_t num_objects, curandState *state)
{

    glm::vec3 accumulated_light = glm::vec3(0.0);
    glm::vec3 bounced_light = glm::vec3(1.0);

    for (int bounce = 0; bounce < 3; ++bounce)
    {
        auto hit = intersectMany(objects, num_objects, start, direction);

        if (!hit.hit)
            break;

        if (hit.object->is_light)
        {
            accumulated_light = bounce == 0 ? bounced_light : accumulated_light;

            break;
        }

        bounced_light *= hit.object->color;

        glm::vec3 light = glm::vec3(0.0);

        for (int light_index = 0; light_index < num_lights; light_index++)
        {

            auto light_position = rand_vec(state) * 2.0f - glm::vec3(1.0);
            light_position = (objects[lights[light_index]]).transform * glm::vec4(light_position, 1.0);

            auto id = intersectMany(objects, num_objects, hit.position + hit.normal * 0.001f, glm::normalize(light_position - hit.position));

            glm::vec3 light_direction = glm::normalize(id.position - hit.position);

            if (id.hit && id.object == objects + lights[light_index])
            {
                light += id.object->color / (id.t * id.t) * dot(hit.normal, light_direction) * pdf(light_direction, hit.normal, direction, hit.object->roughness);
            }
        }

        accumulated_light += bounced_light * light;

        start = hit.position;
        direction = generateSample(hit.object, direction, hit.normal, state);
    }

    return accumulated_light;
}

__device__ void render(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int x, int y, int width, int height, curandState *state)
{

    double aspect_ratio = (float)width / height;

    double x_scaled = ((float)x / (float)width - 0.5) * 2 * aspect_ratio;
    double y_scaled = ((float)y / (float)height - 0.5) * 2;

    glm::vec3 direction = glm::normalize(glm::mat3(glm::inverse(view_matrix)) * glm::vec3(-x_scaled, -y_scaled, 1.0));
    glm::vec3 start = glm::vec3(glm::inverse(view_matrix)[3]);

    glm::vec3 color = glm::vec3(0.0);

    size_t lights[8];
    size_t num_lights = 0;

    for (int i = 0; i < num_objects; i++)
    {

        if (objects[i].is_light)
        {

            lights[num_lights] = i;
            num_lights++;
        }
    }

    for (int i = 0; i < 4; i++)
    {
        color += trace(start, direction, objects, lights, num_lights, num_objects, state);
    }

    setColor(image, glm::vec4(color / 4.0f, 1.0), x, y, width, height);
}

__global__ void makeWhite(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int width, int height, int numBlocks, curandState *states)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int id = x + y * (numBlocks * blockDim.x);

    if (x > width || y > height)
        return;

    render(image, view_matrix, objects, num_objects, x, y, width, height, states + id);
}

int main(int argc, char **argv)
{

    cudaSetDevice(0);

    sf::Clock clock;
    int width = 1280;
    int height = 720;

    sf::RenderWindow window(sf::VideoMode(width, height), "Hello!");
    sf::Texture render_texture;
    render_texture.create(width, height);

    window.clear(sf::Color(255, 255, 255, 255));
    window.display();

    sf::Sprite image(render_texture);
    image.setPosition(sf::Vector2f(width / 2, height / 2));
    image.setOrigin(sf::Vector2f(render_texture.getSize()) / 2.f);

    render_texture.update(window);

    size_t image_size = width * height * sizeof(char) * 4;

    // Do something
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x, (height + threadsperBlock.y - 1) / threadsperBlock.y);

    unsigned char *h_image = new unsigned char[image_size];
    unsigned char *d_image;
    cudaMalloc(&d_image, image_size);

    glm::mat4 view_matrix = glm::lookAtLH(glm::vec3(0.0, 0.0, -3.0), glm::vec3(0.0), glm::vec3(0.0, 1.0, 0.0));
    // Goes from World to local

    float _;
    glm::vec3 _2;

    static const size_t num_objects = 4;

    Object *objects = new Object[num_objects];

    objects[0] = Object{
        glm::mat4(1.0),
        glm::vec3(1.0, 0.0, 0.0),
        0.1,
        false,
        ObjectType::Sphere};
    objects[1] = Object{
        glm::scale(glm::translate(glm::mat4(1.0), glm::vec3(0.0, -1.0, 0.0)), glm::vec3(10.0f)),
        glm::vec3(1.0),
        1.0,
        false,
        ObjectType::Plane};
    objects[2] = Object{
        glm::translate(glm::mat4(1.0), glm::vec3(0.0, 4.0, 0.0)),
        glm::vec3(1.0, 1.0, 1.0) * 10.0f,
        1.0,
        true,
        ObjectType::Sphere};
    objects[3] = Object{
        glm::inverse(glm::lookAtLH(glm::vec3(-2.0, 2.0, -2.0), glm::vec3(0.0), glm::vec3(0.0, 1.0, 0.0))),
        glm::vec3(1.0),
        0.001,
        false,
        ObjectType::Plane};

    Object *d_objects;
    cudaMalloc(&d_objects, sizeof(Object) * num_objects);

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState) * numBlocks.x * threadsperBlock.x * numBlocks.y * threadsperBlock.y);

    printf("Init random state");
    init_curand_state<<<numBlocks, threadsperBlock>>>(d_state, numBlocks.x, 1234);
    printf("Done");

    cudaMemcpy(d_objects, objects, num_objects * sizeof(Object), cudaMemcpyHostToDevice);

    sf::Vector2i mouse_delta;

    sf::Clock fps_timer;
    double last_time = 0.0;

    sf::Font font;
    font.loadFromFile("/usr/share/fonts/opentype/roboto/slab/RobotoSlab-Blod.otd");

    window.setMouseCursorGrabbed(true);
    sf::Mouse::setPosition(window.getPosition() + sf::Vector2i(window.getSize()) / 2);

    // Print view matrix
    printf("%f %f %f %f\n", view_matrix[0][0], view_matrix[0][1], view_matrix[0][2], view_matrix[0][3]);
    printf("%f %f %f %f\n", view_matrix[1][0], view_matrix[1][1], view_matrix[1][2], view_matrix[1][3]);
    printf("%f %f %f %f\n", view_matrix[2][0], view_matrix[2][1], view_matrix[2][2], view_matrix[2][3]);
    printf("%f %f %f %f\n", view_matrix[3][0], view_matrix[3][1], view_matrix[3][2], view_matrix[3][3]);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseMoved)
            {
            }
        }

        if (window.hasFocus())
        {
            sf::Vector2i center(window.getSize().x / 2, window.getSize().y / 2);
            mouse_delta = sf::Mouse::getPosition(window) - center;
            sf::Mouse::setPosition(center, window);

            glm::mat4 rotation = glm::mat4(glm::rotate(glm::mat4(1.0), mouse_delta.y / 100.f, glm::vec3(1.0, 0.0, 0.0)));
            rotation = glm::mat4(glm::rotate(rotation, mouse_delta.x / 100.f, glm::vec3(0.0, 1.0, 0.0)));

            view_matrix = rotation * view_matrix;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        {
            view_matrix = glm::translate(view_matrix, glm::inverse(glm::mat3(view_matrix)) * glm::vec3(0.0, 0.0, -0.01));
        }

        window.clear();

        // std::cout << 1.0f / (float)last_time << std::endl;

        makeWhite<<<numBlocks, threadsperBlock>>>(d_image, view_matrix, d_objects, num_objects, width, height, numBlocks.x, d_state);

        // for (int x = 0; x < width; ++x)
        // {
        //     for (int y = 0; y < height; ++y)
        //     {
        //         render(h_image, view_matrix, objects, num_objects, x, y, width, height, 0.0);
        //     }
        // }

        cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

        // memset(h_image, (unsigned char)255, image_size);

        render_texture.update(h_image);

        window.draw(image);

        window.display();

        last_time = fps_timer.getElapsedTime().asSeconds();
        fps_timer.restart();
    }

    return 0;
}
