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

__host__ __device__ float rand(glm::vec3 co)
{
    float a_;
    return modf((float)(sin(dot(co, glm::vec3(12415.9898, 7318.233, 16126.2512))) * 43758.5453), &a_);
}

__device__ __host__ void
setColor(unsigned char *image, glm::vec4 color, int x, int y, int width, int height)
{

    image[((y * width) + x) * 4] = max(min(color.x * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 1] = max(min(color.y * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 2] = max(min(color.z * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 3] = max(min(color.w * 255, 255.0), 0.0);
}

__host__ __device__ glm::vec3 sample(float u1, float u2)
{

    float r = sqrt(u1);
    float theta = 2 * M_PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return glm::vec3(x, y, sqrt(max(0.0f, 1.0 - u1)));
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

__host__ __device__ glm::vec3 trace(glm::vec3 start, glm::vec3 direction, Object *objects, size_t num_objects, float seed)
{

    glm::vec3 accumulated_light = glm::vec3(0.0);
    glm::vec3 bounced_light = glm::vec3(1.0);

    for (int bounce = 0; bounce < 8; ++bounce)
    {
        auto hit = intersectMany(objects, num_objects, start, direction);

        if (!hit.hit)
            break;

        bounced_light *= hit.object->color;

        auto shadow_hit = intersectMany(objects, num_objects, hit.position + hit.normal * 0.00001f, glm::vec3(0.0, 1.0, 0.0));

        accumulated_light += bounced_light * (shadow_hit.hit ? 0.0f : glm::dot(hit.normal, glm::vec3(0.0, 1.0, 0.0)));

        start = hit.position;
        direction = rotate_towards(hit.normal, sample(rand(hit.position + seed), rand(glm::vec3(hit.position.x + seed, rand(hit.position + seed), hit.position.z))));
    }

    return accumulated_light;
}

__host__ __device__ void render(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int x, int y, int width, int height, double time)
{

    double aspect_ratio = (float)width / height;

    double x_scaled = ((float)x / (float)width - 0.5) * 2 * aspect_ratio;
    double y_scaled = ((float)y / (float)height - 0.5) * 2;

    glm::vec3 direction = glm::normalize(glm::mat3(glm::inverse(view_matrix)) * glm::vec3(-x_scaled, -y_scaled, -1.0));
    glm::vec3 start = glm::vec3(view_matrix[3]);

    glm::vec3 color = glm::vec3(0.0);

    for (int i = 0; i < 10; i++)
    {

        color += trace(start, direction, objects, num_objects, i + time);
    }

    setColor(image, glm::vec4(color / 10.0f, 1.0), x, y, width, height);
}

__global__ void makeWhite(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int width, int height, double time)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height)
        return;

    render(image, view_matrix, objects, num_objects, x, y, width, height, time);
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

    glm::mat4 view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, -3.0), glm::vec3(0.0), glm::vec3(0.0, 1.0, 0.0));

    float _;
    glm::vec3 _2;

    static const size_t num_objects = 2;

    Object *objects = new Object[num_objects];

    objects[0] = Object{
        glm::mat4(1.0),
        glm::vec3(1.0, 0.1, 0.1),
        false,
        ObjectType::Sphere};
    objects[1] = Object{
        glm::translate(glm::mat4(1.0), glm::vec3(0.0, -1.0, 0.0)),
        glm::vec3(1.0),
        false,
        ObjectType::Plane};

    Object *d_objects;
    cudaMalloc(&d_objects, sizeof(Object) * num_objects);

    cudaMemcpy(d_objects, objects, num_objects * sizeof(Object), cudaMemcpyHostToDevice);

    float last_x = 0;
    float last_y = 0;

    sf::Clock fps_timer;
    double last_time = 0.0;

    sf::Font font;
    font.loadFromFile("/usr/share/fonts/opentype/roboto/slab/RobotoSlab-Blod.otd");

    window.setMouseCursorGrabbed(true);
    sf::Mouse::setPosition(window.getPosition() + sf::Vector2i(window.getSize()) / 2);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseMoved)
            {

                view_matrix = glm::rotate(view_matrix, -((float)event.mouseMove.x - (float)last_x) * 0.005f, glm::vec3(0.0, 1.0, 0.0));
                view_matrix = glm::rotate(view_matrix, ((float)event.mouseMove.y - (float)last_y) * 0.005f, glm::vec3(glm::inverse(view_matrix)[0]));

                last_x = event.mouseMove.x;
                last_y = event.mouseMove.y;
            }
        }

        window.clear();

        std::cout << 1.0f / (float)last_time << std::endl;

        makeWhite<<<numBlocks, threadsperBlock>>>(d_image, view_matrix, d_objects, num_objects, width, height, clock.getElapsedTime().asSeconds());

        // for (int x = 0; x < width; ++x)
        // {
        //     for (int y = 0; y < height; ++y)
        //     {
        //         render(h_image, view_matrix, objects, num_objects, x, y, width, height);
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
