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

__device__ __host__ void
setColor(unsigned char *image, glm::vec4 color, int x, int y, int width, int height)
{

    image[((y * width) + x) * 4] = max(min(color.x * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 1] = max(min(color.y * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 2] = max(min(color.z * 255, 255.0), 0.0);
    image[((y * width) + x) * 4 + 3] = max(min(color.w * 255, 255.0), 0.0);
}

__host__ __device__ void render(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int x, int y, int width, int height)
{

    double aspect_ratio = (float)width / height;

    double x_scaled = ((float)x / (float)width - 0.5) * 2 * aspect_ratio;
    double y_scaled = ((float)y / (float)height - 0.5) * 2;

    glm::vec3 direction = glm::normalize(glm::mat3(glm::inverse(view_matrix)) * glm::vec3(-x_scaled, -y_scaled, -1.0));
    glm::vec3 start = glm::vec3(view_matrix[3]);

    float t_ = 0.0;
    Object *hit_object;
    glm::vec3 normal;
    glm::vec3 hit_position;

    bool hit = intersectMany(objects, num_objects, start, direction, t_, hit_object, normal, hit_position);

    bool shadow = intersectMany(objects, num_objects, hit_position + normal * glm::vec3(0.01), glm::vec3(0.0, 1.0, 0.0));

    setColor(image, glm::vec4(hit ? hit_object->color * dot(normal, glm::vec3(0.0, 1.0, 0.0)) * glm::vec3(shadow ? 0.1 : 1.0) : glm::vec3(0.0), 1.0), x, y, width, height);
}

__global__ void makeWhite(unsigned char *image, glm::mat4 view_matrix, Object *objects, size_t num_objects, int width, int height, double time)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height)
        return;

    render(image, view_matrix, objects, num_objects, x, y, width, height);
}

int main(int argc, char **argv)
{

    cudaSetDevice(0);

    sf::Clock clock;

    sf::RenderWindow window(sf::VideoMode(1280, 720), "Hello!");
    sf::Texture render_texture;
    render_texture.create(1280, 720);

    window.clear(sf::Color(255, 255, 255, 255));
    window.display();

    sf::Sprite image(render_texture);
    image.setPosition(sf::Vector2f(1280 / 2, 720 / 2));
    image.setOrigin(sf::Vector2f(render_texture.getSize()) / 2.f);

    render_texture.update(window);
    int width = 1280;
    int height = 720;

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
        glm::vec3(1.0),
        ObjectType::Sphere};
    objects[1] = Object{
        glm::translate(glm::mat4(1.0), glm::vec3(0.0, -1.0, 0.0)),
        glm::vec3(1.0),
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

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseMoved)
            {

                view_matrix = glm::rotate(view_matrix, ((float)event.mouseMove.x - (float)last_x) / 100.0f, glm::vec3(0.0, 1.0, 0.0));
                last_x = event.mouseMove.x;
            }
        }

        window.clear();

        std::cout << 1.0f / (float)last_time << std::endl;

        makeWhite<<<numBlocks, threadsperBlock>>>(d_image, view_matrix, d_objects, num_objects, width, height, clock.getElapsedTime().asSeconds());

        // for (int x = 0; x < width; ++x)
        // {
        //     for (int y = 0; y < height; ++y)
        //     {
        //         render(h_image, view_matrix, objects, 1, x, y, width, height);
        //     }
        // }

        cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

        // memset(h_image, (unsigned char)255, image_size);

        render_texture.update(h_image);

        window.draw(image);

        window.display();

        fps_timer.restart();
        last_time = fps_timer.getElapsedTime().asSeconds();
    }

    return 0;
}
