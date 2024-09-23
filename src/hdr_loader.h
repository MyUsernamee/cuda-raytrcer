#pragma once

#include <fstream>
#include <istream>
#include <iomanip>

struct HDRImage
{
    float *data;
    unsigned int width;
    unsigned int height;
    std::string data_type;
};

std::string pop(std::ifstream &a, size_t count)
{

    std::string sub_string;
    sub_string.resize(count);
    a.read(sub_string.data(), count);

    return sub_string;
}

std::string pop_until(std::ifstream &a, char *end)
{

    std::string s;
    char c;
    a >> std::noskipws >> c;
    s.append(&c);

    while (c != *end)
    {
        a >> std::noskipws >> c;
        s.append(&c);
    }

    return s;
}

float *readHDR(std::ifstream &file)
{

    // First we make sure we are actually reading a radiance file format
    auto header_format = pop(file, 10);

    if (header_format != "#?RADIANCE")
        return nullptr; // Failed :(

    // Good we are in a radiance file
    HDRImage img;

    // Next we need to read the file type :)
    // Discard "FORMAT="
    auto _ = pop(file, 8);

    // Next is the format which we need to read to a 0A of a \n
    auto type = pop_until(file, "\n");
    type.pop_back();
};