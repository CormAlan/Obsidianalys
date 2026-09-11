// Frontendfiler inbäddade i binären (genereras av web/scripts/embed.mjs).
#pragma once

#include <cstddef>

namespace web {

struct Asset {
    const char* path;  // t.ex. "/index.html"
    const char* mime;
    const unsigned char* data;
    std::size_t size;
};

extern const Asset kAssets[];
extern const std::size_t kAssetCount;

}  // namespace web
