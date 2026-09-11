// Lokal HTTP-server: JSON-API + inbäddat React-gränssnitt.
#pragma once

#include <string>

namespace server {

struct Options {
    int port = 7331;            // provar port, port+1, … port+19
    bool open_browser = true;
    bool dev = false;           // fast token "dev" för Vite-proxyn
    std::string web_dir;        // servera frontend från disk i stället för inbäddat
    std::string initial_root;   // vault från kommandoraden
    unsigned threads = 0;
};

int run(const Options& opts);

}  // namespace server
