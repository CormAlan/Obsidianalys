// Plattformsberoende bitar: öppna filer/URL:er, konfiguration, kommandorad.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace platform {

namespace fs = std::filesystem;

struct Config {
    std::string root;
    int top_n = 30;
    bool include_equations = false;
    bool include_code = false;
    std::string theme = "system";  // system | light | dark
};

Config load_config();
void save_config(const Config& cfg);
fs::path config_path();

fs::path home_dir();
std::vector<std::string> filesystem_roots();  // "/" eller Windows-enheter

// Öppnar en fil eller URL med systemets standardprogram (xdg-open/open/ShellExecute).
bool open_with_system(const std::string& target);

std::string random_token();

// argv som UTF-8 (på Windows via GetCommandLineW) + UTF-8-konsol.
std::vector<std::string> utf8_args(int argc, char** argv);

}  // namespace platform
