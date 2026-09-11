#include "platform.hpp"

#include <cstdlib>
#include <fstream>
#include <random>
#include <thread>

#include "paths.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace platform {
namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string narrow(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}
#endif

}  // namespace

fs::path home_dir() {
#ifdef _WIN32
    std::string h = env("USERPROFILE");
#else
    std::string h = env("HOME");
#endif
    return h.empty() ? fs::current_path() : paths::from_utf8(h);
}

fs::path config_path() {
#if defined(_WIN32)
    std::string appdata = env("APPDATA");
    fs::path dir = appdata.empty() ? home_dir() / "Obsidianalys" : paths::from_utf8(appdata) / "Obsidianalys";
#elif defined(__APPLE__)
    fs::path dir = home_dir() / "Library" / "Application Support" / "Obsidianalys";
#else
    std::string xdg = env("XDG_CONFIG_HOME");
    fs::path dir = (xdg.empty() ? home_dir() / ".config" : paths::from_utf8(xdg)) / "obsidianalys";
#endif
    return dir / "config.ini";
}

Config load_config() {
    Config cfg;
    std::ifstream in(config_path());
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (key == "root") cfg.root = value;
        else if (key == "top_n") cfg.top_n = std::max(1, std::atoi(value.c_str()));
        else if (key == "include_equations") cfg.include_equations = value == "1";
        else if (key == "include_code") cfg.include_code = value == "1";
        else if (key == "theme" && (value == "light" || value == "dark" || value == "system")) cfg.theme = value;
    }
    return cfg;
}

void save_config(const Config& cfg) {
    std::error_code ec;
    fs::path p = config_path();
    fs::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::trunc);
    out << "root=" << cfg.root << '\n'
        << "top_n=" << cfg.top_n << '\n'
        << "include_equations=" << (cfg.include_equations ? 1 : 0) << '\n'
        << "include_code=" << (cfg.include_code ? 1 : 0) << '\n'
        << "theme=" << cfg.theme << '\n';
}

std::vector<std::string> filesystem_roots() {
#ifdef _WIN32
    std::vector<std::string> roots;
    DWORD mask = GetLogicalDrives();
    for (int d = 0; d < 26; ++d)
        if (mask & (1u << d)) roots.push_back(std::string(1, char('A' + d)) + ":\\");
    return roots;
#else
    return {"/"};
#endif
}

bool open_with_system(const std::string& target) {
#ifdef _WIN32
    auto r = reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", widen(target).c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    return r > 32;
#else
#ifdef __APPLE__
    const char* tool = "open";
#else
    const char* tool = "xdg-open";
#endif
    // Anroparen skickar alltid absoluta sökvägar eller http-URL:er, aldrig något som
    // kan tolkas som en flagga. Inget skal inblandat.
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
    char* args[] = {const_cast<char*>(tool), const_cast<char*>(target.c_str()), nullptr};
    pid_t pid;
    int rc = posix_spawnp(&pid, tool, &actions, nullptr, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return false;
    std::thread([pid] {
        int status;
        waitpid(pid, &status, 0);
    }).detach();
    return true;
#endif
}

std::string random_token() {
    std::random_device rd;
    static constexpr char kHex[] = "0123456789abcdef";
    std::string token;
    for (int i = 0; i < 8; ++i) {
        uint32_t x = rd();
        for (int k = 0; k < 4; ++k, x >>= 8) {
            token.push_back(kHex[(x >> 4) & 0xF]);
            token.push_back(kHex[x & 0xF]);
        }
    }
    return token;
}

std::vector<std::string> utf8_args(int argc, char** argv) {
#ifdef _WIN32
    (void)argc;
    (void)argv;
    SetConsoleOutputCP(CP_UTF8);
    int n = 0;
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> args;
    for (int i = 0; i < n; ++i) args.push_back(narrow(wargs[i]));
    LocalFree(wargs);
    return args;
#else
    return std::vector<std::string>(argv, argv + argc);
#endif
}

}  // namespace platform
