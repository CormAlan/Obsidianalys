// UTF-8 <-> std::filesystem::path (Windows använder UTF-16 internt).
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace paths {

namespace fs = std::filesystem;

inline std::string to_utf8(const fs::path& p) {
    std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}

inline std::string to_generic_utf8(const fs::path& p) {
    std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

inline fs::path from_utf8(std::string_view s) { return fs::path(std::u8string(s.begin(), s.end())); }

}  // namespace paths
