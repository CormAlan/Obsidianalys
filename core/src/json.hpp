// Minimal JSON-skrivare – vi producerar bara JSON, parsar den aldrig.
#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

#include "text.hpp"

namespace json {

inline void str(std::string& out, std::string_view raw) {
    // Filnamn på Linux kan vara ogiltig UTF-8 – JSON måste vara giltig
    const std::string s = text::sanitize_utf8(std::string(raw));
    static constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (char ch : s) {
        auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(kHex[c >> 4]);
                    out.push_back(kHex[c & 0xF]);
                } else {
                    out.push_back(ch);
                }
        }
    }
    out.push_back('"');
}

inline void num(std::string& out, int64_t v) {
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof buf, v);
    out.append(buf, r.ptr);
}

inline void num(std::string& out, double v) {
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof buf, v);  // kortaste exakta representationen
    out.append(buf, r.ptr);
}

inline void boolean(std::string& out, bool v) { out += v ? "true" : "false"; }

}  // namespace json
