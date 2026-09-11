#include "text.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "unicode_tables.hpp"

namespace text {
namespace {

constexpr std::array<bool, 128> make_ascii_word() {
    std::array<bool, 128> t{};
    for (int c = '0'; c <= '9'; ++c) t[c] = true;
    for (int c = 'A'; c <= 'Z'; ++c) t[c] = true;
    for (int c = 'a'; c <= 'z'; ++c) t[c] = true;
    t['_'] = true;
    return t;
}
constexpr auto kAsciiWord = make_ascii_word();

template <std::size_t N>
bool in_ranges(const uni::Range (&rs)[N], char32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (rs[mid].hi < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < N && rs[lo].lo <= cp;
}

template <std::size_t N>
const uni::Mapping* find_mapping(const uni::Mapping (&ms)[N], char32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (ms[mid].from < cp) lo = mid + 1;
        else hi = mid;
    }
    return (lo < N && ms[lo].from == cp) ? &ms[lo] : nullptr;
}

// Längd på en giltig UTF-8-sekvens som börjar i s[i], eller antal byte som ska
// kastas (negativt) om den är ogiltig – samma byte som Pythons "ignore" tappar.
int utf8_seq(const unsigned char* s, std::size_t i, std::size_t n) {
    unsigned char c = s[i];
    if (c < 0x80) return 1;
    int len;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c == 0xE0) { len = 3; lo = 0xA0; }
    else if (c >= 0xE1 && c <= 0xEC) len = 3;
    else if (c == 0xED) { len = 3; hi = 0x9F; }
    else if (c >= 0xEE && c <= 0xEF) len = 3;
    else if (c == 0xF0) { len = 4; lo = 0x90; }
    else if (c >= 0xF1 && c <= 0xF3) len = 4;
    else if (c == 0xF4) { len = 4; hi = 0x8F; }
    else return -1;
    for (int k = 1; k < len; ++k) {
        if (i + k >= n) return -k;
        unsigned char b = s[i + k];
        unsigned char l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
        if (b < l || b > h) return -k;
    }
    return len;
}

// Tar bort alla icke-överlappande par av avgränsaren d, som en lat regex d(.*?)d
// med DOTALL. Täcker alla fyra strip-mönstren ($$, $, ```, `).
std::string remove_delimited(std::string_view s, std::string_view d) {
    std::string out;
    out.reserve(s.size());
    std::size_t pos = 0;
    for (;;) {
        std::size_t a = s.find(d, pos);
        if (a == std::string_view::npos) break;
        std::size_t b = s.find(d, a + d.size());
        if (b == std::string_view::npos) break;
        out.append(s.substr(pos, a - pos));
        pos = b + d.size();
    }
    out.append(s.substr(pos));
    return out;
}

template <std::size_t N>
std::string map_string(std::string_view s, const uni::Mapping (&table)[N], bool ascii_lower) {
    std::string out;
    out.reserve(s.size());
    const char* p = s.data();
    const char* end = p + s.size();
    while (p < end) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (c < 0x80) {
            out.push_back((ascii_lower && c >= 'A' && c <= 'Z') ? char(c + 32) : char(c));
            ++p;
            continue;
        }
        char32_t cp = decode(p, end);
        if (const uni::Mapping* m = find_mapping(table, cp)) {
            for (int k = 0; k < m->n; ++k) append_utf8(out, m->to[k]);
        } else {
            append_utf8(out, cp);
        }
    }
    return out;
}

bool iends_with_ascii(std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    std::string_view tail = s.substr(s.size() - suffix.size());
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        char a = tail[i];
        if (a >= 'A' && a <= 'Z') a = char(a + 32);
        if (a != suffix[i]) return false;
    }
    return true;
}

constexpr std::string_view kImageExts[] = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
                                           ".bmp", ".tif", ".tiff", ".ico", ".heic"};

// Pythons str.splitlines()-radbrytningar
bool is_line_break(char32_t cp) {
    return cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C || cp == 0x1C || cp == 0x1D ||
           cp == 0x1E || cp == 0x85 || cp == 0x2028 || cp == 0x2029;
}

}  // namespace

std::string sanitize_utf8(std::string raw) {
    const auto* s = reinterpret_cast<const unsigned char*>(raw.data());
    const std::size_t n = raw.size();
    std::size_t i = 0;
    while (i < n) {  // snabbspår: hitta första felet
        int len = utf8_seq(s, i, n);
        if (len < 0) break;
        i += len;
    }
    if (i == n) return raw;

    std::string out(raw, 0, i);
    while (i < n) {
        int len = utf8_seq(s, i, n);
        if (len < 0) {
            i += -len;
            continue;
        }
        out.append(raw, i, len);
        i += len;
    }
    return out;
}

char32_t decode(const char*& p, const char* end) {
    auto c = static_cast<unsigned char>(*p++);
    if (c < 0x80) return c;
    int extra = (c >= 0xF0) ? 3 : (c >= 0xE0) ? 2 : 1;
    char32_t cp = c & (0x3F >> extra);
    for (int k = 0; k < extra && p < end; ++k) cp = (cp << 6) | (static_cast<unsigned char>(*p++) & 0x3F);
    return cp;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(char(cp));
    } else if (cp < 0x800) {
        out.push_back(char(0xC0 | (cp >> 6)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(char(0xE0 | (cp >> 12)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18)));
        out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    }
}

bool is_word(char32_t cp) { return cp < 128 ? kAsciiWord[cp] : in_ranges(uni::kWord, cp); }

bool is_space(char32_t cp) {
    if (cp < 128) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F);
    return in_ranges(uni::kSpace, cp);
}

std::string strip_equations(std::string_view s) { return remove_delimited(remove_delimited(s, "$$"), "$"); }

std::string strip_code(std::string_view s) { return remove_delimited(remove_delimited(s, "```"), "`"); }

int64_t count_words(std::string_view s) {
    int64_t count = 0;
    bool in_word = false;
    const char* p = s.data();
    const char* end = p + s.size();
    while (p < end) {
        auto c = static_cast<unsigned char>(*p);
        bool w;
        if (c < 0x80) {
            w = kAsciiWord[c];
            ++p;
        } else {
            w = is_word(decode(p, end));
        }
        if (w && !in_word) ++count;
        in_word = w;
    }
    return count;
}

std::string lower(std::string_view s) { return map_string(s, uni::kLower, true); }

std::string casefold(std::string_view s) { return map_string(s, uni::kCasefold, true); }

std::string_view strip(std::string_view s) {
    const char* b = s.data();
    const char* e = b + s.size();
    while (b < e) {
        const char* q = b;
        if (!is_space(decode(q, e))) break;
        b = q;
    }
    // Baklänges: hoppa till början av sista kodpunkten
    while (e > b) {
        const char* start = e - 1;
        while (start > b && (static_cast<unsigned char>(*start) & 0xC0) == 0x80) --start;
        const char* q = start;
        if (!is_space(decode(q, e))) break;
        e = start;
    }
    return {b, static_cast<std::size_t>(e - b)};
}

std::string normalize_note_name(std::string_view s) {
    s = strip(s);
    std::string out;
    out.reserve(s.size());
    const char* p = s.data();
    const char* end = p + s.size();
    bool in_space = false;
    while (p < end) {
        const char* start = p;
        char32_t cp = decode(p, end);
        if (is_space(cp)) {
            if (!in_space) out.push_back(' ');
            in_space = true;
        } else {
            out.append(start, p - start);
            in_space = false;
        }
    }
    return out;
}

std::vector<std::string> wikilink_targets(std::string_view s) {
    std::vector<std::string> targets;
    const std::size_t n = s.size();
    std::size_t i = 0;
    while ((i = s.find("[[", i)) != std::string_view::npos) {
        // (?<!!)\[\[([^\[\]]+?)\]\]
        if (i > 0 && s[i - 1] == '!') {
            ++i;
            continue;
        }
        std::size_t j = i + 2;
        while (j < n && s[j] != '[' && s[j] != ']') ++j;
        if (j == i + 2 || j + 1 >= n || s[j] != ']' || s[j + 1] != ']') {
            ++i;
            continue;
        }
        std::string_view inner = strip(s.substr(i + 2, j - i - 2));
        i = j + 2;
        if (inner.empty()) continue;

        std::string_view target = strip(inner.substr(0, inner.find('|')));
        if (target.empty() || target.front() == '#') continue;

        target = target.substr(0, target.find('#'));
        target = strip(target.substr(0, target.find('^')));
        if (target.empty()) continue;

        if (iends_with_ascii(target, ".md")) target.remove_suffix(3);

        if (std::size_t slash = target.rfind('/'); slash != std::string_view::npos)
            target = strip(target.substr(slash + 1));

        std::string low(strip(lower(target)));
        if (std::any_of(std::begin(kImageExts), std::end(kImageExts),
                        [&](std::string_view ext) { return iends_with_ascii(low, ext); }))
            continue;

        std::string norm = normalize_note_name(target);
        if (!norm.empty()) targets.push_back(std::move(norm));
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    return targets;
}

std::vector<Heading> headings(std::string_view s) {
    std::vector<Heading> out;
    const char* p = s.data();
    const char* end = p + s.size();
    char fence = 0;  // '`' eller '~' när vi är inne i ett kodblock
    std::size_t fence_len = 0;

    while (p < end) {
        // En rad = fram till nästa splitlines-radbrytning
        const char* line = p;
        const char* line_end = p;
        while (line_end < end) {
            const char* q = line_end;
            if (is_line_break(decode(q, end))) break;
            line_end = q;
        }
        p = line_end;
        if (p < end) {
            const char* q = p;
            char32_t br = decode(q, end);
            p = q;
            if (br == '\r' && p < end && *p == '\n') ++p;
        }

        // ^\s{0,3}
        const char* c = line;
        for (int k = 0; k < 3 && c < line_end; ++k) {
            const char* q = c;
            if (!is_space(decode(q, line_end))) break;
            c = q;
        }

        // Kodstaket: ``` eller ~~~ (minst 3), stängs av samma tecken med minst samma längd
        if (c < line_end && (*c == '`' || *c == '~')) {
            char ch = *c;
            std::size_t run = 0;
            while (c + run < line_end && c[run] == ch) ++run;
            if (run >= 3) {
                if (!fence) {
                    fence = ch;
                    fence_len = run;
                } else if (ch == fence && run >= fence_len) {
                    fence = 0;
                }
                continue;
            }
        }
        if (fence) continue;

        // (#{1,6})\s+(.*)$
        std::size_t hashes = 0;
        while (c + hashes < line_end && c[hashes] == '#') ++hashes;
        if (hashes == 0 || hashes > 6) continue;
        const char* rest = c + hashes;
        if (rest >= line_end) continue;
        const char* q = rest;
        if (!is_space(decode(q, line_end))) continue;
        out.push_back({static_cast<int>(hashes), std::string(strip({rest, static_cast<std::size_t>(line_end - rest)}))});
    }
    return out;
}

}  // namespace text
