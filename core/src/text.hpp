// Textbehandling för markdown – speglar exakt regex-semantiken i Python-versionen
// (Obsidianalys.py) men utan regex-motor: allt är linjära skanningar.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace text {

// Tar bort ogiltiga UTF-8-byte (som Pythons read_text(errors="ignore")).
// Returnerar indata oförändrat när den redan är giltig.
std::string sanitize_utf8(std::string raw);

// Avkodar en kodpunkt och flyttar fram p. Kräver giltig UTF-8.
char32_t decode(const char*& p, const char* end);
void append_utf8(std::string& out, char32_t cp);

bool is_word(char32_t cp);   // \w
bool is_space(char32_t cp);  // \s / str.isspace()

// re.sub(r"\$\$(.*?)\$\$", "") följt av re.sub(r"\$(.*?)\$", "")
std::string strip_equations(std::string_view s);
// re.sub(r"```.*?```", "") följt av re.sub(r"`[^`]*`", "")
std::string strip_code(std::string_view s);
// len(re.findall(r"\b\w+\b", s))
int64_t count_words(std::string_view s);

std::string lower(std::string_view s);     // str.lower()
std::string casefold(std::string_view s);  // str.casefold()
std::string_view strip(std::string_view s);  // str.strip()
// strip + kollapsa \s+ till ett mellanslag (_normalize_note_name)
std::string normalize_note_name(std::string_view s);

// Unika wikilänk-mål ([[...]], ej ![[...]]) i redan strippad text.
// Samma regler som extract_unique_wikilink_targets().
std::vector<std::string> wikilink_targets(std::string_view stripped);

struct Heading {
    int level;
    std::string text;
};
// Rubriker ^\s{0,3}(#{1,6})\s+(.*)$ per rad – rader i ```/~~~-kodblock hoppas över.
std::vector<Heading> headings(std::string_view s);

}  // namespace text
