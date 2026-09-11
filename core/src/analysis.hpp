// Vault-analys: ordstatistik, länkgraf, PageRank och mappkoppling.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "text.hpp"

namespace analysis {

namespace fs = std::filesystem;

// Ordräkningsvarianter: index = (räkna ekvationer ? 2 : 0) + (räkna kod ? 1 : 0).
// Alla fyra räknas i samma skanning så att växlarna i UI:t är gratis.
constexpr int kWordVariants = 4;

struct Note {
    fs::path path;
    std::string rel;   // relativ sökväg, alltid med '/'
    std::string name;  // filnamn inkl. .md
    int folder = 0;
    int64_t words[kWordVariants] = {};
    std::vector<text::Heading> headings;
    int in_degree = 0;
    int out_degree = 0;
    double pagerank = 0.0;
    // Sökindex (gemener)
    std::string rel_lower;
    std::string content_lower;
    std::string headings_lower;  // rubriker separerade med '\n'
};

struct Folder {
    std::string rel;  // "" för rotmappen
    std::string name;
    std::vector<int> notes;  // anteckningar som ligger direkt i mappen
    int64_t internal = 0;
    int64_t outbound = 0;
    int64_t inbound = 0;
    double density = 0.0;  // internal / (n(n-1))
};

struct FolderPair {
    int from;
    int to;
    int64_t edges;
};

struct Timings {
    double walk_ms = 0, parse_ms = 0, graph_ms = 0, pagerank_ms = 0, folders_ms = 0, total_ms = 0;
};

struct Vault {
    fs::path root;
    std::string root_utf8;
    std::string root_name;
    std::vector<Note> notes;  // sorterade som Pythons sorted(Path)
    std::vector<Folder> folders;
    // Riktad enkel graf i CSR-form (inga multikanter, inga självloopar)
    std::vector<int> out_offsets, out_targets;
    std::vector<int> in_offsets, in_sources;
    int64_t edges = 0;
    int64_t unresolved = 0;
    int64_t ambiguous = 0;
    std::vector<FolderPair> pairs;  // mellan olika mappar, fallande på antal länkar
    Timings timings;
    uint64_t generation = 0;
};

// Skannar alla .md-filer under root. Kastar std::runtime_error vid ogiltig rotmapp.
Vault scan(const fs::path& root, unsigned threads = 0);

enum class SearchMode { Name, Content, Headings };
// Skiftlägesokänslig delsträngssökning (som Pythons q.lower() in text.lower()).
std::vector<int> search(const Vault& vault, std::string_view query, SearchMode mode);

}  // namespace analysis
