#include "analysis.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "paths.hpp"

namespace analysis {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    if (size <= 0) return {};
    std::string data(static_cast<std::size_t>(size), '\0');
    in.seekg(0);
    in.read(data.data(), size);
    data.resize(static_cast<std::size_t>(in.gcount()));
    return data;
}

bool is_markdown_name(std::string_view name) {
    if (name.size() < 3) return false;
    std::string_view ext = name.substr(name.size() - 3);
#ifdef _WIN32
    // Path.rglob("*.md") är skiftlägesokänslig på Windows
    return ext[0] == '.' && (ext[1] | 0x20) == 'm' && (ext[2] | 0x20) == 'd';
#else
    return ext == ".md";
#endif
}

// Pythons Path.stem för en .md-fil
std::string_view stem_of(std::string_view name) { return name == ".md" ? name : name.substr(0, name.size() - 3); }

template <class Fn>
void parallel_for(std::size_t n, unsigned threads, Fn&& fn) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(n, 1)));
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i);
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct NameIndex {
    std::unordered_map<std::string, std::vector<int>> exact;   // normaliserat namn
    std::unordered_map<std::string, std::vector<int>> folded;  // casefold-reserv
};

enum class Status { Ok, Unresolved, Ambiguous };

// Samma policy som _resolve_target_to_id(): exakt träff, annars unik casefold-träff.
std::pair<int, Status> resolve(const NameIndex& index, std::string_view target) {
    std::string t = text::normalize_note_name(target);
    if (auto it = index.exact.find(t); it != index.exact.end())
        return it->second.size() == 1 ? std::pair{it->second[0], Status::Ok} : std::pair{-1, Status::Ambiguous};
    if (auto it = index.folded.find(text::casefold(t)); it != index.folded.end())
        return it->second.size() == 1 ? std::pair{it->second[0], Status::Ok} : std::pair{-1, Status::Ambiguous};
    return {-1, Status::Unresolved};
}

// PageRank via power iteration, identisk med compute_pagerank() i Python-versionen.
void compute_pagerank(Vault& v, double alpha = 0.85, int max_iter = 100, double tol = 1e-8) {
    const std::size_t n = v.notes.size();
    if (n == 0) return;
    std::vector<double> p(n, 1.0 / n), next(n);
    for (int it = 0; it < max_iter; ++it) {
        double dangling = 0.0;
        for (std::size_t u = 0; u < n; ++u)
            if (v.notes[u].out_degree == 0) dangling += p[u];
        const double base = (1.0 - alpha) / n + (dangling != 0.0 ? alpha * dangling / n : 0.0);
        std::fill(next.begin(), next.end(), base);
        for (std::size_t u = 0; u < n; ++u) {
            const int d = v.notes[u].out_degree;
            if (d == 0) continue;
            const double share = alpha * p[u] / d;
            for (int k = v.out_offsets[u]; k < v.out_offsets[u + 1]; ++k) next[v.out_targets[k]] += share;
        }
        double diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) diff += std::abs(next[i] - p[i]);
        p.swap(next);
        if (diff < tol) break;
    }
    double sum = 0.0;
    for (double x : p) sum += x;
    for (std::size_t i = 0; i < n; ++i) v.notes[i].pagerank = sum > 0 ? p[i] / sum : p[i];
}

std::vector<std::string> split_parts(const std::string& rel) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= rel.size(); ++i) {
        if (i == rel.size() || rel[i] == '/') {
            parts.emplace_back(rel, start, i - start);
            start = i + 1;
        }
    }
    return parts;
}

}  // namespace

Vault scan(const fs::path& root_in, unsigned threads) {
    const auto t_start = Clock::now();
    std::error_code ec;

    Vault v;
    v.root = fs::absolute(root_in, ec).lexically_normal();
    if (ec) v.root = root_in.lexically_normal();
    if (!v.root.has_filename() && v.root.has_relative_path()) v.root = v.root.parent_path();  // "a/b/" -> "a/b"
    if (!fs::is_directory(v.root, ec)) throw std::runtime_error("Ogiltig rotmapp: " + paths::to_utf8(root_in));
    v.root_utf8 = paths::to_utf8(v.root);
    v.root_name = paths::to_utf8(v.root.filename());
    if (v.root_name.empty()) v.root_name = v.root_utf8;

    // --- 1. Hitta alla .md-filer (som Path.rglob("*.md"): även dolda mappar, ej symlänkade mappar)
    struct Found {
        fs::path path;
        std::string rel;
        std::vector<std::string> parts;
    };
    std::vector<Found> found;
    for (fs::recursive_directory_iterator it(v.root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        const fs::path& p = it->path();
        if (!is_markdown_name(paths::to_utf8(p.filename()))) continue;
        std::string rel = paths::to_generic_utf8(p.lexically_relative(v.root));
        auto parts = split_parts(rel);
        found.push_back({p, std::move(rel), std::move(parts)});
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.parts < b.parts; });

    std::map<std::string, int> folder_ids;
    v.notes.resize(found.size());
    for (std::size_t i = 0; i < found.size(); ++i) {
        Note& note = v.notes[i];
        note.path = std::move(found[i].path);
        note.name = found[i].parts.back();
        note.rel = std::move(found[i].rel);

        std::string folder_rel = note.rel.size() > note.name.size() ? note.rel.substr(0, note.rel.size() - note.name.size() - 1) : "";
        auto [it, inserted] = folder_ids.try_emplace(folder_rel, static_cast<int>(v.folders.size()));
        if (inserted) {
            Folder f;
            f.rel = folder_rel;
            std::size_t slash = folder_rel.rfind('/');
            f.name = folder_rel.empty() ? v.root_name : folder_rel.substr(slash == std::string::npos ? 0 : slash + 1);
            v.folders.push_back(std::move(f));
        }
        note.folder = it->second;
        v.folders[note.folder].notes.push_back(static_cast<int>(i));
    }
    v.timings.walk_ms = ms_since(t_start);

    // --- 2. Namnindex för länkupplösning
    NameIndex index;
    for (std::size_t i = 0; i < v.notes.size(); ++i) {
        std::string norm = text::normalize_note_name(stem_of(v.notes[i].name));
        index.folded[text::casefold(norm)].push_back(static_cast<int>(i));
        index.exact[std::move(norm)].push_back(static_cast<int>(i));
    }

    // --- 3. Läs och analysera alla filer parallellt
    const auto t_parse = Clock::now();
    const std::size_t n = v.notes.size();
    std::vector<std::vector<int>> out(n);
    std::vector<int> unresolved(n, 0), ambiguous(n, 0);
    parallel_for(n, threads, [&](std::size_t i) {
        Note& note = v.notes[i];
        const std::string raw = text::sanitize_utf8(read_file(note.path));
        const std::string no_eq = text::strip_equations(raw);
        const std::string no_eq_code = text::strip_code(no_eq);
        note.words[3] = text::count_words(raw);
        note.words[2] = text::count_words(text::strip_code(raw));
        note.words[1] = text::count_words(no_eq);
        note.words[0] = text::count_words(no_eq_code);

        // Länkar: kod och LaTeX ignoreras alltid
        for (const std::string& target : text::wikilink_targets(no_eq_code)) {
            auto [id, status] = resolve(index, target);
            if (status == Status::Ok) {
                if (id != static_cast<int>(i)) out[i].push_back(id);
            } else if (status == Status::Ambiguous) {
                ++ambiguous[i];
            } else {
                ++unresolved[i];
            }
        }
        std::sort(out[i].begin(), out[i].end());
        out[i].erase(std::unique(out[i].begin(), out[i].end()), out[i].end());

        note.headings = text::headings(raw);
        note.content_lower = text::lower(raw);
        note.rel_lower = text::lower(note.rel);
        for (const auto& h : note.headings) {
            note.headings_lower += text::lower(h.text);
            note.headings_lower.push_back('\n');
        }
    });
    v.timings.parse_ms = ms_since(t_parse);

    // --- 4. CSR-graf och grader
    const auto t_graph = Clock::now();
    v.out_offsets.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        v.out_offsets[i + 1] = v.out_offsets[i] + static_cast<int>(out[i].size());
        v.notes[i].out_degree = static_cast<int>(out[i].size());
        v.unresolved += unresolved[i];
        v.ambiguous += ambiguous[i];
    }
    v.out_targets.reserve(v.out_offsets[n]);
    for (auto& list : out) {
        for (int t : list) ++v.notes[t].in_degree;
        v.out_targets.insert(v.out_targets.end(), list.begin(), list.end());
    }
    v.edges = static_cast<int64_t>(v.out_targets.size());

    v.in_offsets.assign(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) v.in_offsets[i + 1] = v.in_offsets[i] + v.notes[i].in_degree;
    v.in_sources.resize(v.out_targets.size());
    std::vector<int> fill(v.in_offsets.begin(), v.in_offsets.end() - 1);
    for (std::size_t u = 0; u < n; ++u)
        for (int k = v.out_offsets[u]; k < v.out_offsets[u + 1]; ++k) v.in_sources[fill[v.out_targets[k]]++] = static_cast<int>(u);
    v.timings.graph_ms = ms_since(t_graph);

    // --- 5. PageRank (endast meningsfullt om det finns kanter)
    const auto t_pr = Clock::now();
    if (v.edges > 0) compute_pagerank(v);
    v.timings.pagerank_ms = ms_since(t_pr);

    // --- 6. Mappnivå
    const auto t_folders = Clock::now();
    std::unordered_map<uint64_t, int64_t> pair_counts;
    for (std::size_t u = 0; u < n; ++u) {
        const int fu = v.notes[u].folder;
        for (int k = v.out_offsets[u]; k < v.out_offsets[u + 1]; ++k) {
            const int fv = v.notes[v.out_targets[k]].folder;
            if (fu == fv) {
                ++v.folders[fu].internal;
            } else {
                ++v.folders[fu].outbound;
                ++v.folders[fv].inbound;
                ++pair_counts[(static_cast<uint64_t>(fu) << 32) | static_cast<uint32_t>(fv)];
            }
        }
    }
    for (Folder& f : v.folders) {
        const double m = static_cast<double>(f.notes.size());
        f.density = m < 2 ? 0.0 : static_cast<double>(f.internal) / (m * (m - 1));
    }
    v.pairs.reserve(pair_counts.size());
    for (auto [key, count] : pair_counts)
        v.pairs.push_back({static_cast<int>(key >> 32), static_cast<int>(key & 0xFFFFFFFFu), count});
    std::sort(v.pairs.begin(), v.pairs.end(), [](const FolderPair& a, const FolderPair& b) {
        if (a.edges != b.edges) return a.edges > b.edges;
        return a.from != b.from ? a.from < b.from : a.to < b.to;
    });
    v.timings.folders_ms = ms_since(t_folders);
    v.timings.total_ms = ms_since(t_start);
    return v;
}

std::vector<int> search(const Vault& vault, std::string_view query, SearchMode mode) {
    const std::string q = text::lower(text::strip(query));
    std::vector<int> hits;
    if (q.empty()) return hits;
    for (std::size_t i = 0; i < vault.notes.size(); ++i) {
        const Note& note = vault.notes[i];
        const std::string& hay = mode == SearchMode::Name      ? note.rel_lower
                                 : mode == SearchMode::Content ? note.content_lower
                                                               : note.headings_lower;
        if (hay.find(q) != std::string::npos) hits.push_back(static_cast<int>(i));
    }
    return hits;
}

}  // namespace analysis
