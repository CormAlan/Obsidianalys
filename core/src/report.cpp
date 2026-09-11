#include "report.hpp"

#include "json.hpp"
#include "paths.hpp"

namespace report {
namespace {

void headings(std::string& o, const analysis::Note& note) {
    o += '[';
    for (std::size_t k = 0; k < note.headings.size(); ++k) {
        if (k) o += ',';
        o += '[';
        json::num(o, int64_t{note.headings[k].level});
        o += ',';
        json::str(o, note.headings[k].text);
        o += ']';
    }
    o += ']';
}

void id_list(std::string& o, const std::vector<int>& data, int from, int to) {
    o += '[';
    for (int k = from; k < to; ++k) {
        if (k > from) o += ',';
        json::num(o, int64_t{data[k]});
    }
    o += ']';
}

}  // namespace

std::string scan_json(const analysis::Vault& v, bool full) {
    std::string o;
    o.reserve(v.notes.size() * (full ? 400 : 160) + v.pairs.size() * 16 + 4096);

    int64_t orphans = 0, dead_ends = 0;
    for (const auto& note : v.notes) {
        orphans += note.in_degree == 0;
        dead_ends += note.out_degree == 0;
    }

    o += "{\"generation\":";
    json::num(o, static_cast<int64_t>(v.generation));
    o += ",\"root\":";
    json::str(o, v.root_utf8);
    o += ",\"rootName\":";
    json::str(o, v.root_name);

    const auto& t = v.timings;
    o += ",\"timings\":{\"walk\":";
    json::num(o, t.walk_ms);
    o += ",\"parse\":";
    json::num(o, t.parse_ms);
    o += ",\"graph\":";
    json::num(o, t.graph_ms);
    o += ",\"pagerank\":";
    json::num(o, t.pagerank_ms);
    o += ",\"folders\":";
    json::num(o, t.folders_ms);
    o += ",\"total\":";
    json::num(o, t.total_ms);
    o += '}';

    o += ",\"stats\":{\"files\":";
    json::num(o, static_cast<int64_t>(v.notes.size()));
    o += ",\"folders\":";
    json::num(o, static_cast<int64_t>(v.folders.size()));
    o += ",\"edges\":";
    json::num(o, v.edges);
    o += ",\"unresolved\":";
    json::num(o, v.unresolved);
    o += ",\"ambiguous\":";
    json::num(o, v.ambiguous);
    o += ",\"orphans\":";
    json::num(o, orphans);
    o += ",\"deadEnds\":";
    json::num(o, dead_ends);
    o += '}';

    o += ",\"notes\":[";
    for (std::size_t i = 0; i < v.notes.size(); ++i) {
        const auto& note = v.notes[i];
        if (i) o += ',';
        o += "{\"rel\":";
        json::str(o, note.rel);
        o += ",\"name\":";
        json::str(o, note.name);
        o += ",\"f\":";
        json::num(o, int64_t{note.folder});
        o += ",\"w\":[";
        for (int k = 0; k < analysis::kWordVariants; ++k) {
            if (k) o += ',';
            json::num(o, note.words[k]);
        }
        o += "],\"in\":";
        json::num(o, int64_t{note.in_degree});
        o += ",\"out\":";
        json::num(o, int64_t{note.out_degree});
        o += ",\"pr\":";
        json::num(o, note.pagerank);
        o += ",\"h\":";
        json::num(o, static_cast<int64_t>(note.headings.size()));
        if (full) {
            o += ",\"headings\":";
            headings(o, note);
            o += ",\"links\":";
            id_list(o, v.out_targets, v.out_offsets[i], v.out_offsets[i + 1]);
        }
        o += '}';
    }
    o += ']';

    o += ",\"folders\":[";
    for (std::size_t i = 0; i < v.folders.size(); ++i) {
        const auto& f = v.folders[i];
        if (i) o += ',';
        o += "{\"rel\":";
        json::str(o, f.rel);
        o += ",\"name\":";
        json::str(o, f.name);
        o += ",\"n\":";
        json::num(o, static_cast<int64_t>(f.notes.size()));
        o += ",\"internal\":";
        json::num(o, f.internal);
        o += ",\"density\":";
        json::num(o, f.density);
        o += ",\"out\":";
        json::num(o, f.outbound);
        o += ",\"in\":";
        json::num(o, f.inbound);
        o += '}';
    }
    o += ']';

    o += ",\"pairs\":[";
    for (std::size_t i = 0; i < v.pairs.size(); ++i) {
        if (i) o += ',';
        o += '[';
        json::num(o, int64_t{v.pairs[i].from});
        o += ',';
        json::num(o, int64_t{v.pairs[i].to});
        o += ',';
        json::num(o, v.pairs[i].edges);
        o += ']';
    }
    o += "]}";
    return o;
}

std::string note_json(const analysis::Vault& v, int id) {
    const auto& note = v.notes[id];
    std::string o = "{\"id\":";
    json::num(o, int64_t{id});
    o += ",\"abs\":";
    json::str(o, paths::to_utf8(note.path));
    o += ",\"headings\":";
    headings(o, note);
    o += ",\"out\":";
    id_list(o, v.out_targets, v.out_offsets[id], v.out_offsets[id + 1]);
    o += ",\"in\":";
    id_list(o, v.in_sources, v.in_offsets[id], v.in_offsets[id + 1]);
    o += '}';
    return o;
}

}  // namespace report
