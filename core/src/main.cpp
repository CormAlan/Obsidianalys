// Obsidianalys 2 – snabb analys av en Obsidian-vault.
//
//   obsidianalys [VAULT]                 starta gränssnittet (öppnar webbläsaren)
//   obsidianalys --json VAULT            skriv hela analysen som JSON till stdout
//   obsidianalys --bench VAULT [N]       mät skanningstid (N körningar)
//
// Flaggor: --port N, --no-browser, --threads N, --web DIR, --dev

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "analysis.hpp"
#include "paths.hpp"
#include "platform.hpp"
#include "report.hpp"
#include "server.hpp"

namespace {

void usage() {
    std::puts(
        "Användning:\n"
        "  obsidianalys [VAULT] [--port N] [--no-browser] [--threads N] [--web DIR] [--dev]\n"
        "  obsidianalys --json VAULT [--threads N]\n"
        "  obsidianalys --bench VAULT [RUNS] [--threads N]");
}

int run_json(const std::string& root, unsigned threads) {
    auto vault = analysis::scan(paths::from_utf8(root), threads);
    std::string out = report::scan_json(vault, true);
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fputc('\n', stdout);
    return 0;
}

int run_bench(const std::string& root, int runs, unsigned threads) {
    std::vector<analysis::Timings> all;
    std::size_t files = 0;
    for (int r = 0; r < runs; ++r) {
        auto vault = analysis::scan(paths::from_utf8(root), threads);
        files = vault.notes.size();
        all.push_back(vault.timings);
        std::printf("körning %2d: %8.2f ms (walk %.2f, parse %.2f, graf %.2f, pagerank %.2f, mappar %.2f)\n", r + 1,
                    vault.timings.total_ms, vault.timings.walk_ms, vault.timings.parse_ms, vault.timings.graph_ms,
                    vault.timings.pagerank_ms, vault.timings.folders_ms);
    }
    std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.total_ms < b.total_ms; });
    std::printf("%zu filer – median %.2f ms, bästa %.2f ms\n", files, all[all.size() / 2].total_ms, all.front().total_ms);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args = platform::utf8_args(argc, argv);
    server::Options opts;
    std::string json_root, bench_root;
    int bench_runs = 10;

    try {
        for (std::size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            auto value = [&]() -> const std::string& {
                if (i + 1 >= args.size()) throw std::invalid_argument("saknar värde för " + a);
                return args[++i];
            };
            if (a == "-h" || a == "--help") return usage(), 0;
            else if (a == "--port") opts.port = std::stoi(value());
            else if (a == "--no-browser") opts.open_browser = false;
            else if (a == "--threads") opts.threads = static_cast<unsigned>(std::stoul(value()));
            else if (a == "--web") opts.web_dir = value();
            else if (a == "--dev") opts.dev = true, opts.open_browser = false;
            else if (a == "--json") json_root = value();
            else if (a == "--bench") {
                bench_root = value();
                if (i + 1 < args.size() && !args[i + 1].empty() && std::isdigit(static_cast<unsigned char>(args[i + 1][0])))
                    bench_runs = std::max(1, std::stoi(args[++i]));
            } else if (!a.empty() && a[0] == '-') throw std::invalid_argument("okänd flagga " + a);
            else opts.initial_root = a;
        }

        if (!json_root.empty()) return run_json(json_root, opts.threads);
        if (!bench_root.empty()) return run_bench(bench_root, bench_runs, opts.threads);
        return server::run(opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fel: %s\n", e.what());
        usage();
        return 1;
    }
}
