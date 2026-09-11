#include "server.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#include "analysis.hpp"
#include "httplib.h"
#include "json.hpp"
#include "paths.hpp"
#include "platform.hpp"
#include "report.hpp"
#include "web_assets.hpp"

namespace server {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr const char* kVersion = "2.0.0";

struct State {
    std::mutex mu;  // skyddar vault, config och generation
    std::mutex scan_mu;  // en skanning i taget
    std::shared_ptr<const analysis::Vault> vault;
    platform::Config config;
    uint64_t generation = 0;

    std::shared_ptr<const analysis::Vault> current() {
        std::lock_guard lock(mu);
        return vault;
    }
};

void send_json(httplib::Response& res, std::string body, int status = 200) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(std::move(body), "application/json; charset=utf-8");
}

void send_error(httplib::Response& res, int status, std::string_view message) {
    std::string body = "{\"error\":";
    json::str(body, message);
    body += '}';
    send_json(res, std::move(body), status);
}

bool parse_int(const std::string& s, long long& out) {
    auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

// Slår upp ?id= mot aktuell skanning; ?gen= skyddar mot id:n från en äldre skanning.
std::shared_ptr<const analysis::Vault> note_from_request(State& state, const httplib::Request& req,
                                                         httplib::Response& res, int& id) {
    auto vault = state.current();
    long long v = -1, gen = -1;
    if (!vault) {
        send_error(res, 409, "Ingen skanning gjord");
        return nullptr;
    }
    if (req.has_param("gen") && (!parse_int(req.get_param_value("gen"), gen) || gen != static_cast<long long>(vault->generation))) {
        send_error(res, 409, "Skanningen har ändrats – ladda om");
        return nullptr;
    }
    if (!parse_int(req.get_param_value("id"), v) || v < 0 || v >= static_cast<long long>(vault->notes.size())) {
        send_error(res, 400, "Ogiltigt id");
        return nullptr;
    }
    id = static_cast<int>(v);
    return vault;
}

std::string config_json(const platform::Config& c) {
    std::string o = "{\"root\":";
    json::str(o, c.root);
    o += ",\"topN\":";
    json::num(o, int64_t{c.top_n});
    o += ",\"includeEquations\":";
    json::boolean(o, c.include_equations);
    o += ",\"includeCode\":";
    json::boolean(o, c.include_code);
    o += ",\"theme\":";
    json::str(o, c.theme);
    o += '}';
    return o;
}

std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c + 32);
    return s;
}

std::string list_dir_json(const fs::path& dir) {
    struct Entry {
        std::string name, path, key;
        bool vault;
    };
    std::vector<Entry> entries;
    std::error_code ec;
    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
        std::error_code dec;
        if (!it->is_directory(dec)) continue;
        std::string name = paths::to_utf8(it->path().filename());
        if (name.empty() || name[0] == '.') continue;
        entries.push_back({name, paths::to_utf8(it->path()), lower_ascii(name), fs::exists(it->path() / ".obsidian", dec)});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });

    std::string o = "{\"path\":";
    json::str(o, paths::to_utf8(dir));
    o += ",\"parent\":";
    fs::path parent = dir.parent_path();
    if (parent.empty() || parent == dir) o += "null";
    else json::str(o, paths::to_utf8(parent));
    o += ",\"isVault\":";
    json::boolean(o, fs::exists(dir / ".obsidian", ec));
    o += ",\"dirs\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i) o += ',';
        o += "{\"name\":";
        json::str(o, entries[i].name);
        o += ",\"path\":";
        json::str(o, entries[i].path);
        o += ",\"vault\":";
        json::boolean(o, entries[i].vault);
        o += '}';
    }
    o += "]}";
    return o;
}

const char* mime_for(std::string_view path) {
    auto ends = [&](std::string_view ext) { return path.size() >= ext.size() && path.substr(path.size() - ext.size()) == ext; };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".woff2")) return "font/woff2";
    if (ends(".json")) return "application/json";
    return "application/octet-stream";
}

// Hämtar en frontendfil – från --web-katalogen eller ur binären.
bool load_asset(const std::string& web_dir, const std::string& path, std::string& body, std::string& mime) {
    if (path.find("..") != std::string::npos) return false;
    if (!web_dir.empty()) {
        std::ifstream in(paths::from_utf8(web_dir) / paths::from_utf8(path.substr(1)), std::ios::binary);
        if (!in) return false;
        body.assign(std::istreambuf_iterator<char>(in), {});
        mime = mime_for(path);
        return true;
    }
    for (std::size_t i = 0; i < web::kAssetCount; ++i) {
        if (path == web::kAssets[i].path) {
            body.assign(reinterpret_cast<const char*>(web::kAssets[i].data), web::kAssets[i].size);
            mime = web::kAssets[i].mime;
            return true;
        }
    }
    return false;
}

constexpr const char* kMissingFrontend = R"(<!doctype html><meta charset="utf-8"><title>Obsidianalys</title>
<body style="font-family:system-ui;padding:3rem;max-width:40rem">
<h1>Frontend saknas</h1>
<p>Binären byggdes utan gränssnittet. Kör <code>npm install &amp;&amp; npm run build</code> i <code>web/</code>
och bygg om med CMake – eller starta med <code>--web web/dist</code>.</p></body>)";

}  // namespace

int run(const Options& opts) {
    State state;
    state.config = platform::load_config();
    if (!opts.initial_root.empty()) {
        std::error_code ec;
        state.config.root = paths::to_utf8(fs::absolute(paths::from_utf8(opts.initial_root), ec));
        platform::save_config(state.config);
    }

    const std::string token = opts.dev ? "dev" : platform::random_token();
    httplib::Server svr;

    int port = -1;
    for (int p = opts.port; p < opts.port + 20; ++p) {
        if (svr.bind_to_port("127.0.0.1", p)) {
            port = p;
            break;
        }
    }
    if (port < 0) {
        std::fprintf(stderr, "Kunde inte binda någon port i %d–%d\n", opts.port, opts.port + 19);
        return 1;
    }
    const std::string port_str = std::to_string(port);

    svr.set_default_headers({{"X-Content-Type-Options", "nosniff"}, {"Referrer-Policy", "no-referrer"}});

    // Skydd för en lokal server: Host-kontroll stoppar DNS-rebinding och den hemliga
    // token-headern stoppar andra webbplatser från att anropa API:t (kräver CORS-preflight
    // som vi aldrig godkänner).
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        const std::string host = req.get_header_value("Host");
        if (host != "127.0.0.1:" + port_str && host != "localhost:" + port_str) {
            res.status = 403;
            res.set_content("Forbidden", "text/plain");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.path.rfind("/api/", 0) == 0 && req.get_header_value("X-Obsidianalys-Token") != token) {
            send_error(res, 403, "Ogiltig token");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/api/state", [&](const httplib::Request&, httplib::Response& res) {
        std::string o = "{\"version\":";
        json::str(o, kVersion);
        {
            std::lock_guard lock(state.mu);
            o += ",\"config\":" + config_json(state.config);
        }
        o += ",\"home\":";
        json::str(o, paths::to_utf8(platform::home_dir()));
        o += ",\"roots\":[";
        auto roots = platform::filesystem_roots();
        for (std::size_t i = 0; i < roots.size(); ++i) {
            if (i) o += ',';
            json::str(o, roots[i]);
        }
        o += "],\"configPath\":";
        json::str(o, paths::to_utf8(platform::config_path()));
        o += '}';
        send_json(res, std::move(o));
    });

    svr.Post("/api/scan", [&](const httplib::Request& req, httplib::Response& res) {
        std::string root = req.get_param_value("root");
        if (root.empty()) return send_error(res, 400, "Ange en rotmapp");
        std::lock_guard scan_lock(state.scan_mu);
        try {
            auto vault = std::make_shared<analysis::Vault>(analysis::scan(paths::from_utf8(root), opts.threads));
            {
                std::lock_guard lock(state.mu);
                vault->generation = ++state.generation;
                state.vault = vault;
                state.config.root = vault->root_utf8;
                platform::save_config(state.config);
            }
            send_json(res, report::scan_json(*vault, false));
        } catch (const std::exception& e) {
            send_error(res, 400, e.what());
        }
    });

    svr.Get("/api/note", [&](const httplib::Request& req, httplib::Response& res) {
        int id;
        if (auto vault = note_from_request(state, req, res, id)) send_json(res, report::note_json(*vault, id));
    });

    svr.Get("/api/search", [&](const httplib::Request& req, httplib::Response& res) {
        auto vault = state.current();
        if (!vault) return send_error(res, 409, "Ingen skanning gjord");
        const std::string mode_str = req.get_param_value("mode");
        auto mode = mode_str == "content"    ? analysis::SearchMode::Content
                    : mode_str == "headings" ? analysis::SearchMode::Headings
                                             : analysis::SearchMode::Name;
        const auto t0 = Clock::now();
        auto hits = analysis::search(*vault, req.get_param_value("q"), mode);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::string o = "{\"generation\":";
        json::num(o, static_cast<int64_t>(vault->generation));
        o += ",\"ms\":";
        json::num(o, ms);
        o += ",\"ids\":[";
        for (std::size_t i = 0; i < hits.size(); ++i) {
            if (i) o += ',';
            json::num(o, int64_t{hits[i]});
        }
        o += "]}";
        send_json(res, std::move(o));
    });

    svr.Post("/api/open", [&](const httplib::Request& req, httplib::Response& res) {
        int id;
        auto vault = note_from_request(state, req, res, id);
        if (!vault) return;
        // Endast filer från skanningen kan öppnas – aldrig godtyckliga sökvägar
        if (!platform::open_with_system(paths::to_utf8(vault->notes[id].path)))
            return send_error(res, 500, "Kunde inte öppna filen");
        send_json(res, "{\"ok\":true}");
    });

    svr.Get("/api/ls", [&](const httplib::Request& req, httplib::Response& res) {
        std::string p = req.get_param_value("path");
        fs::path dir = p.empty() ? platform::home_dir() : paths::from_utf8(p);
        std::error_code ec;
        dir = fs::absolute(dir, ec).lexically_normal();
        if (!dir.has_filename() && dir.has_relative_path()) dir = dir.parent_path();
        if (!fs::is_directory(dir, ec)) return send_error(res, 400, "Mappen finns inte");
        send_json(res, list_dir_json(dir));
    });

    svr.Post("/api/config", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard lock(state.mu);
        long long v;
        if (req.has_param("topN") && parse_int(req.get_param_value("topN"), v) && v >= 1)
            state.config.top_n = static_cast<int>(std::min<long long>(v, 1000000));
        if (req.has_param("eq")) state.config.include_equations = req.get_param_value("eq") == "1";
        if (req.has_param("code")) state.config.include_code = req.get_param_value("code") == "1";
        if (req.has_param("theme")) {
            std::string t = req.get_param_value("theme");
            if (t == "light" || t == "dark" || t == "system") state.config.theme = t;
        }
        platform::save_config(state.config);
        send_json(res, config_json(state.config));
    });

    svr.Post("/api/quit", [&](const httplib::Request&, httplib::Response& res) {
        send_json(res, "{\"ok\":true}");
        std::thread([&svr] {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            svr.stop();
        }).detach();
    });

    svr.Get(R"(/.*)", [&](const httplib::Request& req, httplib::Response& res) {
        std::string path = req.path == "/" ? "/index.html" : req.path;
        std::string body, mime;
        bool found = load_asset(opts.web_dir, path, body, mime);
        if (!found && path.rfind('.') < path.rfind('/')) {  // SPA-route utan filändelse
            path = "/index.html";
            found = load_asset(opts.web_dir, path, body, mime);
        }
        if (!found) {
            if (path == "/index.html") return res.set_content(kMissingFrontend, "text/html; charset=utf-8");
            res.status = 404;
            return res.set_content("Not found", "text/plain");
        }
        if (path == "/index.html") {
            const std::string meta = "<meta name=\"obsidianalys-token\" content=\"" + token + "\">";
            if (std::size_t head = body.find("</head>"); head != std::string::npos) body.insert(head, meta);
            res.set_header("Cache-Control", "no-store");
            res.set_header("Content-Security-Policy",
                           "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                           "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'");
        } else if (path.rfind("/assets/", 0) == 0) {
            res.set_header("Cache-Control", "public, max-age=31536000, immutable");
        }
        res.set_content(std::move(body), mime);
    });

    const std::string url = "http://127.0.0.1:" + port_str + "/";
    std::printf("Obsidianalys %s körs på %s\n", kVersion, url.c_str());
    std::printf("Konfiguration: %s\n", paths::to_utf8(platform::config_path()).c_str());
    std::printf("Avsluta med Ctrl+C eller knappen i gränssnittet.\n");
    std::fflush(stdout);
    if (opts.open_browser) platform::open_with_system(url);

    svr.listen_after_bind();
    return 0;
}

}  // namespace server
