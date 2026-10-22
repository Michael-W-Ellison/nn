// File: src/storage/pattern_database.cpp
#include "storage/pattern_database.hpp"
#include "storage/memory_backend.hpp"
#ifdef DPAN_HAS_SQLITE
#include "storage/persistent_backend.hpp"
#endif
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace dpan {

namespace {

using FlatConfig = std::map<std::string, std::string>;

std::string Trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        start++;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(start, end - start);
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string Unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

/// Minimal JSON reader: nested objects of scalar values, flattened to
/// "section.key" -> value. Arrays are not supported (not used by the config).
class JsonFlattener {
public:
    explicit JsonFlattener(const std::string& text) : text_(text) {}

    FlatConfig Parse() {
        FlatConfig out;
        SkipSpace();
        ParseObject("", out);
        SkipSpace();
        if (pos_ != text_.size()) {
            Fail("unexpected trailing content");
        }
        return out;
    }

private:
    const std::string& text_;
    size_t pos_{0};

    [[noreturn]] void Fail(const std::string& what) const {
        throw std::runtime_error("Invalid JSON pattern database config at offset " +
                                 std::to_string(pos_) + ": " + what);
    }

    void SkipSpace() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (std::isspace(static_cast<unsigned char>(c))) {
                pos_++;
            } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
                // Allow // line comments (the documented example uses them)
                while (pos_ < text_.size() && text_[pos_] != '\n') {
                    pos_++;
                }
            } else {
                break;
            }
        }
    }

    void Expect(char c) {
        SkipSpace();
        if (pos_ >= text_.size() || text_[pos_] != c) {
            Fail(std::string("expected '") + c + "'");
        }
        pos_++;
    }

    std::string ParseString() {
        Expect('"');
        std::string result;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            char c = text_[pos_++];
            if (c == '\\' && pos_ < text_.size()) {
                char e = text_[pos_++];
                switch (e) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    default: result += e; break;
                }
            } else {
                result += c;
            }
        }
        if (pos_ >= text_.size()) {
            Fail("unterminated string");
        }
        pos_++;  // closing quote
        return result;
    }

    std::string ParseScalar() {
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == '"') {
            return ParseString();
        }
        size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] != ',' && text_[pos_] != '}' &&
               !std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            pos_++;
        }
        if (start == pos_) {
            Fail("expected a value");
        }
        return text_.substr(start, pos_ - start);
    }

    void ParseObject(const std::string& prefix, FlatConfig& out) {
        Expect('{');
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            pos_++;
            return;
        }
        while (true) {
            SkipSpace();
            std::string key = ToLower(ParseString());
            Expect(':');
            SkipSpace();
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            if (pos_ < text_.size() && text_[pos_] == '{') {
                ParseObject(full_key, out);
            } else if (pos_ < text_.size() && text_[pos_] == '[') {
                Fail("arrays are not supported");
            } else {
                out[full_key] = ParseScalar();
            }
            SkipSpace();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                pos_++;
                continue;
            }
            Expect('}');
            return;
        }
    }
};

/// Minimal YAML / key-value reader: "key: value" or "key = value" lines,
/// '#' comments, and one level of indented sections ("memory:" followed by
/// indented keys), flattened to "section.key" -> value.
FlatConfig ParseKeyValue(const std::string& text) {
    FlatConfig out;
    std::istringstream in(text);
    std::string line;
    std::string section;
    size_t line_no = 0;

    while (std::getline(in, line)) {
        line_no++;
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        if (Trim(line).empty()) {
            continue;
        }

        bool indented = std::isspace(static_cast<unsigned char>(line[0]));
        std::string content = Trim(line);

        auto sep = content.find(':');
        if (sep == std::string::npos) {
            sep = content.find('=');
        }
        if (sep == std::string::npos) {
            throw std::runtime_error("Invalid pattern database config line " +
                                     std::to_string(line_no) + ": expected 'key: value'");
        }

        std::string key = ToLower(Unquote(Trim(content.substr(0, sep))));
        std::string value = Unquote(Trim(content.substr(sep + 1)));

        if (!indented) {
            section.clear();
        }
        if (value.empty()) {
            // Section header
            section = key;
            continue;
        }
        out[section.empty() || !indented ? key : section + "." + key] = value;
    }
    return out;
}

FlatConfig ParseConfigFile(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("Cannot open pattern database config: " + config_path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    std::string trimmed = Trim(text);
    if (!trimmed.empty() && trimmed.front() == '{') {
        return JsonFlattener(text).Parse();
    }
    return ParseKeyValue(text);
}

/// Look up the first of several keys (e.g. "memory.cache_size", "cache_size")
const std::string* Find(const FlatConfig& config, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        auto it = config.find(key);
        if (it != config.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

size_t GetSize(const FlatConfig& config, std::initializer_list<const char*> keys, size_t fallback) {
    const std::string* value = Find(config, keys);
    if (!value) {
        return fallback;
    }
    try {
        size_t used = 0;
        long long parsed = std::stoll(*value, &used);
        if (used != value->size() || parsed < 0) {
            throw std::invalid_argument("not a non-negative integer");
        }
        return static_cast<size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer in pattern database config: '" + *value + "'");
    }
}

bool GetBool(const FlatConfig& config, std::initializer_list<const char*> keys, bool fallback) {
    const std::string* value = Find(config, keys);
    if (!value) {
        return fallback;
    }
    std::string v = ToLower(*value);
    if (v == "true" || v == "yes" || v == "on" || v == "1") {
        return true;
    }
    if (v == "false" || v == "no" || v == "off" || v == "0") {
        return false;
    }
    throw std::runtime_error("Invalid boolean in pattern database config: '" + *value + "'");
}

std::string GetString(const FlatConfig& config, std::initializer_list<const char*> keys,
                      const std::string& fallback) {
    const std::string* value = Find(config, keys);
    return value ? *value : fallback;
}

}  // namespace

std::unique_ptr<PatternDatabase> CreatePatternDatabase(const std::string& config_path) {
    FlatConfig config = ParseConfigFile(config_path);

    std::string backend = ToLower(GetString(config, {"backend", "storage.backend"}, "memory"));

    if (backend == "memory" || backend == "mmap") {
        MemoryBackend::Config mem;
        mem.initial_capacity = GetSize(config, {"memory.initial_capacity", "mmap.initial_capacity",
                                                "initial_capacity"}, mem.initial_capacity);
        mem.enable_cache = GetBool(config, {"memory.enable_cache", "mmap.enable_cache",
                                            "enable_cache"}, mem.enable_cache);
        mem.cache_size = GetSize(config, {"memory.cache_size", "mmap.cache_size", "cache_size"},
                                 mem.cache_size);
        mem.mmap_path = GetString(config, {"mmap.path", "memory.mmap_path", "mmap_path", "path"}, "");
        mem.use_mmap = backend == "mmap" ||
                       GetBool(config, {"memory.use_mmap", "use_mmap"}, !mem.mmap_path.empty());
        if (mem.use_mmap && mem.mmap_path.empty()) {
            throw std::runtime_error("Pattern database config: backend '" + backend +
                                     "' requires a file path (mmap.path)");
        }
        return std::make_unique<MemoryBackend>(mem);
    }

    if (backend == "persistent" || backend == "sqlite") {
#ifndef DPAN_HAS_SQLITE
        throw std::runtime_error("Pattern database config: backend '" + backend +
                                 "' is unavailable (DPAN was built without SQLite)");
#else
        PersistentBackend::Config db;
        db.db_path = GetString(config, {"persistent.db_path", "sqlite.db_path", "persistent.path",
                                        "sqlite.path", "db_path", "path"}, "");
        if (db.db_path.empty()) {
            throw std::runtime_error("Pattern database config: backend '" + backend +
                                     "' requires db_path");
        }
        db.enable_wal = GetBool(config, {"persistent.enable_wal", "sqlite.enable_wal", "enable_wal"},
                                db.enable_wal);
        db.cache_size_kb = GetSize(config, {"persistent.cache_size_kb", "sqlite.cache_size_kb",
                                            "cache_size_kb"}, db.cache_size_kb);
        db.page_size = GetSize(config, {"persistent.page_size", "sqlite.page_size", "page_size"},
                               db.page_size);
        db.enable_auto_vacuum = GetBool(config, {"persistent.enable_auto_vacuum",
                                                 "sqlite.enable_auto_vacuum", "enable_auto_vacuum"},
                                        db.enable_auto_vacuum);
        db.synchronous = GetString(config, {"persistent.synchronous", "sqlite.synchronous",
                                            "synchronous"}, db.synchronous);
        std::string sync = ToLower(db.synchronous);
        std::transform(sync.begin(), sync.end(), db.synchronous.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (sync != "full" && sync != "normal" && sync != "off") {
            throw std::runtime_error("Pattern database config: synchronous must be FULL, NORMAL or OFF");
        }
        return std::make_unique<PersistentBackend>(db);
#endif
    }

    throw std::runtime_error("Pattern database config: unsupported backend '" + backend +
                             "' (supported: memory, mmap, persistent/sqlite)");
}

} // namespace dpan
