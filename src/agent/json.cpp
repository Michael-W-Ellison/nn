// File: src/agent/json.cpp
#include "agent/json.hpp"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace agent {

const char* ToString(Json::Type type) {
    switch (type) {
        case Json::Type::NUL: return "null";
        case Json::Type::BOOLEAN: return "boolean";
        case Json::Type::NUMBER: return "number";
        case Json::Type::STRING: return "string";
        case Json::Type::ARRAY: return "array";
        case Json::Type::OBJECT: return "object";
    }
    return "unknown";
}

namespace {

[[noreturn]] void WrongType(Json::Type expected, Json::Type actual) {
    throw std::invalid_argument(std::string("expected a JSON ") + ToString(expected) + ", got " + ToString(actual));
}

constexpr size_t kMaxDepth = 256;

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json ParseDocument() {
        SkipSpace();
        Json value = ParseValue(0);
        SkipSpace();
        if (pos_ != text_.size()) Fail("unexpected text after the value");
        return value;
    }

private:
    [[noreturn]] void Fail(const std::string& message) const {
        // Line and column of the error, 1-based
        size_t line = 1, column = 1;
        for (size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw std::invalid_argument("JSON: " + message + " at line " + std::to_string(line) + ", column " +
                                    std::to_string(column));
    }

    void SkipSpace() {
        while (pos_ < text_.size() &&
               (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool Consume(const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (text_.compare(pos_, n, word) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    Json ParseValue(size_t depth) {
        if (depth > kMaxDepth) Fail("nesting too deep");
        if (pos_ >= text_.size()) Fail("unexpected end of text");
        char c = text_[pos_];
        if (c == '{') return ParseObject(depth);
        if (c == '[') return ParseArray(depth);
        if (c == '"') return Json(ParseString());
        if (c == '-' || (c >= '0' && c <= '9')) return Json(ParseNumber());
        if (Consume("true")) return Json(true);
        if (Consume("false")) return Json(false);
        if (Consume("null")) return Json();
        Fail(std::string("unexpected character '") + c + "'");
    }

    Json ParseObject(size_t depth) {
        ++pos_;  // {
        Json result = Json::MakeObject();
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return result;
        }
        while (true) {
            SkipSpace();
            if (pos_ >= text_.size() || text_[pos_] != '"') Fail("expected a member name");
            std::string key = ParseString();
            SkipSpace();
            if (pos_ >= text_.size() || text_[pos_] != ':') Fail("expected ':'");
            ++pos_;
            SkipSpace();
            if (result.Find(key)) Fail("duplicate member \"" + key + "\"");
            result.Set(key, ParseValue(depth + 1));
            SkipSpace();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == '}') {
                ++pos_;
                return result;
            }
            Fail("expected ',' or '}'");
        }
    }

    Json ParseArray(size_t depth) {
        ++pos_;  // [
        Json result = Json::MakeArray();
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return result;
        }
        while (true) {
            SkipSpace();
            result.Push(ParseValue(depth + 1));
            SkipSpace();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == ']') {
                ++pos_;
                return result;
            }
            Fail("expected ',' or ']'");
        }
    }

    unsigned ParseHex4() {
        if (pos_ + 4 > text_.size()) Fail("incomplete \\u escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else Fail("invalid \\u escape");
        }
        return value;
    }

    static void AppendUtf8(std::string& out, unsigned code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    std::string ParseString() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) Fail("unterminated string");
            char c = text_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) Fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) Fail("unterminated string");
            char e = text_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = ParseHex4();
                    if (code >= 0xD800 && code < 0xDC00) {
                        if (!Consume("\\u")) Fail("unpaired surrogate");
                        unsigned low = ParseHex4();
                        if (low < 0xDC00 || low >= 0xE000) Fail("unpaired surrogate");
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    } else if (code >= 0xDC00 && code < 0xE000) {
                        Fail("unpaired surrogate");
                    }
                    AppendUtf8(out, code);
                    break;
                }
                default: Fail(std::string("invalid escape '\\") + e + "'");
            }
        }
    }

    double ParseNumber() {
        size_t start = pos_;
        if (text_[pos_] == '-') ++pos_;
        auto digits = [&] {
            size_t begin = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            return pos_ - begin;
        };
        if (pos_ < text_.size() && text_[pos_] == '0') {
            ++pos_;
        } else if (digits() == 0) {
            Fail("invalid number");
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (digits() == 0) Fail("invalid number");
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (digits() == 0) Fail("invalid number");
        }
        // Classic locale: the host may have set one with another decimal point
        std::istringstream stream(text_.substr(start, pos_ - start));
        stream.imbue(std::locale::classic());
        double value = 0.0;
        stream >> value;
        if (stream.fail()) {
            // Out of the double range: the stream sets failbit; JSON allows
            // any magnitude, so saturate as strtod would
            value = text_[start] == '-' ? -HUGE_VAL : HUGE_VAL;
            bool tiny = false;
            for (size_t i = start; i < pos_; ++i) {
                if (text_[i] == 'e' || text_[i] == 'E') tiny = text_[i + 1] == '-';
            }
            if (tiny) value = 0.0;
        }
        return value;
    }

    const std::string& text_;
    size_t pos_{0};
};

void DumpString(const std::string& s, std::string& out) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
                    out += buffer;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

}  // namespace

Json Json::Parse(const std::string& text) { return Parser(text).ParseDocument(); }

bool Json::AsBool() const {
    if (type_ != Type::BOOLEAN) WrongType(Type::BOOLEAN, type_);
    return boolean_;
}
double Json::AsNumber() const {
    if (type_ != Type::NUMBER) WrongType(Type::NUMBER, type_);
    return number_;
}
const std::string& Json::AsString() const {
    if (type_ != Type::STRING) WrongType(Type::STRING, type_);
    return string_;
}
const Json::Array& Json::AsArray() const {
    if (type_ != Type::ARRAY) WrongType(Type::ARRAY, type_);
    return array_;
}
Json::Array& Json::AsArray() {
    if (type_ != Type::ARRAY) WrongType(Type::ARRAY, type_);
    return array_;
}
const Json::Object& Json::AsObject() const {
    if (type_ != Type::OBJECT) WrongType(Type::OBJECT, type_);
    return object_;
}
Json::Object& Json::AsObject() {
    if (type_ != Type::OBJECT) WrongType(Type::OBJECT, type_);
    return object_;
}

const Json* Json::Find(const std::string& key) const {
    for (const auto& member : AsObject()) {
        if (member.first == key) return &member.second;
    }
    return nullptr;
}

Json& Json::Set(const std::string& key, Json value) {
    AsObject().emplace_back(key, std::move(value));
    return object_.back().second;
}

Json& Json::Push(Json value) {
    AsArray().push_back(std::move(value));
    return array_.back();
}

std::string Json::Dump() const {
    std::string out;
    DumpTo(out);
    return out;
}

void Json::DumpTo(std::string& out) const {
    switch (type_) {
        case Type::NUL: out += "null"; break;
        case Type::BOOLEAN: out += boolean_ ? "true" : "false"; break;
        case Type::NUMBER: {
            if (!std::isfinite(number_)) {
                out += "null";
                break;
            }
            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            // Integers print without an exponent or fraction
            if (number_ == std::floor(number_) && std::fabs(number_) < 1e15) {
                stream << std::fixed << std::setprecision(0) << number_;
            } else {
                stream << std::setprecision(17) << number_;
            }
            out += stream.str();
            break;
        }
        case Type::STRING: DumpString(string_, out); break;
        case Type::ARRAY: {
            out += '[';
            for (size_t i = 0; i < array_.size(); ++i) {
                if (i) out += ',';
                array_[i].DumpTo(out);
            }
            out += ']';
            break;
        }
        case Type::OBJECT: {
            out += '{';
            for (size_t i = 0; i < object_.size(); ++i) {
                if (i) out += ',';
                DumpString(object_[i].first, out);
                out += ':';
                object_[i].second.DumpTo(out);
            }
            out += '}';
            break;
        }
    }
}

}  // namespace agent
}  // namespace dpan
