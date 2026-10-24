// File: src/agent/json.hpp
//
// Minimal JSON value, parser and writer (RFC 8259) for the agent's text
// interfaces: the environment spec and configuration given to the C API and
// the Python bindings, and the predictions and statistics they return.
// Objects keep their members in the order they were added; numbers are
// doubles.
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dpan {
namespace agent {

class Json {
public:
    enum class Type { NUL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT };
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;

    Json() = default;  // null
    Json(std::nullptr_t) {}
    Json(bool value) : type_(Type::BOOLEAN), boolean_(value) {}
    Json(double value) : type_(Type::NUMBER), number_(value) {}
    Json(float value) : Json(static_cast<double>(value)) {}
    Json(int value) : Json(static_cast<double>(value)) {}
    Json(long value) : Json(static_cast<double>(value)) {}
    Json(long long value) : Json(static_cast<double>(value)) {}
    Json(unsigned value) : Json(static_cast<double>(value)) {}
    Json(unsigned long value) : Json(static_cast<double>(value)) {}
    Json(unsigned long long value) : Json(static_cast<double>(value)) {}
    Json(const char* value) : type_(Type::STRING), string_(value) {}
    Json(std::string value) : type_(Type::STRING), string_(std::move(value)) {}
    Json(Array value) : type_(Type::ARRAY), array_(std::move(value)) {}
    Json(Object value) : type_(Type::OBJECT), object_(std::move(value)) {}

    static Json MakeArray() { return Json(Array{}); }
    static Json MakeObject() { return Json(Object{}); }

    /// @throws std::invalid_argument with the position of the first error
    static Json Parse(const std::string& text);

    /// Compact text. Numbers keep 17 significant digits (float values
    /// round-trip); NaN and infinities are written as null.
    std::string Dump() const;

    Type GetType() const { return type_; }
    bool IsNull() const { return type_ == Type::NUL; }
    bool IsBool() const { return type_ == Type::BOOLEAN; }
    bool IsNumber() const { return type_ == Type::NUMBER; }
    bool IsString() const { return type_ == Type::STRING; }
    bool IsArray() const { return type_ == Type::ARRAY; }
    bool IsObject() const { return type_ == Type::OBJECT; }

    /// Accessors throw std::invalid_argument for the wrong type
    bool AsBool() const;
    double AsNumber() const;
    const std::string& AsString() const;
    const Array& AsArray() const;
    Array& AsArray();
    const Object& AsObject() const;
    Object& AsObject();

    /// Object member or nullptr
    const Json* Find(const std::string& key) const;
    /// Adds a member to an object (does not check for duplicates)
    Json& Set(const std::string& key, Json value);
    /// Appends to an array
    Json& Push(Json value);

private:
    void DumpTo(std::string& out) const;

    Type type_{Type::NUL};
    bool boolean_{false};
    double number_{0.0};
    std::string string_;
    Array array_;
    Object object_;
};

/// Name of a JSON type ("null", "boolean", "number", "string", "array", "object")
const char* ToString(Json::Type type);

}  // namespace agent
}  // namespace dpan
