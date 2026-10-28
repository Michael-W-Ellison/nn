// File: src/agent/encoders.cpp
#include "agent/encoders.hpp"

#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace agent {

namespace {

// Hash seeds keep the feature families in separate hash spaces
constexpr uint64_t kCategorySeed = 1;
constexpr uint64_t kWordSeed = 2;
constexpr uint64_t kTrigramSeed = 3;

/// FNV-1a with a seed, followed by the splitmix64 finalizer. Fixed across
/// platforms and compilers (unlike std::hash), so saved models stay valid.
uint64_t Hash(const std::string& text, uint64_t seed) {
    uint64_t h = 1469598103934665603ULL ^ (seed * 0x9E3779B97F4A7C15ULL);
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

/// Number or vector as floats
bool AsFloats(const Value& value, std::vector<float>& out) {
    if (const double* number = std::get_if<double>(&value)) {
        out.assign(1, static_cast<float>(*number));
        return true;
    }
    if (const auto* vector = std::get_if<std::vector<float>>(&value)) {
        out = *vector;
        return true;
    }
    return false;
}

float Clip(double x, float limit) {
    return static_cast<float>(std::max<double>(-limit, std::min<double>(limit, x)));
}

/// Area-averaging weights mapping n_in samples onto n_out samples
std::vector<std::vector<std::pair<size_t, float>>> AreaWeights(size_t n_in, size_t n_out) {
    std::vector<std::vector<std::pair<size_t, float>>> weights(n_out);
    double ratio = static_cast<double>(n_in) / static_cast<double>(n_out);
    for (size_t o = 0; o < n_out; ++o) {
        double start = o * ratio;
        double end = (o + 1) * ratio;
        auto first = static_cast<size_t>(std::floor(start));
        auto last = std::min(n_in, static_cast<size_t>(std::ceil(end)));
        for (size_t i = first; i < last; ++i) {
            double overlap = std::min(end, static_cast<double>(i + 1)) -
                             std::max(start, static_cast<double>(i));
            if (overlap > 0.0) {
                weights[o].emplace_back(i, static_cast<float>(overlap / ratio));
            }
        }
    }
    return weights;
}

}  // namespace

void ValidateEncoderConfig(const EncoderConfig& config) {
    if (!(config.clip_sigma > 0.0f)) {
        throw std::invalid_argument("EncoderConfig: clip_sigma must be > 0");
    }
    if (config.category_buckets == 0 || config.text_buckets == 0) {
        throw std::invalid_argument("EncoderConfig: hash bucket counts must be > 0");
    }
    if (config.time_series_window == 0) {
        throw std::invalid_argument("EncoderConfig: time_series_window must be > 0");
    }
    if (config.max_image_size == 0) {
        throw std::invalid_argument("EncoderConfig: max_image_size must be > 0");
    }
    if (config.text_sequence_length > 0 &&
        (config.text_vocabulary == 0 || config.text_word_dimension == 0 || config.text_sequence_units == 0)) {
        throw std::invalid_argument(
            "EncoderConfig: with text_sequence_length > 0, text_vocabulary, text_word_dimension and "
            "text_sequence_units must be > 0");
    }
}

// ============================================================================
// Encoder
// ============================================================================

std::string Encoder::Describe() const {
    return "input '" + spec_.name + "' (" + ToString(spec_.type) + ")";
}

void Encoder::RejectType(const Value& value, const char* expected) const {
    throw std::invalid_argument(Describe() + " expects " + expected + ", got " +
                                ValueTypeName(value));
}

void RunningStats::Add(double x) {
    ++count;
    double delta = x - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (x - mean);
}

double RunningStats::StdDev() const {
    return count == 0 ? 0.0 : std::sqrt(m2 / static_cast<double>(count));
}

float RunningStats::Standardize(double x, float clip) const {
    double deviation = x - mean;
    double sd = StdDev();
    if (sd <= 1e-12) {
        return 0.0f;
    }
    return Clip(deviation / sd, clip);
}

void RunningStats::Save(std::ostream& out) const {
    io::Write<uint64_t>(out, count);
    io::Write<double>(out, mean);
    io::Write<double>(out, m2);
}

void RunningStats::Load(std::istream& in) {
    count = io::Read<uint64_t>(in);
    mean = io::Read<double>(in);
    m2 = io::Read<double>(in);
}

namespace {

void SaveStats(std::ostream& out, const std::vector<RunningStats>& stats) {
    io::WriteSize(out, stats.size());
    for (const auto& s : stats) s.Save(out);
}

std::vector<RunningStats> LoadStats(std::istream& in) {
    std::vector<RunningStats> stats(io::ReadSize(in, 1u << 24));
    for (auto& s : stats) s.Load(in);
    return stats;
}

}  // namespace

// ============================================================================
// NumericEncoder
// ============================================================================

NumericEncoder::NumericEncoder(InputSpec spec, const EncoderConfig& config)
    : Encoder(std::move(spec)), clip_(config.clip_sigma), statistics_limit_(config.statistics_limit), size_(spec_.size) {
    stats_.resize(size_);
}

std::vector<float> NumericEncoder::Values(const Value& value) const {
    std::vector<float> values;
    if (!AsFloats(value, values)) {
        RejectType(value, "a number or a vector");
    }
    if (values.empty()) {
        throw std::invalid_argument(Describe() + " received an empty vector");
    }
    if (size_ != 0 && values.size() != size_) {
        throw std::invalid_argument(Describe() + " expects " + std::to_string(size_) +
                                    " values, got " + std::to_string(values.size()));
    }
    return values;
}

std::vector<float> NumericEncoder::Encode(const Value* value) const {
    if (value == nullptr) {
        if (size_ == 0) {
            return {};
        }
        std::vector<float> out(size_ + 1, 0.0f);
        out.back() = 1.0f;
        return out;
    }

    std::vector<float> values = Values(*value);
    std::vector<float> out(values.size() + 1, 0.0f);
    size_t missing = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            ++missing;
            continue;
        }
        RunningStats stats = i < stats_.size() ? stats_[i] : RunningStats{};
        if (Learning(stats)) stats.Add(values[i]);
        out[i] = stats.Standardize(values[i], clip_);
    }
    out.back() = static_cast<float>(missing) / static_cast<float>(values.size());
    return out;
}

void NumericEncoder::Update(const Value* value) {
    if (value == nullptr) {
        return;
    }
    std::vector<float> values = Values(*value);
    if (size_ == 0) {
        size_ = values.size();
        stats_.resize(size_);
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i])) {
            if (Learning(stats_[i])) stats_[i].Add(values[i]);
        }
    }
}

void NumericEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, size_);
    SaveStats(out, stats_);
}

void NumericEncoder::LoadState(std::istream& in) {
    size_ = io::ReadSize(in, 1u << 24);
    stats_ = LoadStats(in);
}

// ============================================================================
// OneHotEncoder
// ============================================================================

OneHotEncoder::OneHotEncoder(InputSpec spec)
    : Encoder(std::move(spec)), counts_(spec_.categories.size() + 1, 0) {
    for (size_t i = 0; i < spec_.categories.size(); ++i) {
        index_.emplace(spec_.categories[i], i);
    }
}

std::optional<size_t> OneHotEncoder::Index(const Value& value) const {
    size_t n = spec_.categories.size();
    if (const auto* name = std::get_if<std::string>(&value)) {
        auto it = index_.find(*name);
        return it == index_.end() ? n : it->second;
    }
    if (const double* number = std::get_if<double>(&value)) {
        if (std::isnan(*number)) {
            return std::nullopt;
        }
        if (*number < 0.0 || *number >= static_cast<double>(n) || std::floor(*number) != *number) {
            std::ostringstream message;
            message.imbue(std::locale::classic());
            message << Describe() << ": category index " << *number << " is not an integer in [0, "
                    << n << ")";
            throw std::invalid_argument(message.str());
        }
        return static_cast<size_t>(*number);
    }
    RejectType(value, "a category name or index");
}

std::vector<float> OneHotEncoder::Encode(const Value* value) const {
    std::vector<float> out(OutputSize(), 0.0f);
    std::optional<size_t> index = value ? Index(*value) : std::nullopt;
    if (index) {
        out[*index] = 1.0f;
    } else {
        out.back() = 1.0f;
    }
    return out;
}

void OneHotEncoder::Update(const Value* value) {
    if (value == nullptr) {
        return;
    }
    if (auto index = Index(*value)) {
        ++counts_[*index];
    }
}

void OneHotEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, counts_.size());
    for (uint64_t c : counts_) io::Write<uint64_t>(out, c);
}

void OneHotEncoder::LoadState(std::istream& in) {
    size_t n = io::ReadSize(in, 1u << 24);
    if (n != counts_.size()) throw std::runtime_error(Describe() + ": saved state has another category count");
    for (auto& c : counts_) c = io::Read<uint64_t>(in);
}

// ============================================================================
// HashedCategoryEncoder
// ============================================================================

HashedCategoryEncoder::HashedCategoryEncoder(InputSpec spec, const EncoderConfig& config)
    : Encoder(std::move(spec)),
      buckets_(config.category_buckets),
      embedding_size_(config.category_embedding_size),
      max_tracked_(config.max_tracked_categories) {}

std::optional<std::string> HashedCategoryEncoder::Category(const Value& value) const {
    if (const auto* name = std::get_if<std::string>(&value)) {
        return *name;
    }
    if (const double* number = std::get_if<double>(&value)) {
        if (std::isnan(*number)) {
            return std::nullopt;
        }
        // Integers print without a decimal point, so 3 and 3.0 are one category
        std::ostringstream text;
        text.imbue(std::locale::classic());
        if (std::floor(*number) == *number && std::fabs(*number) < 1e15) {
            text << static_cast<long long>(*number);
        } else {
            text.precision(17);
            text << *number;
        }
        return text.str();
    }
    RejectType(value, "a category name or number");
}

size_t HashedCategoryEncoder::Bucket(const std::string& category) const {
    auto it = slots_.find(category);
    if (it != slots_.end()) return it->second;
    if (slot_names_.size() < buckets_) return slot_names_.size();
    return static_cast<size_t>(Hash(category, kCategorySeed) % buckets_);
}

std::vector<float> HashedCategoryEncoder::Encode(const Value* value) const {
    std::vector<float> out(OutputSize(), 0.0f);
    std::optional<std::string> category = value ? Category(*value) : std::nullopt;
    if (category) {
        out[Bucket(*category)] = 1.0f;
    } else {
        out.back() = 1.0f;
    }
    return out;
}

void HashedCategoryEncoder::Update(const Value* value) {
    if (value == nullptr) {
        return;
    }
    if (auto category = Category(*value)) {
        if (slot_names_.size() < buckets_ && slots_.emplace(*category, slot_names_.size()).second) {
            slot_names_.push_back(*category);
        }
        auto it = observed_.find(*category);
        if (it != observed_.end()) {
            ++it->second;
        } else if (observed_.size() < max_tracked_) {
            observed_.emplace(*category, 1);
        }
    }
}

void HashedCategoryEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, observed_.size());
    for (const auto& [category, count] : observed_) {
        io::WriteString(out, category);
        io::Write<uint64_t>(out, count);
    }
    io::WriteSize(out, slot_names_.size());
    for (const auto& category : slot_names_) io::WriteString(out, category);
}

void HashedCategoryEncoder::LoadState(std::istream& in) {
    observed_.clear();
    size_t n = io::ReadSize(in, 1u << 30);
    for (size_t i = 0; i < n; ++i) {
        std::string category = io::ReadString(in);
        observed_[category] = io::Read<uint64_t>(in);
    }
    slots_.clear();
    slot_names_.clear();
    n = io::ReadSize(in, 1u << 30);
    if (n > buckets_) throw std::runtime_error("HashedCategoryEncoder::LoadState: more slots than category_buckets");
    for (size_t i = 0; i < n; ++i) {
        std::string category = io::ReadString(in);
        if (!slots_.emplace(category, i).second) {
            throw std::runtime_error("HashedCategoryEncoder::LoadState: duplicate category");
        }
        slot_names_.push_back(std::move(category));
    }
}

// ============================================================================
// TextEncoder
// ============================================================================

TextEncoder::TextEncoder(InputSpec spec, const EncoderConfig& config)
    : Encoder(std::move(spec)),
      buckets_(config.text_buckets),
      embedding_size_(config.text_embedding_size),
      sequence_length_(config.text_sequence_length),
      vocabulary_(config.text_vocabulary),
      config_(config) {}

void TextEncoder::LoadWordVectors() const {
    if (vectors_loaded_) return;
    vectors_loaded_ = true;
    if (sequence_length_ == 0 || config_.text_word_vectors.empty()) return;
    std::ifstream file(config_.text_word_vectors);
    if (!file) {
        throw std::invalid_argument("TEXT word vectors: cannot open '" + config_.text_word_vectors + "'");
    }
    std::vector<float> values;
    size_t dimension = 0;
    std::string line;
    size_t line_number = 0;
    while (words_.size() < config_.text_word_vectors_limit && std::getline(file, line)) {
        ++line_number;
        std::istringstream fields(line);
        fields.imbue(std::locale::classic());
        std::string word;
        if (!(fields >> word)) continue;
        std::vector<float> vector;
        float x;
        while (fields >> x) vector.push_back(x);
        if (line_number == 1 && vector.size() == 1) continue;  // "count dimension" header
        if (dimension == 0) dimension = vector.size();
        if (vector.empty() || vector.size() != dimension) {
            throw std::invalid_argument("TEXT word vectors: line " + std::to_string(line_number) + " of '" +
                                        config_.text_word_vectors + "' has " + std::to_string(vector.size()) +
                                        " values, expected " + std::to_string(dimension));
        }
        if (ids_.count(word)) continue;
        ids_.emplace(word, static_cast<uint32_t>(words_.size() + 1));
        words_.push_back(word);
        values.insert(values.end(), vector.begin(), vector.end());
    }
    if (words_.empty()) {
        throw std::invalid_argument("TEXT word vectors: no vectors in '" + config_.text_word_vectors + "'");
    }
    vectors_ = std::make_shared<nn::Matrix>(words_.size(), dimension, std::move(values));
}

size_t TextEncoder::PretrainedWords() const {
    LoadWordVectors();
    return vectors_ ? vectors_->Rows() : 0;
}

size_t TextEncoder::WordDimension() const {
    LoadWordVectors();
    return vectors_ ? vectors_->Cols() : config_.text_word_dimension;
}

std::shared_ptr<const nn::Matrix> TextEncoder::WordVectors() const {
    LoadWordVectors();
    return vectors_;
}

std::vector<uint32_t> TextEncoder::WordIds(const std::string& text) const {
    LoadWordVectors();
    std::vector<uint32_t> ids(sequence_length_, 0);
    const size_t capacity = PretrainedWords() + vocabulary_;
    const auto unknown = static_cast<uint32_t>(capacity + 1);
    std::map<std::string, uint32_t> fresh;  // words this text would add
    auto words = Words(text);
    for (size_t i = 0; i < words.size() && i < sequence_length_; ++i) {
        auto known = ids_.find(words[i]);
        if (known != ids_.end()) {
            ids[i] = known->second;
            continue;
        }
        auto added = fresh.find(words[i]);
        if (added == fresh.end()) {
            size_t next = words_.size() + fresh.size();
            added = fresh.emplace(words[i], next < capacity ? static_cast<uint32_t>(next + 1) : unknown).first;
        }
        ids[i] = added->second;
    }
    return ids;
}

std::vector<std::string> TextEncoder::Words(const std::string& text) {
    std::vector<std::string> words;
    std::string word;
    for (unsigned char c : text) {
        bool ascii_alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        if (ascii_alnum || c >= 0x80) {
            word += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
        } else if (!word.empty()) {
            words.push_back(std::move(word));
            word.clear();
        }
    }
    if (!word.empty()) {
        words.push_back(std::move(word));
    }
    return words;
}

std::map<size_t, float> TextEncoder::Features(const std::string& text) const {
    std::map<size_t, float> features;
    for (const auto& word : Words(text)) {
        features[Hash(word, kWordSeed) % buckets_] += 1.0f;
        std::string marked = "^" + word + "$";
        for (size_t i = 0; i + 3 <= marked.size(); ++i) {
            features[Hash(marked.substr(i, 3), kTrigramSeed) % buckets_] += 1.0f;
        }
    }
    return features;
}

std::vector<float> TextEncoder::Encode(const Value* value) const {
    std::vector<float> out(OutputSize(), 0.0f);
    if (value == nullptr) {
        out.back() = 1.0f;
        return out;
    }
    const auto* text = std::get_if<std::string>(value);
    if (text == nullptr) {
        RejectType(*value, "text");
    }
    double norm = 0.0;
    for (const auto& [bucket, count] : Features(*text)) {
        out[bucket] = count;
        norm += static_cast<double>(count) * count;
    }
    if (norm > 0.0) {
        auto scale = static_cast<float>(1.0 / std::sqrt(norm));
        for (size_t i = 0; i < buckets_; ++i) {
            out[i] *= scale;
        }
    }
    if (sequence_length_ > 0) {
        auto ids = WordIds(*text);
        for (size_t i = 0; i < ids.size(); ++i) out[buckets_ + i] = static_cast<float>(ids[i]);
    }
    return out;
}

void TextEncoder::Update(const Value* value) {
    if (value == nullptr) {
        return;
    }
    const auto* text = std::get_if<std::string>(value);
    if (text == nullptr) {
        RejectType(*value, "text");
    }
    ++texts_seen_;
    if (sequence_length_ == 0) return;
    auto ids = WordIds(*text);
    auto words = Words(*text);
    const uint32_t unknown = static_cast<uint32_t>(PretrainedWords() + vocabulary_ + 1);
    for (size_t i = 0; i < ids.size() && i < words.size(); ++i) {
        if (ids[i] == unknown || ids[i] <= words_.size()) continue;
        ids_.emplace(words[i], ids[i]);
        words_.push_back(words[i]);
    }
}

void TextEncoder::SaveState(std::ostream& out) const {
    io::Write<uint64_t>(out, texts_seen_);
    LoadWordVectors();
    io::WriteSize(out, words_.size());
    for (const auto& word : words_) io::WriteString(out, word);
    io::Write<uint8_t>(out, vectors_ ? 1 : 0);
    if (vectors_) vectors_->Serialize(out);
}

void TextEncoder::LoadState(std::istream& in) {
    texts_seen_ = io::Read<uint64_t>(in);
    words_.clear();
    ids_.clear();
    size_t count = io::ReadSize(in, 1u << 26);
    for (size_t i = 0; i < count; ++i) {
        std::string word = io::ReadString(in, 1u << 16);
        ids_.emplace(word, static_cast<uint32_t>(i + 1));
        words_.push_back(std::move(word));
    }
    vectors_.reset();
    if (io::Read<uint8_t>(in)) vectors_ = std::make_shared<nn::Matrix>(nn::Matrix::Deserialize(in));
    vectors_loaded_ = true;  // the file is not needed again
    if (ids_.size() != words_.size() || (vectors_ && vectors_->Rows() > words_.size())) {
        throw std::runtime_error("TextEncoder::LoadState: malformed vocabulary");
    }
}

// ============================================================================
// TimeSeriesEncoder
// ============================================================================

TimeSeriesEncoder::TimeSeriesEncoder(InputSpec spec, const EncoderConfig& config)
    : Encoder(std::move(spec)),
      clip_(config.clip_sigma),
      statistics_limit_(config.statistics_limit),
      window_(config.time_series_window),
      series_(spec_.size) {
    state_.resize(series_);
}

size_t TimeSeriesEncoder::OutputSize() const {
    return series_ == 0 ? 0 : series_ * PerSeriesSize() + 2;
}

std::vector<float> TimeSeriesEncoder::Values(const Value& value) const {
    std::vector<float> values;
    if (!AsFloats(value, values)) {
        RejectType(value, "a number or a vector (one value per series)");
    }
    if (values.empty()) {
        throw std::invalid_argument(Describe() + " received an empty vector");
    }
    if (series_ != 0 && values.size() != series_) {
        throw std::invalid_argument(Describe() + " expects " + std::to_string(series_) +
                                    " series, got " + std::to_string(values.size()));
    }
    return values;
}

void TimeSeriesEncoder::Step(std::vector<Series>& series, const Value* value) const {
    std::vector<float> values;
    if (value != nullptr) {
        values = Values(*value);
    }
    for (size_t s = 0; s < series.size(); ++s) {
        auto& current = series[s];
        if (!values.empty() && std::isfinite(values[s])) {
            if (Learning(current.stats)) current.stats.Add(values[s]);
            current.values.push_back(values[s]);
        } else if (!current.values.empty()) {
            current.values.push_back(current.values.back());  // carry forward
        }
        while (current.values.size() > window_) {
            current.values.pop_front();
        }
    }
}

std::vector<float> TimeSeriesEncoder::Encode(const Value* value) const {
    size_t count = series_;
    std::vector<float> values;
    if (value != nullptr) {
        values = Values(*value);
        count = values.size();
    }
    if (count == 0) {
        return {};
    }

    std::vector<Series> series = state_;
    series.resize(count);
    Step(series, value);

    std::vector<float> out;
    out.reserve(count * PerSeriesSize() + 2);
    size_t filled = 0;
    for (const auto& current : series) {
        size_t length = current.values.size();
        filled += length;

        std::vector<float> window(window_, 0.0f);
        if (length > 0) {
            size_t pad = window_ - length;
            for (size_t i = 0; i < length; ++i) {
                window[pad + i] = current.stats.Standardize(current.values[i], clip_);
            }
            std::fill(window.begin(), window.begin() + pad, window[pad]);
        }
        out.insert(out.end(), window.begin(), window.end());
        for (size_t i = 0; i + 1 < window_; ++i) {
            out.push_back(window[i + 1] - window[i]);
        }

        float mean = 0.0f, sd = 0.0f, low = 0.0f, high = 0.0f;
        if (length > 0) {
            auto first = window.begin() + (window_ - length);
            low = *std::min_element(first, window.end());
            high = *std::max_element(first, window.end());
            double sum = 0.0, sum_sq = 0.0;
            for (auto it = first; it != window.end(); ++it) {
                sum += *it;
                sum_sq += static_cast<double>(*it) * *it;
            }
            double m = sum / length;
            mean = static_cast<float>(m);
            sd = static_cast<float>(std::sqrt(std::max(0.0, sum_sq / length - m * m)));
        }
        out.push_back(mean);
        out.push_back(sd);
        out.push_back(low);
        out.push_back(high);
    }

    out.push_back(static_cast<float>(filled) / static_cast<float>(count * window_));
    if (value == nullptr) {
        out.push_back(1.0f);
    } else {
        size_t missing = std::count_if(values.begin(), values.end(),
                                       [](float x) { return !std::isfinite(x); });
        out.push_back(static_cast<float>(missing) / static_cast<float>(count));
    }
    return out;
}

void TimeSeriesEncoder::Update(const Value* value) {
    if (series_ == 0) {
        if (value == nullptr) {
            return;
        }
        series_ = Values(*value).size();
        state_.resize(series_);
    }
    Step(state_, value);
}

void TimeSeriesEncoder::ResetEpisode() {
    for (auto& current : state_) {
        current.values.clear();
    }
}

void TimeSeriesEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, series_);
    io::WriteSize(out, state_.size());
    for (const auto& current : state_) {
        current.stats.Save(out);
        io::WriteSize(out, current.values.size());
        for (double v : current.values) io::Write<double>(out, v);
    }
}

void TimeSeriesEncoder::LoadState(std::istream& in) {
    series_ = io::ReadSize(in, 1u << 24);
    state_.assign(io::ReadSize(in, 1u << 24), Series());
    for (auto& current : state_) {
        current.stats.Load(in);
        size_t n = io::ReadSize(in, 1u << 24);
        for (size_t i = 0; i < n; ++i) current.values.push_back(io::Read<double>(in));
    }
}

// ============================================================================
// ImageEncoder
// ============================================================================

ImageEncoder::ImageEncoder(InputSpec spec, const EncoderConfig& config)
    : Encoder(std::move(spec)),
      fixed_side_(config.image_size),
      max_side_(config.max_image_size),
      embedding_size_(config.image_embedding_size),
      width_(spec_.width),
      height_(spec_.height) {}

std::pair<size_t, size_t> ImageEncoder::EncodedDims(size_t width, size_t height) const {
    if (fixed_side_ > 0) return {fixed_side_, fixed_side_};
    size_t longer = std::max(width, height);
    if (longer <= max_side_) return {height, width};
    double scale = static_cast<double>(max_side_) / static_cast<double>(longer);
    return {std::max<size_t>(1, static_cast<size_t>(std::lround(height * scale))),
            std::max<size_t>(1, static_cast<size_t>(std::lround(width * scale)))};
}

size_t ImageEncoder::OutputSize() const {
    if (width_ == 0 && fixed_side_ == 0) return 0;  // size known with the first image
    auto [h, w] = EncodedDims(width_, height_);
    return h * w * spec_.image_channels + 1;
}

float ImageEncoder::ScaleFor(float max_pixel) {
    if (max_pixel <= 1.0f) return 1.0f;
    if (max_pixel <= 255.0f) return 255.0f;
    if (max_pixel <= 65535.0f) return 65535.0f;
    return max_pixel;
}

const Image& ImageEncoder::Check(const Value& value) const {
    const auto* image = std::get_if<Image>(&value);
    if (image == nullptr) {
        RejectType(value, "an image");
    }
    if (image->width == 0 || image->height == 0 || image->channels == 0) {
        throw std::invalid_argument(Describe() + " received an image with a zero dimension");
    }
    if (image->pixels.size() != image->width * image->height * image->channels) {
        throw std::invalid_argument(Describe() + ": image has " + std::to_string(image->pixels.size()) +
                                    " pixel values, expected width x height x channels = " +
                                    std::to_string(image->width * image->height * image->channels));
    }
    if (width_ != 0 && (image->width != width_ || image->height != height_)) {
        throw std::invalid_argument(Describe() + " expects " + std::to_string(width_) + "x" +
                                    std::to_string(height_) + " images, got " +
                                    std::to_string(image->width) + "x" +
                                    std::to_string(image->height));
    }
    return *image;
}

std::vector<float> ImageEncoder::Encode(const Value* value) const {
    const size_t channels = spec_.image_channels;
    if (value == nullptr) {
        std::vector<float> out(OutputSize(), 0.0f);
        if (!out.empty()) out.back() = 1.0f;
        return out;
    }
    const Image& image = Check(*value);
    const auto [out_h, out_w] = EncodedDims(image.width, image.height);
    std::vector<float> out(out_h * out_w * channels + 1, 0.0f);

    float max_pixel = max_pixel_;
    for (float p : image.pixels) {
        if (std::isfinite(p)) {
            max_pixel = std::max(max_pixel, p);
        }
    }
    const float scale = ScaleFor(max_pixel);
    auto pixel = [&](size_t index) {
        float p = image.pixels[index];
        return std::isfinite(p) ? std::min(1.0f, std::max(0.0f, p / scale)) : 0.0f;
    };

    // Convert to the spec's channel layout: gray, RGB or RGBA. Source images
    // with 1-2 channels are gray (+ alpha); with 3-4 they are RGB (+ alpha).
    const size_t source = image.channels;
    const size_t w = image.width, h = image.height;
    std::vector<float> converted(w * h * channels);
    for (size_t p = 0; p < w * h; ++p) {
        size_t base = p * source;
        float* target = &converted[p * channels];
        auto rgb = [&](size_t k) { return source <= 2 ? pixel(base) : pixel(base + std::min(k, source - 1)); };
        float alpha = (source == 2 || source == 4) ? pixel(base + source - 1) : 1.0f;
        if (channels == 1) {
            target[0] = source <= 2 ? pixel(base) : (rgb(0) + rgb(1) + rgb(2)) / 3.0f;
        } else {
            for (size_t k = 0; k < 3; ++k) {
                target[k] = rgb(k);
            }
            if (channels == 4) {
                target[3] = alpha;
            }
        }
    }

    if (out_h == h && out_w == w) {
        std::copy(converted.begin(), converted.end(), out.begin());
        return out;
    }

    // Resample: columns first, then rows
    auto x_weights = AreaWeights(w, out_w);
    auto y_weights = AreaWeights(h, out_h);
    std::vector<float> columns(h * out_w * channels, 0.0f);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < out_w; ++x) {
            for (const auto& [source_x, weight] : x_weights[x]) {
                for (size_t k = 0; k < channels; ++k) {
                    columns[(y * out_w + x) * channels + k] +=
                        weight * converted[(y * w + source_x) * channels + k];
                }
            }
        }
    }
    for (size_t y = 0; y < out_h; ++y) {
        for (const auto& [source_y, weight] : y_weights[y]) {
            for (size_t i = 0; i < out_w * channels; ++i) {
                out[y * out_w * channels + i] += weight * columns[source_y * out_w * channels + i];
            }
        }
    }
    return out;
}

void ImageEncoder::Update(const Value* value) {
    if (value == nullptr) {
        return;
    }
    const Image& image = Check(*value);
    if (width_ == 0) {
        width_ = image.width;
        height_ = image.height;
    }
    for (float p : image.pixels) {
        if (std::isfinite(p)) {
            max_pixel_ = std::max(max_pixel_, p);
        }
    }
}

void ImageEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, width_);
    io::WriteSize(out, height_);
    io::Write<float>(out, max_pixel_);
}

void ImageEncoder::LoadState(std::istream& in) {
    width_ = io::ReadSize(in, 1u << 24);
    height_ = io::ReadSize(in, 1u << 24);
    max_pixel_ = io::Read<float>(in);
}

// ============================================================================
// Factory and ObservationEncoder
// ============================================================================

namespace {

void ThrowIfInvalid(const std::vector<InputSpec>& inputs) {
    auto errors = InputValidationErrors(inputs);
    if (errors.empty()) {
        return;
    }
    std::string message = "Invalid input spec:";
    for (const auto& error : errors) {
        message += "\n  - " + error;
    }
    throw std::invalid_argument(message);
}

std::unique_ptr<Encoder> MakeEncoder(const InputSpec& spec, const EncoderConfig& config) {
    switch (spec.type) {
        case InputType::NUMERIC:
            return std::make_unique<NumericEncoder>(spec, config);
        case InputType::CATEGORICAL:
            if (spec.categories.empty()) {
                return std::make_unique<HashedCategoryEncoder>(spec, config);
            }
            return std::make_unique<OneHotEncoder>(spec);
        case InputType::TEXT:
            return std::make_unique<TextEncoder>(spec, config);
        case InputType::IMAGE:
            return std::make_unique<ImageEncoder>(spec, config);
        case InputType::TIME_SERIES:
            return std::make_unique<TimeSeriesEncoder>(spec, config);
    }
    throw std::invalid_argument("Unknown input type");
}

}  // namespace

std::unique_ptr<Encoder> CreateEncoder(const InputSpec& spec, const EncoderConfig& config) {
    ValidateEncoderConfig(config);
    ThrowIfInvalid({spec});
    return MakeEncoder(spec, config);
}

ObservationEncoder::ObservationEncoder(std::vector<InputSpec> inputs, EncoderConfig config)
    : config_(config) {
    ValidateEncoderConfig(config_);
    ThrowIfInvalid(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        index_.emplace(inputs[i].name, i);
        encoders_.push_back(MakeEncoder(inputs[i], config_));
    }
}

bool ObservationEncoder::Ready() const {
    return std::all_of(encoders_.begin(), encoders_.end(),
                       [](const auto& encoder) { return encoder->OutputSize() > 0; });
}

size_t ObservationEncoder::MaxNonZero() const {
    size_t total = 0;
    for (const auto& encoder : encoders_) total += encoder->MaxNonZero();
    return total;
}

size_t ObservationEncoder::OutputSize() const {
    if (!Ready()) {
        return 0;
    }
    size_t total = 0;
    for (const auto& encoder : encoders_) {
        total += encoder->OutputSize();
    }
    return total;
}

std::vector<ChannelLayout> ObservationEncoder::Layout() const {
    std::vector<ChannelLayout> layout;
    size_t offset = 0;
    for (const auto& encoder : encoders_) {
        size_t size = encoder->OutputSize();
        ChannelLayout channel{encoder->Spec().name, encoder->Spec().type, offset, size, encoder->EmbeddingSize()};
        if (const auto* image = dynamic_cast<const ImageEncoder*>(encoder.get())) {
            channel.image_height = image->EncodedHeight();
            channel.image_width = image->EncodedWidth();
            channel.image_channels = image->Spec().image_channels;
        }
        if (const auto* text = dynamic_cast<const TextEncoder*>(encoder.get()); text && text->SequenceLength() > 0) {
            channel.sequence_length = text->SequenceLength();
            channel.learned_words = text->LearnedWords();
            channel.word_dimension = text->WordDimension();
            channel.sequence_units = text->Config().text_sequence_units;
            channel.train_word_vectors = text->Config().train_word_vectors;
            channel.bidirectional = text->Config().text_bidirectional;
            channel.word_vectors = text->WordVectors();
        }
        layout.push_back(channel);
        offset += size;
    }
    return layout;
}

const Encoder& ObservationEncoder::Channel(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::out_of_range("Unknown input channel '" + name + "'");
    }
    return *encoders_[it->second];
}

std::vector<const Value*> ObservationEncoder::Lookup(const Observation& observation) const {
    std::vector<const Value*> values(encoders_.size(), nullptr);
    for (const auto& [name, value] : observation) {
        auto it = index_.find(name);
        if (it == index_.end()) {
            throw std::invalid_argument("Unknown input channel '" + name + "'");
        }
        values[it->second] = &value;
    }
    return values;
}

std::optional<std::vector<float>> ObservationEncoder::Peek(const Observation& observation) const {
    auto values = Lookup(observation);
    std::vector<float> fused;
    bool complete = true;
    for (size_t i = 0; i < encoders_.size(); ++i) {
        std::vector<float> part = encoders_[i]->Encode(values[i]);
        complete = complete && !part.empty();
        fused.insert(fused.end(), part.begin(), part.end());
    }
    if (!complete) {
        return std::nullopt;
    }
    return fused;
}

std::optional<std::vector<float>> ObservationEncoder::Encode(const Observation& observation) {
    // Peek validates every value before any state changes
    auto fused = Peek(observation);
    auto values = Lookup(observation);
    for (size_t i = 0; i < encoders_.size(); ++i) {
        encoders_[i]->Update(values[i]);
    }
    return fused;
}

void ObservationEncoder::SaveState(std::ostream& out) const {
    io::WriteSize(out, encoders_.size());
    for (const auto& encoder : encoders_) encoder->SaveState(out);
}

void ObservationEncoder::LoadState(std::istream& in) {
    if (io::ReadSize(in, 1u << 20) != encoders_.size()) {
        throw std::runtime_error("ObservationEncoder::LoadState: saved state has another channel count");
    }
    for (auto& encoder : encoders_) encoder->LoadState(in);
}

void ObservationEncoder::ResetEpisode() {
    for (auto& encoder : encoders_) {
        encoder->ResetEpisode();
    }
}

}  // namespace agent
}  // namespace dpan
