// File: src/agent/encoders.hpp
//
// Input encoders (docs/AGENT_PLAN.md, section 2). Each input channel gets an
// encoder that turns its values into a fixed-length float vector; the
// ObservationEncoder concatenates them into one fused vector.
//
// Every encoder output ends with a "missing" indicator: 1 when the channel
// is absent from an observation, the fraction of missing elements for
// partially missing NUMERIC/TIME_SERIES vectors (NaN elements), else 0.
//
// Encoders learn from the data they see (running statistics, inferred sizes,
// observed categories, pixel range). Encode() is const and returns the
// encoding the value would have after Update() with it, so an observation
// can be encoded for prediction without changing any state.
//
// Encoders are not thread-safe; synchronize Update() with other calls.
#pragma once

#include "agent/environment.hpp"
#include "nn/matrix.hpp"

#include <cstdint>
#include <deque>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dpan {
namespace agent {

struct EncoderConfig {
    /// NUMERIC and TIME_SERIES values are standardized and clipped to ±clip_sigma
    float clip_sigma{5.0f};

    /// NUMERIC and TIME_SERIES: values per element after which the running
    /// statistics stop changing (0 = never). Fixed statistics keep the
    /// encoding of a value stable, so stored samples stay consistent with
    /// new ones.
    uint64_t statistics_limit{5000};

    /// CATEGORICAL with an open vocabulary: slots and the width of the
    /// learned embedding the agent puts after them. The first
    /// category_buckets distinct categories get one slot each, in the order
    /// they first appear; later ones are hashed into the slots.
    size_t category_buckets{1024};
    size_t category_embedding_size{8};

    /// TEXT: hash buckets for words and character tri-grams, embedding width
    size_t text_buckets{1024};
    size_t text_embedding_size{32};

    /// TEXT word sequences (docs/AGENT_PLAN.md, Phase 10): the first
    /// text_sequence_length words become word ids (0 = off), slots in order
    /// of first appearance up to text_vocabulary (later new words share one
    /// id). With text_word_vectors (a file in GloVe text format: a word and
    /// its values per line, an optional "count dimension" header), its first
    /// text_word_vectors_limit words take the first ids with their vectors,
    /// which are trained further (fixed if not train_word_vectors). Other words get learned vectors of
    /// text_word_dimension values (the file's dimension when there is one).
    /// The agent's network reads the words with a GRU of
    /// text_sequence_units (and, with text_bidirectional, a second one from
    /// the last word to the first).
    size_t text_sequence_length{32};
    size_t text_vocabulary{20000};
    std::string text_word_vectors;
    size_t text_word_vectors_limit{40000};
    /// The file's vectors are scaled by one factor to this mean row norm
    /// (0 = as in the file; GloVe 6B 100d is about 5.3). Larger vectors
    /// change less, relative to their size, when they are trained.
    float text_word_vector_norm{10.0f};
    bool train_word_vectors{true};
    bool text_bidirectional{true};
    /// The word reader pools the GRUs' states with learned attention weights
    /// (false: each GRU's final state and the mean of its states)
    bool text_attention{false};
    /// Training only: probability that a word is read as an empty vector, and
    /// that a value of the word reader's output is zeroed
    float text_word_dropout{0.0f};
    float text_dropout{0.0f};
    size_t text_word_dimension{50};
    size_t text_sequence_units{64};

    /// TIME_SERIES: number of most recent values kept per series
    size_t time_series_window{16};

    /// IMAGE: 0 = keep each image's size, scaled down if its longer side
    /// exceeds max_image_size; > 0 = resample to image_size x image_size.
    /// The agent's network reads images with convolution and pooling layers
    /// and projects them to image_embedding_size values.
    size_t image_size{0};
    size_t max_image_size{64};
    size_t image_embedding_size{64};

    /// Open-vocabulary categories recorded for inspection (hashing itself
    /// has no limit)
    size_t max_tracked_categories{10000};
};

/// @throws std::invalid_argument if a field is out of range
void ValidateEncoderConfig(const EncoderConfig& config);

class Encoder {
public:
    virtual ~Encoder() = default;

    const InputSpec& Spec() const { return spec_; }

    /// Length of the encoding; 0 while the size has not been inferred yet
    virtual size_t OutputSize() const = 0;

    /// Most non-zero values an encoding can have (bounds the memory of a
    /// sparse sample); OutputSize() unless the encoding is one-hot
    virtual size_t MaxNonZero() const { return OutputSize(); }

    /// Width of the learned embedding layer the agent applies to this
    /// encoding (0 = use the encoding directly). The layer is part of the
    /// agent's network, not of the encoder.
    virtual size_t EmbeddingSize() const { return 0; }

    /// Encoding of `value` (nullptr = missing) as it would be after
    /// Update(value). Empty if the size is unknown and cannot be inferred
    /// from `value` (missing).
    /// @throws std::invalid_argument if the value has the wrong type or size
    virtual std::vector<float> Encode(const Value* value) const = 0;

    /// Learn from `value` (statistics, sizes, vocabularies, windows)
    /// @throws std::invalid_argument like Encode()
    virtual void Update(const Value* value) = 0;

    /// Clear per-episode state (TIME_SERIES windows); learned statistics stay
    virtual void ResetEpisode() {}

    /// Write / restore everything learned (statistics, sizes, vocabularies,
    /// windows). LoadState expects an encoder built from the same spec.
    /// @throws std::runtime_error on malformed data
    virtual void SaveState(std::ostream& out) const = 0;
    virtual void LoadState(std::istream& in) = 0;

protected:
    explicit Encoder(InputSpec spec) : spec_(std::move(spec)) {}

    /// "input 'name' (TYPE)" for error messages
    std::string Describe() const;

    /// Throws std::invalid_argument naming the channel, what it expects and
    /// what it got
    [[noreturn]] void RejectType(const Value& value, const char* expected) const;

    InputSpec spec_;
};

/// Running mean and variance (Welford)
struct RunningStats {
    uint64_t count{0};
    double mean{0.0};
    double m2{0.0};

    void Add(double x);
    double StdDev() const;

    /// (x - mean) / stddev clipped to ±clip; 0 when the deviation is 0
    float Standardize(double x, float clip) const;

    void Save(std::ostream& out) const;
    void Load(std::istream& in);
};

/// NUMERIC: standardized with running mean and variance, clipped to ±5σ.
/// Output: size values, missing indicator.
class NumericEncoder : public Encoder {
public:
    NumericEncoder(InputSpec spec, const EncoderConfig& config);

    size_t OutputSize() const override { return size_ == 0 ? 0 : size_ + 1; }
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

    size_t Size() const { return size_; }
    const std::vector<RunningStats>& Statistics() const { return stats_; }

private:
    std::vector<float> Values(const Value& value) const;

    bool Learning(const RunningStats& stats) const {
        return statistics_limit_ == 0 || stats.count < statistics_limit_;
    }

    float clip_;
    uint64_t statistics_limit_;
    size_t size_;
    std::vector<RunningStats> stats_;
};

/// CATEGORICAL with known categories: one-hot.
/// Output: one slot per category, an "unknown category" slot, missing indicator.
class OneHotEncoder : public Encoder {
public:
    explicit OneHotEncoder(InputSpec spec);

    size_t OutputSize() const override { return spec_.categories.size() + 2; }
    size_t MaxNonZero() const override { return 1; }
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

    /// Times each category was seen; the last entry counts unknown categories
    const std::vector<uint64_t>& Counts() const { return counts_; }

private:
    /// Category index, categories.size() for an unknown category, nullopt
    /// for a missing value (NaN)
    std::optional<size_t> Index(const Value& value) const;

    std::map<std::string, size_t> index_;
    std::vector<uint64_t> counts_;
};

/// CATEGORICAL with an open vocabulary. Categories get slots in the order
/// they first appear, one each until all category_buckets slots are taken
/// (no collisions); after that, new categories are hashed into the slots.
/// Output: category_buckets slots (one set), missing indicator.
class HashedCategoryEncoder : public Encoder {
public:
    HashedCategoryEncoder(InputSpec spec, const EncoderConfig& config);

    size_t OutputSize() const override { return buckets_ + 1; }
    size_t MaxNonZero() const override { return 1; }
    size_t EmbeddingSize() const override { return embedding_size_; }
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

    /// Categories seen so far with their counts (up to max_tracked_categories)
    const std::map<std::string, uint64_t>& ObservedCategories() const { return observed_; }

    /// Slot of a category: its own if it has one, else the next free slot if
    /// any is left (the one Update will give it), else its hash bucket
    size_t Bucket(const std::string& category) const;

    /// Categories with their own slot, in slot order
    const std::vector<std::string>& SlotCategories() const { return slot_names_; }

private:
    /// Category as a string; nullopt for a missing value (NaN)
    std::optional<std::string> Category(const Value& value) const;

    size_t buckets_;
    size_t embedding_size_;
    size_t max_tracked_;
    std::map<std::string, uint64_t> observed_;
    std::map<std::string, size_t> slots_;
    std::vector<std::string> slot_names_;
};

/// TEXT: lowercased words and character tri-grams of each word (with
/// boundary markers "^word$"; tri-grams are of bytes, so non-ASCII
/// characters are split), counted in text_buckets hash buckets and
/// scaled to unit length. Output: text_buckets values, missing indicator.
class TextEncoder : public Encoder {
public:
    TextEncoder(InputSpec spec, const EncoderConfig& config);

    size_t OutputSize() const override { return buckets_ + sequence_length_ + 1; }
    size_t EmbeddingSize() const override { return embedding_size_; }
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

    /// Lowercased words: runs of ASCII letters and digits, or of non-ASCII
    /// bytes (kept whole, so UTF-8 words are not split)
    static std::vector<std::string> Words(const std::string& text);

    /// Hash features of `text` before scaling (bucket -> count)
    std::map<size_t, float> Features(const std::string& text) const;

    uint64_t TextsSeen() const { return texts_seen_; }

    /// Word sequences: ids per text (0 = none), the vocabulary and vectors
    size_t SequenceLength() const { return sequence_length_; }
    /// Ids the next text would get for its first SequenceLength() words
    std::vector<uint32_t> WordIds(const std::string& text) const;
    /// Words with an id (pretrained first), in id order from id 1
    const std::vector<std::string>& Vocabulary() const { return words_; }
    size_t PretrainedWords() const;
    /// Learned-vector ids after the pretrained ones (the last is shared by
    /// the words beyond the vocabulary)
    size_t LearnedWords() const { return vocabulary_ + 1; }
    size_t WordDimension() const;
    /// The file's vectors (rows = pretrained words); null without a file
    std::shared_ptr<const nn::Matrix> WordVectors() const;
    const EncoderConfig& Config() const { return config_; }

private:
    /// Reads text_word_vectors once (on first use, so a restored encoder
    /// does not need the file)
    void LoadWordVectors() const;

    size_t buckets_;
    size_t embedding_size_;
    size_t sequence_length_;
    size_t vocabulary_;
    EncoderConfig config_;
    uint64_t texts_seen_{0};
    mutable bool vectors_loaded_{false};
    mutable std::shared_ptr<nn::Matrix> vectors_;
    mutable std::vector<std::string> words_;
    mutable std::map<std::string, uint32_t> ids_;
};

/// TIME_SERIES: each observation adds one value per series. Per series, the
/// last `time_series_window` values (standardized with the series' running
/// statistics, oldest first, padded with the oldest value), their
/// differences, and the window's mean, standard deviation, minimum and
/// maximum. Then the window fill fraction and the missing indicator.
/// A missing value repeats the series' previous value.
class TimeSeriesEncoder : public Encoder {
public:
    TimeSeriesEncoder(InputSpec spec, const EncoderConfig& config);

    size_t OutputSize() const override;
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;
    void ResetEpisode() override;

    size_t SeriesCount() const { return series_; }
    size_t Window() const { return window_; }

    /// Encoding values per series
    size_t PerSeriesSize() const { return 2 * window_ + 3; }

    /// Running statistics of one series' values
    const RunningStats& SeriesStatistics(size_t series) const { return state_.at(series).stats; }

private:
    struct Series {
        RunningStats stats;
        std::deque<double> values;  // raw values, oldest first
    };

    std::vector<float> Values(const Value& value) const;

    /// Applies one step to `series` (simulated in Encode, committed in Update)
    void Step(std::vector<Series>& series, const Value* value) const;

    bool Learning(const RunningStats& stats) const {
        return statistics_limit_ == 0 || stats.count < statistics_limit_;
    }

    float clip_;
    uint64_t statistics_limit_;
    size_t window_;
    size_t series_;
    std::vector<Series> state_;
};

/// IMAGE: pixel values scaled to [0, 1] (the range, [0, 1] or [0, 255] or
/// wider, is inferred from the largest pixel seen), converted to the spec's
/// channel count and kept at their size (longer side at most
/// max_image_size) or resampled to image_size x image_size, by area
/// averaging. Output: height x width x channels values (row-major,
/// interleaved channels), missing indicator. The size is unknown until the
/// first image if the spec does not declare it.
class ImageEncoder : public Encoder {
public:
    ImageEncoder(InputSpec spec, const EncoderConfig& config);

    size_t OutputSize() const override;
    size_t EmbeddingSize() const override { return embedding_size_; }

    /// Size of the encoded image (0 while unknown)
    size_t EncodedHeight() const { return width_ == 0 && fixed_side_ == 0 ? 0 : EncodedDims(width_, height_).first; }
    size_t EncodedWidth() const { return width_ == 0 && fixed_side_ == 0 ? 0 : EncodedDims(width_, height_).second; }
    std::vector<float> Encode(const Value* value) const override;
    void Update(const Value* value) override;
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

    /// Image size declared in the spec or inferred from the first image
    size_t Width() const { return width_; }
    size_t Height() const { return height_; }

    /// Divisor that maps pixel values to [0, 1]
    float PixelScale() const { return ScaleFor(max_pixel_); }

private:
    const Image& Check(const Value& value) const;
    static float ScaleFor(float max_pixel);

    /// (height, width) of the encoding of a width x height image
    std::pair<size_t, size_t> EncodedDims(size_t width, size_t height) const;

    size_t fixed_side_;
    size_t max_side_;
    size_t embedding_size_;
    size_t width_;
    size_t height_;
    float max_pixel_{0.0f};
};

/// Encoder for one input channel
/// @throws std::invalid_argument if the spec or config is invalid
std::unique_ptr<Encoder> CreateEncoder(const InputSpec& spec, const EncoderConfig& config = {});

/// Position of one channel in the fused vector
struct ChannelLayout {
    std::string name;
    InputType type;
    size_t offset;
    size_t size;
    size_t embedding_size;
    /// IMAGE channels: shape of the encoded image (0 for other types)
    size_t image_height{0};
    size_t image_width{0};
    size_t image_channels{0};
    /// TEXT channels with word sequences: the channel is [hash features
    /// (size - sequence_length - 1)] [sequence_length word ids] [missing];
    /// ids 1..pretrained rows have the vectors in `word_vectors`, the next
    /// learned_words ids learned vectors of word_dimension values
    size_t sequence_length{0};
    size_t learned_words{0};
    size_t word_dimension{0};
    size_t sequence_units{0};
    bool bidirectional{false};
    bool attention{false};
    float word_dropout{0.0f};
    float dropout{0.0f};
    bool train_word_vectors{false};
    std::shared_ptr<const nn::Matrix> word_vectors;
};

/// Encodes whole observations into one fused vector (channels in spec order)
class ObservationEncoder {
public:
    /// @throws std::invalid_argument if the inputs or config are invalid
    explicit ObservationEncoder(std::vector<InputSpec> inputs, EncoderConfig config = {});

    /// True once every channel's size is known
    bool Ready() const;

    /// Fused vector length (0 until Ready())
    size_t OutputSize() const;
    /// Most non-zero values of a fused vector (sum over the channels)
    size_t MaxNonZero() const;

    /// Channel positions in the fused vector (offsets valid once Ready())
    std::vector<ChannelLayout> Layout() const;

    /// Encode and learn from an observation. Returns nullopt while a
    /// channel's size is still unknown (the observation is still learned
    /// from). Nothing is learned if the observation is rejected.
    /// @throws std::invalid_argument for an unknown channel name or a value
    ///         of the wrong type or size
    std::optional<std::vector<float>> Encode(const Observation& observation);

    /// Like Encode() but changes no state
    std::optional<std::vector<float>> Peek(const Observation& observation) const;

    /// Clear per-episode state of all channels
    void ResetEpisode();

    /// Write / restore the learned state of every channel (the inputs and
    /// config come from the constructor)
    void SaveState(std::ostream& out) const;
    void LoadState(std::istream& in);

    size_t ChannelCount() const { return encoders_.size(); }
    const Encoder& Channel(size_t index) const { return *encoders_.at(index); }

    /// @throws std::out_of_range for an unknown name
    const Encoder& Channel(const std::string& name) const;

    const EncoderConfig& Config() const { return config_; }

private:
    std::vector<const Value*> Lookup(const Observation& observation) const;

    EncoderConfig config_;
    std::vector<std::unique_ptr<Encoder>> encoders_;
    std::map<std::string, size_t> index_;
};

}  // namespace agent
}  // namespace dpan
