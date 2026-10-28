// File: tests/agent/encoders_test.cpp
#include "agent/encoders.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace agent {
namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();

Value Num(double x) { return Value(x); }
Value Vec(std::vector<float> v) { return Value(std::move(v)); }
Value Text(const std::string& s) { return Value(s); }

float Cosine(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return static_cast<float>(dot / std::sqrt(na * nb));
}

/// Message of the std::invalid_argument thrown by `f`
template <typename F>
std::string ErrorOf(F f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

// ============================================================================
// NumericEncoder
// ============================================================================

TEST(NumericEncoderTest, InfersSizeFromFirstValue) {
    NumericEncoder encoder({"x", InputType::NUMERIC}, {});
    EXPECT_EQ(0u, encoder.OutputSize());
    EXPECT_TRUE(encoder.Encode(nullptr).empty());

    Value v = Vec({1, 2, 3});
    EXPECT_EQ(4u, encoder.Encode(&v).size());  // inferred from the value itself
    encoder.Update(&v);
    EXPECT_EQ(3u, encoder.Size());
    EXPECT_EQ(4u, encoder.OutputSize());
}

TEST(NumericEncoderTest, StandardizesWithRunningStatistics) {
    NumericEncoder encoder({"x", InputType::NUMERIC, 1}, {});
    for (double x : {2, 4, 4, 4, 5, 5, 7}) {
        Value v = Num(x);
        encoder.Update(&v);
    }
    // With 9 included: mean 5, population standard deviation 2
    Value nine = Num(9);
    auto encoded = encoder.Encode(&nine);
    ASSERT_EQ(2u, encoded.size());
    EXPECT_NEAR(2.0f, encoded[0], 1e-5f);
    EXPECT_FLOAT_EQ(0.0f, encoded[1]);

    // Encode() is the encoding after Update()
    encoder.Update(&nine);
    EXPECT_NEAR(encoder.Statistics()[0].Standardize(9, 5.0f), encoded[0], 1e-6f);
    EXPECT_NEAR(5.0, encoder.Statistics()[0].mean, 1e-9);
}

TEST(NumericEncoderTest, ClipsToConfiguredSigma) {
    NumericEncoder encoder({"x", InputType::NUMERIC}, {});
    for (int i = 0; i < 100; ++i) {
        Value v = Num(i % 2);
        encoder.Update(&v);
    }
    Value far = Num(1e6);
    EXPECT_FLOAT_EQ(5.0f, encoder.Encode(&far)[0]);
    Value low = Num(-1e6);
    EXPECT_FLOAT_EQ(-5.0f, encoder.Encode(&low)[0]);
}

TEST(NumericEncoderTest, FirstValueEncodesAsZero) {
    NumericEncoder encoder({"x", InputType::NUMERIC}, {});
    Value v = Num(123.0);
    EXPECT_FLOAT_EQ(0.0f, encoder.Encode(&v)[0]);
}

TEST(NumericEncoderTest, MissingChannelAndMissingElements) {
    NumericEncoder encoder({"x", InputType::NUMERIC, 2}, {});
    auto missing = encoder.Encode(nullptr);
    EXPECT_EQ((std::vector<float>{0, 0, 1}), missing);

    Value first = Vec({1, 10});
    encoder.Update(&first);
    Value partial = Vec({kNaN, 20});
    auto encoded = encoder.Encode(&partial);
    EXPECT_FLOAT_EQ(0.0f, encoded[0]);
    EXPECT_GT(encoded[1], 0.0f);
    EXPECT_FLOAT_EQ(0.5f, encoded[2]);

    encoder.Update(&partial);
    EXPECT_EQ(1u, encoder.Statistics()[0].count);  // NaN not learned
    EXPECT_EQ(2u, encoder.Statistics()[1].count);
}

TEST(NumericEncoderTest, RejectsWrongTypeAndSize) {
    NumericEncoder encoder({"speed", InputType::NUMERIC, 2}, {});
    Value text = Text("fast");
    EXPECT_EQ("input 'speed' (NUMERIC) expects a number or a vector, got text",
              ErrorOf([&] { encoder.Encode(&text); }));
    Value three = Vec({1, 2, 3});
    EXPECT_EQ("input 'speed' (NUMERIC) expects 2 values, got 3",
              ErrorOf([&] { encoder.Update(&three); }));
    Value empty = Vec({});
    EXPECT_THROW(encoder.Encode(&empty), std::invalid_argument);
}

// ============================================================================
// OneHotEncoder
// ============================================================================

TEST(OneHotEncoderTest, EncodesKnownUnknownAndMissing) {
    OneHotEncoder encoder({"color", InputType::CATEGORICAL, 0, {"red", "green", "blue"}});
    EXPECT_EQ(5u, encoder.OutputSize());

    Value green = Text("green");
    EXPECT_EQ((std::vector<float>{0, 1, 0, 0, 0}), encoder.Encode(&green));
    Value index = Num(2);
    EXPECT_EQ((std::vector<float>{0, 0, 1, 0, 0}), encoder.Encode(&index));
    Value other = Text("purple");
    EXPECT_EQ((std::vector<float>{0, 0, 0, 1, 0}), encoder.Encode(&other));
    EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 1}), encoder.Encode(nullptr));
    Value nan = Num(kNaN);
    EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 1}), encoder.Encode(&nan));
}

TEST(OneHotEncoderTest, CountsCategories) {
    OneHotEncoder encoder({"color", InputType::CATEGORICAL, 0, {"red", "green"}});
    for (const char* name : {"red", "red", "green", "pink"}) {
        Value v = Text(name);
        encoder.Update(&v);
    }
    EXPECT_EQ((std::vector<uint64_t>{2, 1, 1}), encoder.Counts());
}

TEST(OneHotEncoderTest, RejectsBadIndicesAndTypes) {
    OneHotEncoder encoder({"color", InputType::CATEGORICAL, 0, {"red", "green"}});
    Value fraction = Num(0.5);
    Value high = Num(2);
    Value vector = Vec({1});
    EXPECT_NE(std::string::npos,
              ErrorOf([&] { encoder.Encode(&fraction); }).find("is not an integer in [0, 2)"));
    EXPECT_THROW(encoder.Encode(&high), std::invalid_argument);
    EXPECT_EQ("input 'color' (CATEGORICAL) expects a category name or index, got vector",
              ErrorOf([&] { encoder.Update(&vector); }));
}

// ============================================================================
// HashedCategoryEncoder
// ============================================================================

TEST(HashedCategoryEncoderTest, SameCategorySameBucket) {
    HashedCategoryEncoder encoder({"user", InputType::CATEGORICAL}, {});
    EXPECT_EQ(1025u, encoder.OutputSize());
    EXPECT_EQ(8u, encoder.EmbeddingSize());

    Value a = Text("alice");
    auto first = encoder.Encode(&a);
    EXPECT_EQ(first, encoder.Encode(&a));
    EXPECT_FLOAT_EQ(1.0f, std::accumulate(first.begin(), first.end(), 0.0f));
    EXPECT_FLOAT_EQ(1.0f, first[encoder.Bucket("alice")]);

    // Numbers are categories too; 3 and 3.0 and "3" are the same one
    Value three = Num(3);
    Value three_text = Text("3");
    EXPECT_EQ(encoder.Encode(&three), encoder.Encode(&three_text));

    auto missing = encoder.Encode(nullptr);
    EXPECT_FLOAT_EQ(1.0f, missing.back());
    EXPECT_FLOAT_EQ(1.0f, std::accumulate(missing.begin(), missing.end(), 0.0f));
}

TEST(HashedCategoryEncoderTest, CategoriesGetOwnSlotsInOrderOfAppearance) {
    HashedCategoryEncoder encoder({"user", InputType::CATEGORICAL}, {});
    for (const char* name : {"carol", "alice", "carol", "bob"}) {
        Value v = Text(name);
        // Encoded before Update (as ObservationEncoder does): a new category
        // already gets the slot Update will give it
        auto encoded = encoder.Encode(&v);
        encoder.Update(&v);
        EXPECT_FLOAT_EQ(1.0f, encoded[encoder.Bucket(name)]) << name;
    }
    EXPECT_EQ(0u, encoder.Bucket("carol"));
    EXPECT_EQ(1u, encoder.Bucket("alice"));
    EXPECT_EQ(2u, encoder.Bucket("bob"));
    EXPECT_EQ((std::vector<std::string>{"carol", "alice", "bob"}), encoder.SlotCategories());
    EXPECT_EQ(3u, encoder.Bucket("dave"));  // the next free slot

    // 1000 categories, 1024 slots: no two share a slot
    std::set<size_t> used;
    for (int i = 0; i < 1000; ++i) {
        Value v = Text("user" + std::to_string(i));
        encoder.Update(&v);
        used.insert(encoder.Bucket("user" + std::to_string(i)));
    }
    EXPECT_EQ(1000u, used.size());
}

TEST(HashedCategoryEncoderTest, HashesCategoriesOnceTheSlotsAreTaken) {
    EncoderConfig config;
    config.category_buckets = 8;
    HashedCategoryEncoder encoder({"user", InputType::CATEGORICAL}, config);
    HashedCategoryEncoder other({"other", InputType::CATEGORICAL}, config);
    for (int i = 0; i < 8; ++i) {
        Value v = Text("first" + std::to_string(i));
        encoder.Update(&v);
        other.Update(&v);
    }
    std::set<size_t> used;
    for (int i = 0; i < 1000; ++i) {
        std::string name = "user" + std::to_string(i);
        Value v = Text(name);
        size_t before = encoder.Bucket(name);
        encoder.Update(&v);
        EXPECT_EQ(before, encoder.Bucket(name));  // no slot of its own any more
        EXPECT_LT(before, 8u);
        used.insert(before);
    }
    EXPECT_EQ(8u, used.size());
    // Hash buckets do not depend on the channel; saved models rely on them
    EXPECT_EQ(encoder.Bucket("alice"), other.Bucket("alice"));
    EXPECT_EQ(8u, encoder.SlotCategories().size());
}

TEST(HashedCategoryEncoderTest, SavesAndLoadsSlots) {
    HashedCategoryEncoder encoder({"user", InputType::CATEGORICAL}, {});
    for (const char* name : {"x", "y", "z"}) {
        Value v = Text(name);
        encoder.Update(&v);
    }
    std::stringstream buffer;
    encoder.SaveState(buffer);
    HashedCategoryEncoder loaded({"user", InputType::CATEGORICAL}, {});
    loaded.LoadState(buffer);
    EXPECT_EQ(encoder.SlotCategories(), loaded.SlotCategories());
    for (const char* name : {"x", "y", "z", "new"}) EXPECT_EQ(encoder.Bucket(name), loaded.Bucket(name)) << name;

    // More saved slots than this encoder has is malformed
    EncoderConfig small;
    small.category_buckets = 2;
    HashedCategoryEncoder too_small({"user", InputType::CATEGORICAL}, small);
    std::stringstream again;
    encoder.SaveState(again);
    EXPECT_THROW(too_small.LoadState(again), std::runtime_error);
}

TEST(HashedCategoryEncoderTest, TracksObservedVocabularyUpToLimit) {
    EncoderConfig config;
    config.max_tracked_categories = 3;
    HashedCategoryEncoder encoder({"user", InputType::CATEGORICAL}, config);
    for (const char* name : {"a", "b", "a", "c", "d", "a"}) {
        Value v = Text(name);
        encoder.Update(&v);
    }
    const auto& observed = encoder.ObservedCategories();
    ASSERT_EQ(3u, observed.size());
    EXPECT_EQ(3u, observed.at("a"));
    EXPECT_EQ(1u, observed.at("b"));
    EXPECT_EQ(0u, observed.count("d"));
}

// ============================================================================
// TextEncoder
// ============================================================================

TEST(TextEncoderTest, WordIdsFollowFirstAppearance) {
    EncoderConfig config;
    config.text_sequence_length = 4;
    config.text_vocabulary = 3;
    TextEncoder encoder({"t", InputType::TEXT}, config);
    EXPECT_EQ(3u + 1u, encoder.LearnedWords());  // the last id is shared by later words
    Value a = Text("The dog saw the cat");
    auto before = encoder.Encode(&a);
    // Ids as they will be after Update: the, dog, saw, the (the fifth word is cut)
    std::vector<float> ids(before.begin() + 1024, before.begin() + 1028);
    EXPECT_EQ((std::vector<float>{1, 2, 3, 1}), ids);
    encoder.Update(&a);
    EXPECT_EQ(before, encoder.Encode(&a));
    EXPECT_EQ((std::vector<std::string>{"the", "dog", "saw"}), encoder.Vocabulary());
    // The vocabulary is full: new words share id 4; shorter texts end with 0
    Value b = Text("cat dog");
    EXPECT_EQ((std::vector<uint32_t>{4, 2, 0, 0}), encoder.WordIds("cat dog"));
    encoder.Update(&b);
    EXPECT_EQ(3u, encoder.Vocabulary().size());

    EncoderConfig off;
    off.text_sequence_length = 0;
    EXPECT_EQ(1025u, TextEncoder({"t", InputType::TEXT}, off).OutputSize());
    EncoderConfig bad;
    bad.text_vocabulary = 0;
    EXPECT_THROW(ValidateEncoderConfig(bad), std::invalid_argument);
}

TEST(TextEncoderTest, ReadsWordVectorsAndRestoresWithoutTheFile) {
    std::string path = ::testing::TempDir() + "/dpan_word_vectors.txt";
    {
        std::ofstream file(path);
        file << "4 3\n"
             << "good 1 0 0\n"
             << "bad -1 0 0\n"
             << "movie 0 1 0\n"
             << "unused 0 0 1\n";
    }
    EncoderConfig config;
    config.text_sequence_length = 3;
    config.text_vocabulary = 5;
    config.text_word_vectors = path;
    config.text_word_vectors_limit = 3;
    TextEncoder encoder({"t", InputType::TEXT}, config);
    EXPECT_EQ(3u, encoder.PretrainedWords());  // the limit
    EXPECT_EQ(3u, encoder.WordDimension());
    auto vectors = encoder.WordVectors();
    ASSERT_TRUE(vectors);
    EXPECT_EQ((std::vector<float>{-1, 0, 0}), vectors->Row(1));
    // Known words keep their ids; new ones follow the file's words
    Value v = Text("a good movie");
    encoder.Update(&v);
    EXPECT_EQ((std::vector<uint32_t>{4, 1, 3}), encoder.WordIds("a good movie"));

    std::stringstream state;
    encoder.SaveState(state);
    std::remove(path.c_str());
    TextEncoder restored({"t", InputType::TEXT}, config);
    restored.LoadState(state);
    EXPECT_EQ(encoder.WordIds("a bad movie"), restored.WordIds("a bad movie"));
    EXPECT_EQ(vectors->Values(), restored.WordVectors()->Values());

    // Without the file a new encoder cannot start
    TextEncoder missing({"t", InputType::TEXT}, config);
    EXPECT_THROW(missing.WordIds("good"), std::invalid_argument);
}

TEST(TextEncoderTest, SplitsLowercaseWords) {
    EXPECT_EQ((std::vector<std::string>{"hello", "world", "42", "na\xC3\xAFve"}),
              TextEncoder::Words("Hello, WORLD!  42 na\xC3\xAFve"));
    EXPECT_TRUE(TextEncoder::Words(" ,.; ").empty());
}

TEST(TextEncoderTest, CountsWordsAndTrigrams) {
    TextEncoder encoder({"t", InputType::TEXT}, {});
    // "cat" -> word + "^ca", "cat", "at$"
    auto features = encoder.Features("cat");
    float total = 0;
    for (const auto& [bucket, count] : features) total += count;
    EXPECT_FLOAT_EQ(4.0f, total);
    EXPECT_FLOAT_EQ(8.0f, [&] {
        float t = 0;
        for (const auto& [bucket, count] : encoder.Features("cat cat")) t += count;
        return t;
    }());
}

TEST(TextEncoderTest, UnitLengthAndCaseInsensitive) {
    TextEncoder encoder({"t", InputType::TEXT}, {});
    EXPECT_EQ(1024u + 32u + 1u, encoder.OutputSize());  // features, word ids, missing
    EXPECT_EQ(32u, encoder.EmbeddingSize());

    Value a = Text("The quick brown fox");
    Value b = Text("THE QUICK BROWN FOX!");
    auto ea = encoder.Encode(&a);
    EXPECT_EQ(ea, encoder.Encode(&b));
    double norm = 0;
    for (size_t i = 0; i < 1024; ++i) norm += ea[i] * ea[i];
    EXPECT_NEAR(1.0, norm, 1e-5);
    EXPECT_FLOAT_EQ(0.0f, ea.back());
}

TEST(TextEncoderTest, SimilarTextsAreCloser) {
    TextEncoder encoder({"t", InputType::TEXT}, {});
    Value base = Text("server disk is full");
    Value near = Text("the server disks are full");
    Value far = Text("lunch menu tomorrow");
    auto eb = encoder.Encode(&base), en = encoder.Encode(&near), ef = encoder.Encode(&far);
    EXPECT_GT(Cosine(eb, en, 1024), 0.5f);
    EXPECT_LT(Cosine(eb, ef, 1024), 0.2f);
}

TEST(TextEncoderTest, EmptyAndMissingText) {
    TextEncoder encoder({"t", InputType::TEXT}, {});
    Value empty = Text("");
    auto e = encoder.Encode(&empty);
    EXPECT_FLOAT_EQ(0.0f, std::accumulate(e.begin(), e.end(), 0.0f));
    auto missing = encoder.Encode(nullptr);
    EXPECT_FLOAT_EQ(1.0f, missing.back());
    Value number = Num(1);
    EXPECT_EQ("input 't' (TEXT) expects text, got number", ErrorOf([&] { encoder.Update(&number); }));
}

// ============================================================================
// TimeSeriesEncoder
// ============================================================================

EncoderConfig SmallWindow() {
    EncoderConfig config;
    config.time_series_window = 4;
    return config;
}

TEST(TimeSeriesEncoderTest, InfersSeriesCount) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES}, SmallWindow());
    EXPECT_EQ(0u, encoder.OutputSize());
    Value v = Vec({1, 2, 3});
    encoder.Update(&v);
    EXPECT_EQ(3u, encoder.SeriesCount());
    EXPECT_EQ(3u * (2 * 4 + 3) + 2, encoder.OutputSize());
    EXPECT_EQ(encoder.OutputSize(), encoder.Encode(&v).size());
}

TEST(TimeSeriesEncoderTest, WindowDifferencesAndSummary) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES, 1}, SmallWindow());
    for (double x : {1.0, 2.0}) {
        Value v = Num(x);
        encoder.Update(&v);
    }
    Value three = Num(3);
    auto e = encoder.Encode(&three);
    ASSERT_EQ(13u, e.size());

    // Values 1, 2, 3: mean 2, standard deviation sqrt(2/3)
    float z = static_cast<float>(1.0 / std::sqrt(2.0 / 3.0));
    std::vector<float> window = {-z, -z, 0, z};  // padded with the oldest value
    for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(window[i], e[i], 1e-5f) << i;
    std::vector<float> diffs = {0, z, z};
    for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(diffs[i], e[4 + i], 1e-5f) << i;
    EXPECT_NEAR(0.0f, e[7], 1e-5f);   // mean of the real values
    EXPECT_NEAR(1.0f, e[8], 1e-5f);   // their standard deviation
    EXPECT_NEAR(-z, e[9], 1e-5f);     // min
    EXPECT_NEAR(z, e[10], 1e-5f);     // max
    EXPECT_FLOAT_EQ(0.75f, e[11]);    // window fill
    EXPECT_FLOAT_EQ(0.0f, e[12]);     // missing
}

TEST(TimeSeriesEncoderTest, KeepsOnlyTheLastWindow) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES, 1}, SmallWindow());
    for (int i = 0; i < 10; ++i) {
        Value v = Num(i);
        encoder.Update(&v);
    }
    Value next = Num(10);
    auto e = encoder.Encode(&next);
    EXPECT_FLOAT_EQ(1.0f, e[11]);
    // Values 7..10 are evenly spaced, so all differences are equal
    EXPECT_NEAR(e[4], e[5], 1e-5f);
    EXPECT_NEAR(e[5], e[6], 1e-5f);
    EXPECT_GT(e[4], 0.0f);
}

TEST(TimeSeriesEncoderTest, MissingStepRepeatsPreviousValue) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES, 1}, SmallWindow());
    for (double x : {1.0, 2.0, 3.0}) {
        Value v = Num(x);
        encoder.Update(&v);
    }
    auto e = encoder.Encode(nullptr);
    EXPECT_FLOAT_EQ(1.0f, e[12]);
    EXPECT_FLOAT_EQ(1.0f, e[11]);
    EXPECT_FLOAT_EQ(e[2], e[3]);      // repeated value
    EXPECT_FLOAT_EQ(0.0f, e[6]);      // no change

    Value partial = Vec({kNaN});
    EXPECT_FLOAT_EQ(1.0f, encoder.Encode(&partial)[12]);
}

TEST(TimeSeriesEncoderTest, EncodeDoesNotChangeStateAndResetClearsWindow) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES, 1}, SmallWindow());
    for (double x : {1.0, 5.0, 2.0}) {
        Value v = Num(x);
        encoder.Update(&v);
    }
    Value v = Num(4);
    auto first = encoder.Encode(&v);
    EXPECT_EQ(first, encoder.Encode(&v));

    encoder.ResetEpisode();
    auto after_reset = encoder.Encode(&v);
    EXPECT_FLOAT_EQ(0.25f, after_reset[11]);  // one value in the window
    EXPECT_NE(0.0f, after_reset[3]);          // statistics were kept
}

TEST(TimeSeriesEncoderTest, RejectsWrongSeriesCount) {
    TimeSeriesEncoder encoder({"load", InputType::TIME_SERIES, 2}, SmallWindow());
    Value one = Num(1);
    EXPECT_EQ("input 'load' (TIME_SERIES) expects 2 series, got 1",
              ErrorOf([&] { encoder.Encode(&one); }));
}

// ============================================================================
// ImageEncoder
// ============================================================================

/// Phase 1 behaviour: every image resampled to 16 x 16
EncoderConfig Resample16() {
    EncoderConfig config;
    config.image_size = 16;
    return config;
}

Image Gray(size_t w, size_t h, std::function<float(size_t, size_t)> f) {
    Image image{w, h, 1, std::vector<float>(w * h)};
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) image.pixels[y * w + x] = f(x, y);
    return image;
}

TEST(ImageEncoderTest, DownsamplesByAreaAveraging) {
    ImageEncoder encoder({"cam", InputType::IMAGE}, Resample16());
    EXPECT_EQ(16u * 16u + 1, encoder.OutputSize());
    Value checker = Gray(32, 32, [](size_t x, size_t y) { return float((x + y) % 2); });
    auto e = encoder.Encode(&checker);
    for (size_t i = 0; i < 256; ++i) EXPECT_NEAR(0.5f, e[i], 1e-6f) << i;
    EXPECT_FLOAT_EQ(0.0f, e[256]);
}

TEST(ImageEncoderTest, UpsamplesSmallImages) {
    ImageEncoder encoder({"cam", InputType::IMAGE}, Resample16());
    Value small = Gray(4, 4, [](size_t x, size_t y) { return (y * 4 + x) / 15.0f; });
    auto e = encoder.Encode(&small);
    for (size_t y = 0; y < 16; ++y)
        for (size_t x = 0; x < 16; ++x)
            EXPECT_NEAR(((y / 4) * 4 + x / 4) / 15.0f, e[y * 16 + x], 1e-5f);
}

TEST(ImageEncoderTest, PreservesMeanForUnevenSizes) {
    ImageEncoder encoder({"cam", InputType::IMAGE}, Resample16());
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(0, 1);
    Value image = Gray(20, 12, [&](size_t, size_t) { return u(rng); });
    const auto& pixels = std::get<Image>(image).pixels;
    float input_mean = std::accumulate(pixels.begin(), pixels.end(), 0.0f) / pixels.size();
    auto e = encoder.Encode(&image);
    float output_mean = std::accumulate(e.begin(), e.end() - 1, 0.0f) / 256.0f;
    EXPECT_NEAR(input_mean, output_mean, 1e-4f);
}

TEST(ImageEncoderTest, InfersPixelRange) {
    ImageEncoder encoder({"cam", InputType::IMAGE}, Resample16());
    EXPECT_FLOAT_EQ(1.0f, encoder.PixelScale());
    Value bytes = Gray(16, 16, [](size_t x, size_t) { return x < 8 ? 0.0f : 255.0f; });
    auto e = encoder.Encode(&bytes);
    EXPECT_FLOAT_EQ(1.0f, e[15]);
    EXPECT_FLOAT_EQ(0.0f, e[0]);
    encoder.Update(&bytes);
    EXPECT_FLOAT_EQ(255.0f, encoder.PixelScale());

    // Later images use the learned range
    Value dim = Gray(16, 16, [](size_t, size_t) { return 51.0f; });
    EXPECT_NEAR(0.2f, encoder.Encode(&dim)[0], 1e-6f);
}

TEST(ImageEncoderTest, ConvertsChannels) {
    InputSpec gray_spec{"cam", InputType::IMAGE};
    ImageEncoder gray(gray_spec, Resample16());
    Value rgb = Image{1, 1, 3, {0.3f, 0.6f, 0.9f}};
    EXPECT_NEAR(0.6f, gray.Encode(&rgb)[0], 1e-6f);

    InputSpec rgb_spec{"cam", InputType::IMAGE, 0, {}, 0, 0, 3};
    ImageEncoder color(rgb_spec, Resample16());
    EXPECT_EQ(16u * 16u * 3 + 1, color.OutputSize());
    Value single = Image{1, 1, 1, {0.4f}};
    auto e = color.Encode(&single);
    EXPECT_FLOAT_EQ(0.4f, e[0]);
    EXPECT_FLOAT_EQ(0.4f, e[1]);
    EXPECT_FLOAT_EQ(0.4f, e[2]);

    InputSpec rgba_spec{"cam", InputType::IMAGE, 0, {}, 0, 0, 4};
    ImageEncoder alpha(rgba_spec, Resample16());
    auto ea = alpha.Encode(&rgb);
    EXPECT_FLOAT_EQ(0.9f, ea[2]);
    EXPECT_FLOAT_EQ(1.0f, ea[3]);  // opaque
}

TEST(ImageEncoderTest, KeepsSizeUpToTheLimitByDefault) {
    ImageEncoder encoder({"cam", InputType::IMAGE}, {});
    EXPECT_EQ(0u, encoder.OutputSize());  // unknown until the first image
    EXPECT_TRUE(encoder.Encode(nullptr).empty());
    EXPECT_EQ(64u, encoder.EmbeddingSize());
    Value small = Gray(20, 12, [](size_t x, size_t y) { return float(x + y) / 30.0f; });
    auto e = encoder.Encode(&small);
    ASSERT_EQ(20u * 12u + 1, e.size());
    EXPECT_FLOAT_EQ(float(3 + 2) / 30.0f, e[2 * 20 + 3]);  // pixel (3, 2) unchanged
    encoder.Update(&small);
    EXPECT_EQ(12u, encoder.EncodedHeight());
    EXPECT_EQ(20u, encoder.EncodedWidth());

    ImageEncoder large({"cam", InputType::IMAGE, 0, {}, 128, 96}, {});
    EXPECT_EQ(48u, large.EncodedHeight());  // longer side scaled to 64
    EXPECT_EQ(64u, large.EncodedWidth());
    EXPECT_EQ(48u * 64u + 1, large.OutputSize());
    Value big = Gray(128, 96, [](size_t, size_t) { return 0.5f; });
    EXPECT_NEAR(0.5f, large.Encode(&big)[100], 1e-6f);
}

TEST(ImageEncoderTest, ChecksSizes) {
    ImageEncoder declared({"cam", InputType::IMAGE, 0, {}, 8, 8}, {});
    Value wrong = Gray(4, 4, [](size_t, size_t) { return 0.0f; });
    EXPECT_EQ("input 'cam' (IMAGE) expects 8x8 images, got 4x4",
              ErrorOf([&] { declared.Encode(&wrong); }));

    ImageEncoder inferred({"cam", InputType::IMAGE}, {});
    inferred.Update(&wrong);
    EXPECT_EQ(4u, inferred.Width());
    Value other = Gray(5, 4, [](size_t, size_t) { return 0.0f; });
    EXPECT_THROW(inferred.Encode(&other), std::invalid_argument);

    Value broken = Image{2, 2, 1, {0.0f}};
    EXPECT_NE(std::string::npos, ErrorOf([&] { inferred.Encode(&broken); }).find("pixel values"));

    auto missing = inferred.Encode(nullptr);
    EXPECT_FLOAT_EQ(1.0f, missing.back());
}

// ============================================================================
// ObservationEncoder
// ============================================================================

std::vector<InputSpec> MixedInputs() {
    return {
        {"reading", InputType::NUMERIC},
        {"color", InputType::CATEGORICAL, 0, {"red", "green"}},
        {"user", InputType::CATEGORICAL},
        {"note", InputType::TEXT},
    };
}

TEST(ObservationEncoderTest, CreateEncoderPicksTheEncoderForEachType) {
    EXPECT_NE(nullptr, dynamic_cast<NumericEncoder*>(CreateEncoder({"a", InputType::NUMERIC}).get()));
    EXPECT_NE(nullptr, dynamic_cast<OneHotEncoder*>(
                           CreateEncoder({"a", InputType::CATEGORICAL, 0, {"x"}}).get()));
    EXPECT_NE(nullptr,
              dynamic_cast<HashedCategoryEncoder*>(CreateEncoder({"a", InputType::CATEGORICAL}).get()));
    EXPECT_NE(nullptr, dynamic_cast<TextEncoder*>(CreateEncoder({"a", InputType::TEXT}).get()));
    EXPECT_NE(nullptr, dynamic_cast<ImageEncoder*>(CreateEncoder({"a", InputType::IMAGE}).get()));
    EXPECT_NE(nullptr,
              dynamic_cast<TimeSeriesEncoder*>(CreateEncoder({"a", InputType::TIME_SERIES}).get()));
    EXPECT_THROW(CreateEncoder({"a", InputType::TEXT, 3}), std::invalid_argument);
}

TEST(ObservationEncoderTest, FusesChannelsInSpecOrder) {
    ObservationEncoder encoder(MixedInputs());
    EXPECT_FALSE(encoder.Ready());
    EXPECT_EQ(0u, encoder.OutputSize());

    // The numeric size is unknown until a value arrives
    EXPECT_FALSE(encoder.Encode({{"color", Text("red")}}).has_value());
    auto fused = encoder.Encode({{"reading", Vec({1, 2})}, {"note", Text("hi")}});
    ASSERT_TRUE(fused.has_value());
    EXPECT_TRUE(encoder.Ready());

    auto layout = encoder.Layout();
    ASSERT_EQ(4u, layout.size());
    EXPECT_EQ("reading", layout[0].name);
    EXPECT_EQ(0u, layout[0].offset);
    EXPECT_EQ(3u, layout[0].size);
    EXPECT_EQ(3u, layout[1].offset);
    EXPECT_EQ(4u, layout[1].size);
    EXPECT_EQ(7u, layout[2].offset);
    EXPECT_EQ(1025u, layout[2].size);
    EXPECT_EQ(8u, layout[2].embedding_size);
    EXPECT_EQ(1032u, layout[3].offset);
    EXPECT_EQ(1024u + 32u + 1u, layout[3].size);  // features, word ids, missing
    EXPECT_EQ(32u, layout[3].sequence_length);
    EXPECT_EQ(0u, layout[2].sequence_length);
    EXPECT_EQ(2089u, encoder.OutputSize());
    EXPECT_EQ(2089u, fused->size());

    // Missing channels set their indicators
    EXPECT_FLOAT_EQ(1.0f, (*fused)[3 + 3]);      // color missing
    EXPECT_FLOAT_EQ(1.0f, (*fused)[7 + 1024]);   // user missing
    EXPECT_FLOAT_EQ(0.0f, (*fused)[1032 + 1056]);  // note present
    EXPECT_FLOAT_EQ(1.0f, (*fused)[1032 + 1024]);  // its first word has id 1
}

TEST(ObservationEncoderTest, PeekMatchesEncodeWithoutLearning) {
    ObservationEncoder encoder(MixedInputs());
    encoder.Encode({{"reading", Num(1)}});
    Observation observation = {{"reading", Num(3)}, {"user", Text("u1")}};
    auto peeked = encoder.Peek(observation);
    EXPECT_EQ(peeked, encoder.Peek(observation));
    EXPECT_EQ(peeked, encoder.Encode(observation));
    const auto& numeric = dynamic_cast<const NumericEncoder&>(encoder.Channel("reading"));
    EXPECT_EQ(2u, numeric.Statistics()[0].count);
}

TEST(ObservationEncoderTest, RejectedObservationTeachesNothing) {
    ObservationEncoder encoder(MixedInputs());
    encoder.Encode({{"reading", Num(1)}});
    EXPECT_EQ("Unknown input channel 'speed'",
              ErrorOf([&] { encoder.Encode({{"reading", Num(2)}, {"speed", Num(1)}}); }));
    EXPECT_THROW(encoder.Encode({{"reading", Num(2)}, {"note", Num(1)}}), std::invalid_argument);
    const auto& numeric = dynamic_cast<const NumericEncoder&>(encoder.Channel("reading"));
    EXPECT_EQ(1u, numeric.Statistics()[0].count);
}

TEST(ObservationEncoderTest, ResetEpisodeReachesEveryChannel) {
    ObservationEncoder encoder({{"load", InputType::TIME_SERIES, 1}}, SmallWindow());
    for (double x : {1.0, 2.0, 3.0}) encoder.Encode({{"load", Num(x)}});
    encoder.ResetEpisode();
    auto fused = encoder.Peek({{"load", Num(4)}});
    ASSERT_TRUE(fused.has_value());
    EXPECT_FLOAT_EQ(0.25f, (*fused)[11]);
}

TEST(ObservationEncoderTest, RejectsInvalidInputsAndConfig) {
    std::string message = ErrorOf([] {
        ObservationEncoder encoder({{"a", InputType::NUMERIC}, {"a", InputType::TEXT, 2}});
    });
    EXPECT_NE(std::string::npos, message.find("duplicate input name 'a'"));
    EXPECT_NE(std::string::npos, message.find("size must be 0"));

    EncoderConfig config;
    config.text_buckets = 0;
    EXPECT_THROW(ObservationEncoder({{"a", InputType::TEXT}}, config), std::invalid_argument);
    EXPECT_THROW(ObservationEncoder(std::vector<InputSpec>{}), std::invalid_argument);
}

TEST(ObservationEncoderTest, AllInputTypesTogether) {
    std::vector<InputSpec> inputs = MixedInputs();
    inputs.push_back({"load", InputType::TIME_SERIES});
    inputs.push_back({"cam", InputType::IMAGE, 0, {}, 0, 0, 3});
    ObservationEncoder encoder(inputs);

    Observation observation = {
        {"reading", Num(1.5)},
        {"color", Text("green")},
        {"user", Num(17)},
        {"note", Text("all good")},
        {"load", Vec({0.1f, 0.2f})},
        {"cam", Image{2, 2, 3, std::vector<float>(12, 128.0f)}},
    };
    auto fused = encoder.Encode(observation);
    ASSERT_TRUE(fused.has_value());
    size_t expected = 2 + 4 + 1025 + (1025 + 32) + (2 * (2 * 16 + 3) + 2) + (2 * 2 * 3 + 1);
    EXPECT_EQ(expected, fused->size());
    auto layout = encoder.Layout();
    EXPECT_EQ(2u, layout.back().image_height);
    EXPECT_EQ(2u, layout.back().image_width);
    EXPECT_EQ(3u, layout.back().image_channels);
    EXPECT_EQ(64u, layout.back().embedding_size);
    EXPECT_EQ(0u, layout.front().image_height);
    for (float x : *fused) EXPECT_TRUE(std::isfinite(x));
}

}  // namespace
}  // namespace agent
}  // namespace dpan
