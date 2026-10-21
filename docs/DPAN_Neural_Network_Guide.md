# DPAN Neural Network Module

`dpan::nn` (headers under `nn/`) provides dense feed-forward neural networks
trained with backpropagation, and three ways to use them with DPAN:

| Integration | Class | Plugs into |
|---|---|---|
| Learned feature extraction | `NeuralEncoder`, `Autoencoder` | Encode features before `PatternEngine::CreatePattern` |
| Learned similarity | `EmbeddingSimilarity` | `PatternEngine::Config::similarity_metric_instance`, or any `SimilarityMetric` user |
| Learned prediction scoring | `NeuralPredictionScorer` | `AssociationLearningSystem::SetPredictionScorer` |

Link `dpan::nn` (or `dpan::dpan`). A complete program is in
`examples/nn/neural_network_example.cpp`.

## Core

### Data

`nn::Matrix` is a row-major float matrix. A batch is one sample per row:
`Matrix::FromRows({{0, 1}, {1, 0}})`, `Matrix::RowVector({...})`,
`OneHot(labels, num_classes)`, `FeatureMatrix(std::vector<FeatureVector>)`.
Shape mismatches throw `std::invalid_argument`.

### Layers

| Layer | Notes |
|---|---|
| `Dense(in, out, init, seed)` | `y = xW + b`; `init`: `XAVIER` (tanh/sigmoid), `HE` (ReLU), `ZEROS` |
| `ReLU`, `LeakyReLU(alpha)`, `Sigmoid`, `Tanh` | Element-wise |
| `Softmax` | Row-wise; omit it when training with `SoftmaxCrossEntropy` |
| `Dropout(rate, seed)` | Inverted dropout; identity at inference |

### Losses and optimizers

- `MeanSquaredError` — regression and autoencoders
- `BinaryCrossEntropy` — probabilities from a final `Sigmoid`
- `SoftmaxCrossEntropy` — multi-class, applied to logits (no final `Softmax`);
  use `Softmax::Compute(logits)` for probabilities
- `SGD(learning_rate, momentum, weight_decay)`, `Adam(learning_rate, beta1, beta2, epsilon, weight_decay)`
- Metrics: `Accuracy` (arg-max), `BinaryAccuracy` (threshold 0.5)

### Model

```cpp
using namespace dpan::nn;

Sequential model;
model.Emplace<Dense>(4, 16, Initialization::HE).Emplace<ReLU>()
     .Emplace<Dropout>(0.2f)
     .Emplace<Dense>(16, 3);

Adam optimizer(0.01f);
Sequential::FitOptions options;
options.epochs = 100;
options.batch_size = 32;
options.validation_split = 0.2f;          // last 20% of rows
options.early_stopping_patience = 10;     // optional
auto history = model.Fit(x, one_hot_labels, SoftmaxCrossEntropy(), optimizer, options);

Matrix logits = model.Predict(x_test);
float accuracy = Accuracy(logits, test_labels);

model.Save("classifier.dpnn");
Sequential restored = Sequential::Load("classifier.dpnn");
```

- `Fit` shuffles each epoch (seeded), reports per-epoch losses in
  `History`, and calls `FitOptions::on_epoch_end` (return `false` to stop).
- `TrainBatch` performs one optimization step for custom training loops;
  `Forward` / `Backward` / `ZeroGradients` / `Parameters` expose the steps.
- `Slice(begin, end)` copies a range of layers (e.g. an encoder half).
- `Predict` and `Evaluate` are `const` and may run concurrently; training
  methods need external synchronization.
- An optimizer keeps per-parameter state; use one optimizer per model, or
  call `Reset()` before reusing it.

## Integration with DPAN

### Learned feature extraction

`NeuralEncoder` wraps a trained network as a feature transform.
`Autoencoder` trains one without labels by reconstructing its input:

```cpp
Autoencoder::Config config;
config.hidden_sizes = {32};
config.embedding_size = 8;
config.epochs = 200;
Autoencoder autoencoder(input_dimension, config);
autoencoder.Train(feature_vectors);              // std::vector<FeatureVector> or Matrix

NeuralEncoder encoder = autoencoder.Encoder();
PatternID id = engine.CreatePattern(encoder.Encode(raw_pattern_data));  // stores the embedding
```

`Encode(PatternData)` keeps the pattern's modality. Encoders can be saved
with `Save` and loaded with `NeuralEncoder::Load`.

### Learned similarity

`EmbeddingSimilarity` compares raw feature vectors in embedding space
(`max(0, cosine)` of the encodings). Use it to search patterns that were
stored unencoded:

```cpp
PatternEngine::Config engine_config;
engine_config.similarity_metric_instance =
    std::make_shared<EmbeddingSimilarity>(autoencoder.Encoder());
PatternEngine engine(engine_config);
auto similar = engine.FindSimilarPatterns(query, 10);
```

If patterns are stored as embeddings (previous section), the built-in
`"cosine"` metric compares them directly.

### Learned prediction scoring

`AssociationLearningSystem` ranks predictions by association strength. A
`PredictionScorer` adds a learned estimate:

    score = (1 - weight) * strength + weight * scorer.Score(source, candidate, strength, context)

`NeuralPredictionScorer` learns that estimate from outcomes, using the
features of both patterns (optionally encoded) and the association strength:

```cpp
auto scorer = std::make_shared<NeuralPredictionScorer>(&engine.GetDatabase(), feature_dimension);

// Whenever the true next pattern is known: label each outgoing candidate
for (const auto* edge : system.GetAssociationMatrix().GetOutgoingAssociations(current)) {
    scorer->AddExample(current, edge->GetTarget(), edge->GetStrength(),
                       edge->GetTarget() == actual_next);
}

scorer->Train();                                  // retrain periodically
system.SetPredictionScorer(scorer, 0.5f);
```

- Until trained, or when a pattern's features are unavailable, `Score`
  returns the association strength, so ranking is unchanged.
- `Train()` builds a new model and swaps it in; `Score()` may run
  concurrently.
- `SaveModel` / `LoadModel` persist the trained network.
- Any other `PredictionScorer` implementation (rules, external models) can
  be set the same way.

## Limitations

- CPU only, single-threaded matrix operations (no BLAS or GPU).
- Dense feed-forward networks only (no convolutional or recurrent layers).
- `Fit` holds the whole dataset in memory.
