# DPAN CLI Quick Start

## What is This?

An interactive command-line interface that lets you directly communicate with DPAN (Dynamic Pattern Association Network). DPAN learns from everything you say and builds associations to predict and generate responses.

## Build & Run

```bash
# Build the CLI
cd /home/user/nn/build
make dpan_cli

# Run it
cd /home/user/nn
./build/src/cli/dpan_cli
```

## 30-Second Tutorial

### 1. Start talking to DPAN

```
dpan> Hello
→ [Learning...]

dpan> Hi there
→ [Learning...]

dpan> How are you?
→ [Learning...]
```

At first, DPAN is learning. Keep going!

### 2. Teach it a conversation pattern

```
dpan> Hello
→ [Learning...]

dpan> Hi there!
→ [Learning...]

dpan> Hello
→ Hi there! [confidence: 0.75]
```

See? It learned that "Hello" → "Hi there!"

### 3. Batch learn from a file

```
dpan> /learn examples/training_data/basic_conversation.txt
✓ Learned from 24 lines in 156 ms
  Patterns created: 18
```

### 4. See what it learned

```
dpan> /stats
...
Patterns: 18
Associations: 34
Average strength: 0.68

dpan> /predict Hello
Predictions for "Hello":
  1. "Hi there!" [0.850]
  2. "How are you?" [0.720]

dpan> Hello
→ Hi there! [confidence: 0.85]
```

### 5. Enable active learning

```
dpan> /active
Active learning mode: ON

dpan> something confusing
[ACTIVE LEARNING] I'm not confident about that. Can you tell me more or rephrase?
```

DPAN now asks questions when uncertain!

## Essential Commands

- `<text>` - Talk to DPAN (it learns from everything)
- `/learn <file>` - Batch learn from a text file
- `/stats` - Show learning statistics
- `/patterns` - List learned patterns
- `/predict <text>` - See what DPAN predicts will follow
- `/active` - Toggle active learning mode
- `/help` - Show all commands
- `exit` - Quit (auto-saves)

## Goal

Train DPAN with enough conversations that it can:
- Recognize patterns in your input
- Predict appropriate responses
- Generate realistic replies
- Build on previous knowledge

## Training Tips

1. **Start with a training file**:
   ```
   dpan> /learn examples/training_data/basic_conversation.txt
   ```

2. **Check progress**:
   ```
   dpan> /stats
   ```

3. **Test predictions**:
   ```
   dpan> /predict Hello
   ```

4. **Have conversations**:
   ```
   dpan> Hello
   → Hi there! [confidence: 0.85]
   ```

5. **Monitor associations**:
   ```
   dpan> /associations
   Strongest associations:
     1. "Hello" → "Hi there!" [0.950]
   ```

Strong associations (> 0.8) mean good learning!

## Success Metrics

- **Patterns > 100**: Good vocabulary
- **Associations > 200**: Rich connection network
- **Average strength > 0.65**: Strong learned connections
- **Responses make sense**: The real test!

## Next Steps

See the full guide: [docs/DPAN_CLI_Guide.md](docs/DPAN_CLI_Guide.md)

---

**Now go teach DPAN to communicate! Every conversation makes it smarter.**
