# DPAN Interactive CLI Guide

## Overview

The DPAN CLI provides a direct, interactive interface for communicating with and training the Dynamic Pattern Association Network. Unlike the library-only approach, the CLI allows you to:

- **Converse directly** with DPAN using natural text
- **Actively learn** - DPAN asks for clarification when uncertain
- **Batch learn** from text files
- **Inspect patterns and associations** in real-time
- **Persist sessions** across multiple interactions
- **Test realistic communications** to see if DPAN can learn to respond naturally

## Key Philosophy

The CLI enables DPAN to **"want" to learn** by:

1. **Active Learning Mode**: DPAN requests more information when uncertain
2. **Continuous Learning**: Every interaction improves the network
3. **Association Building**: DPAN learns relationships between inputs
4. **Predictive Responses**: DPAN attempts to predict and generate appropriate responses
5. **Session Persistence**: Knowledge accumulates across sessions

## Installation & Building

### Build the CLI

```bash
cd /home/user/nn/build
cmake ..
make dpan_cli
```

The executable will be located at: `/home/user/nn/build/src/cli/dpan_cli`

### Run the CLI

```bash
cd /home/user/nn
./build/src/cli/dpan_cli
```

## Getting Started

### First Launch

When you first run DPAN CLI, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   DPAN Interactive Learning Interface                       ║
║   Dynamic Pattern Association Network                       ║
║                                                              ║
║   A neural network that learns and grows from interaction   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands, or just start talking!
The system will learn from everything you say.

No previous session found. Starting fresh.

dpan>
```

### Basic Interaction

Just type naturally - DPAN will learn from everything you say:

```
dpan> Hello
[Processing: "Hello"]
[Created 1 new pattern(s)]
→ [Learning... I don't have enough context yet to respond.]

dpan> How are you?
[Processing: "How are you?"]
[Created 1 new pattern(s)]
→ [Learning... I don't have enough context yet to respond.]

dpan> I am fine
[Processing: "I am fine"]
[Created 1 new pattern(s)]
→ [Learning... I don't have enough context yet to respond.]
```

As DPAN learns more patterns, it will start making predictions and responses based on associations.

## Available Commands

All commands start with `/` - regular text is treated as conversational input.

### Conversation Commands

#### `/predict <text>`
Show what DPAN predicts will follow the given text.

```
dpan> /predict Hello
Predictions for "Hello":
  1. "How are you?" [0.750]
  2. "Nice to meet you" [0.620]
  3. "Good morning" [0.580]
```

### Learning Commands

#### `/learn <filepath>`
Batch learn from a text file (one line = one input pattern).

```
dpan> /learn conversation_data.txt
Learning from file: conversation_data.txt
  Processed 100 lines...
  Processed 200 lines...
✓ Learned from 250 lines in 1247 ms
  Patterns created: 187
```

**Example conversation file format:**
```txt
Hello
How are you?
I am fine
What is your name?
My name is Alice
Nice to meet you
Goodbye
```

#### `/active`
Toggle active learning mode. When enabled, DPAN will ask for clarification when uncertain.

```
dpan> /active
Active learning mode: ON
DPAN will now ask for clarification when uncertain.

dpan> something confusing
[Processing: "something confusing"]
[Created 1 new pattern(s)]

[ACTIVE LEARNING] I'm not confident about that. Can you tell me more or rephrase?
```

### Information Commands

#### `/stats`
Display comprehensive learning statistics.

```
dpan> /stats

╔══════════════════════════════════════════╗
║         DPAN Learning Statistics         ║
╚══════════════════════════════════════════╝

Session:
  Inputs processed: 150
  Patterns learned: 98
  Conversation length: 150
  Vocabulary size: 98 unique inputs

Patterns:
  Total patterns: 98
  Atomic: 98
  Composite: 0
  Average confidence: 0.75

Associations:
  Total associations: 234
  Average strength: 0.62
  Strongest association: 0.95

Storage:
  Database: dpan_session.db
  Size: 45 KB
  Active learning: ON
```

#### `/patterns`
List all learned patterns with their associations.

```
dpan> /patterns

Learned Patterns (text mappings):
================================

   1. "Hello"
      Pattern ID: 12345678901234567890
      Confidence: 0.90
      Leads to: "How are you?" "Hi there" "Good morning"

   2. "How are you?"
      Pattern ID: 12345678901234567891
      Confidence: 0.85
      Leads to: "I am fine" "Good, thanks" "Great!"

... (18 more patterns)

Use /verbose to see all patterns
```

#### `/associations`
Show the association graph and strongest connections.

```
dpan> /associations

Association Graph:
==================
Total associations: 234
Average strength: 0.62

Strongest associations:
  1. "Hello" → "How are you?" [0.950]
  2. "How are you?" → "I am fine" [0.920]
  3. "What is your name?" → "My name is" [0.890]
  4. "Good morning" → "Good morning" [0.870]
  5. "Thank you" → "You're welcome" [0.850]
```

#### `/verbose`
Toggle verbose output showing internal processing details.

```
dpan> /verbose
Verbose mode: ON

dpan> Hello
[Processing: "Hello"]
[Activated existing pattern]
→ How are you? [confidence: 0.85]
   Other possibilities: "Hi" "Hey there"
```

### Session Management

#### `/save`
Manually save the current session (also auto-saves on exit).

```
dpan> /save
Saving session to dpan_session.db...
✓ Session saved successfully
✓ Saved 98 text mappings
```

#### `/load`
Reload the previous session (auto-loads on startup if exists).

```
dpan> /load
Loading previous session...
✓ Loaded associations
✓ Loaded 98 text mappings
Session loaded: 98 patterns
```

#### `/reset`
Reset and clear all learned data (requires confirmation).

```
dpan> /reset
Are you sure you want to reset? This will erase all learning. (y/N): y
✓ Session reset. Starting fresh.
```

### Utility Commands

#### `/clear`
Clear the terminal screen.

#### `/help`
Show command reference.

#### `exit` or `quit`
Exit the CLI (auto-saves session).

```
dpan> exit

Shutting down...
Saving session to dpan_session.db...
✓ Session saved successfully
✓ Saved 98 text mappings

Session Summary:
  Inputs processed: 150
  Patterns learned: 98
  Conversation length: 150

Thank you for teaching me! Goodbye.
```

## Training DPAN for Realistic Communication

### Strategy 1: Conversation Files

Create text files with realistic conversations:

**conversation_training.txt:**
```
Hello
Hi there!
How are you?
I'm doing great, thanks for asking!
What did you do today?
I worked on some projects.
That sounds interesting.
Yes, it was productive.
Do you like coffee?
Yes, I love coffee!
Me too!
```

Load it:
```
dpan> /learn conversation_training.txt
```

### Strategy 2: Interactive Training

Have actual conversations, teaching DPAN as you go:

```
dpan> Hello
→ [Learning...]

dpan> Hi there!
→ [Learning...]

dpan> How are you?
→ [Learning...]

dpan> I am fine, thank you
→ [Learning...]

# After some training...

dpan> Hello
→ Hi there! [confidence: 0.75]

dpan> How are you?
→ I am fine, thank you [confidence: 0.82]
```

### Strategy 3: Enable Active Learning

Let DPAN ask for clarification:

```
dpan> /active
Active learning mode: ON

dpan> The quantum entanglement manifold
[ACTIVE LEARNING] I'm not confident about that. Can you tell me more or rephrase?

dpan> It's a physics concept about particles
[Processing...]
→ [Learned new pattern]
```

### Strategy 4: Predictive Validation

Test predictions to see what DPAN has learned:

```
dpan> /predict Hello
Predictions for "Hello":
  1. "Hi there!" [0.850]
  2. "How are you?" [0.720]
  3. "Good morning" [0.650]
```

### Strategy 5: Monitor Progress

Regularly check statistics to see learning progress:

```
dpan> /stats
...
Associations:
  Total associations: 1,234
  Average strength: 0.68
```

Higher association counts and strengths indicate better learning.

## Advanced Usage

### Persistent Learning Sessions

DPAN automatically saves to `dpan_session.db` in the current directory. This includes:
- All learned patterns
- Association graph
- Text-to-pattern mappings

To use different sessions:

```bash
# Session 1: General conversation
cd /home/user/training/general
dpan_cli  # Creates dpan_session.db here

# Session 2: Technical terms
cd /home/user/training/technical
dpan_cli  # Creates separate dpan_session.db here
```

### Batch Learning from Large Corpora

For large text datasets:

```
dpan> /learn large_corpus_part1.txt
✓ Learned from 10,000 lines in 8,234 ms
  Patterns created: 4,567

dpan> /stats
...
Patterns: 4,567
Associations: 12,345
```

### Testing Communication Quality

1. **Train with diverse data**:
```
dpan> /learn greetings.txt
dpan> /learn questions.txt
dpan> /learn responses.txt
```

2. **Test predictions**:
```
dpan> Hello
→ Hi there! [confidence: 0.85]

dpan> How are you?
→ I'm doing great! [confidence: 0.82]
```

3. **Check association quality**:
```
dpan> /associations
Strongest associations:
  1. "Hello" → "Hi there!" [0.950]
  2. "How are you?" → "I'm fine" [0.920]
  ...
```

Strong associations (> 0.8) indicate good learning.

## Understanding DPAN's Learning

### How DPAN Learns

1. **Pattern Creation**: Each unique input creates a pattern
2. **Co-occurrence Tracking**: Patterns that appear together in time are tracked
3. **Association Formation**: Frequent co-occurrences form associations
4. **Reinforcement**: Correct predictions strengthen associations
5. **Competition**: Associations compete, strong ones get stronger
6. **Decay**: Unused associations weaken over time
7. **Pruning**: Very weak associations are removed

### Reading the Output

```
dpan> Hello
→ Hi there! [confidence: 0.85]
   Other possibilities: "Hey" "Yo"
```

- `→` indicates DPAN's response
- `[confidence: 0.85]` shows association strength (0.0 - 1.0)
- Higher confidence = stronger learned connection
- Other possibilities show alternative predictions

### When DPAN Says "Learning..."

```
→ [Learning... I don't have enough context yet to respond.]
```

This means:
- No associations exist for this pattern yet
- Need more training data
- DPAN has seen this pattern but not what follows it

Continue providing examples to build associations.

### When DPAN Asks Questions (Active Mode)

```
[ACTIVE LEARNING] I'm not confident about that. Can you tell me more or rephrase?
```

This means:
- Confidence score < 0.6
- DPAN recognizes uncertainty
- Requesting clarification to improve learning

## Performance Tips

### For Faster Learning

1. **Batch load** large datasets with `/learn` instead of typing
2. **Higher similarity thresholds** match patterns more easily
3. **Lower formation thresholds** create associations faster

### For Better Quality

1. **Diverse training data** - varied conversations
2. **Active learning** - let DPAN ask questions
3. **Reinforcement** - repeat correct patterns
4. **Monitor associations** - ensure strong connections form

### Memory Management

- DPAN uses SQLite with WAL mode
- Database grows with learning
- Automatic pruning removes weak associations
- Monitor with `/stats` to see storage size

## Troubleshooting

### "No patterns matched or created"

**Cause**: Input too similar to existing but below match threshold

**Solution**:
- Try rephrasing
- Check `/patterns` to see what's learned
- Continue training

### "I don't have enough context yet"

**Cause**: No associations formed for this pattern

**Solution**:
- Provide more examples of what follows
- Use `/learn` to batch train
- Enable `/active` mode

### Responses seem random

**Cause**: Weak associations or insufficient training

**Solution**:
- Check `/associations` - look for low average strength
- Provide more training data
- Ensure consistent patterns in training

### Database errors

**Cause**: Corrupted session file

**Solution**:
```
dpan> /reset
```

Or manually delete:
```bash
rm dpan_session.db*
```

## Examples

### Example 1: Training a Greeting Bot

```
dpan> /learn greetings.txt
dpan> /verbose
Verbose mode: ON

dpan> Hello
→ Hi there! [confidence: 0.85]

dpan> Good morning
→ Good morning! How are you? [confidence: 0.82]

dpan> /stats
...
Patterns: 15
Associations: 28
```

### Example 2: Technical Q&A Learning

```
dpan> What is DPAN?
→ [Learning...]

dpan> DPAN is a Dynamic Pattern Association Network
→ [Learning...]

dpan> What is DPAN?
→ DPAN is a Dynamic Pattern Association Network [confidence: 0.95]
```

### Example 3: Progressive Learning Check

```
# Day 1
dpan> /stats
Patterns: 50
Associations: 120

# Day 2 (after more training)
dpan> /stats
Patterns: 250
Associations: 580

# Day 3
dpan> /stats
Patterns: 500
Associations: 1,240
Average strength: 0.72  # Getting stronger!
```

## Session Files

DPAN creates these files:

- `dpan_session.db` - Main SQLite database with patterns
- `dpan_session.db.associations` - Association graph
- `dpan_session.db.mappings` - Text-to-pattern mappings
- `dpan_session.db-wal` - Write-Ahead Log (SQLite)
- `dpan_session.db-shm` - Shared memory (SQLite)

All files are needed for full session restoration.

## Next Steps

1. **Start small**: Train with simple conversations
2. **Monitor progress**: Use `/stats` and `/associations`
3. **Scale up**: Add more diverse training data
4. **Test quality**: Use `/predict` to validate learning
5. **Enable active learning**: Let DPAN guide its own education
6. **Iterate**: Continuous training improves responses

## Goal: Realistic Communication

The ultimate goal is for DPAN to learn enough patterns and associations that it can:

- Recognize context from input
- Predict appropriate responses
- Generate natural-seeming replies
- Build on previous conversations
- Adapt to new information

Monitor these metrics for success:
- **Association count > 1,000**: Rich connection network
- **Average strength > 0.65**: Strong learned connections
- **Vocabulary size > 200**: Diverse pattern knowledge
- **Prediction accuracy**: Responses make sense in context

Keep training, and DPAN will continue to grow and improve!

## Technical Details

- **Pattern Extraction**: Text → byte vector → feature extraction
- **Association Learning**: Hebbian learning + competitive dynamics
- **Storage**: SQLite persistent backend with WAL mode
- **Thread Safety**: All operations are thread-safe
- **Memory**: Automatic pruning and maintenance

For implementation details, see the [library documentation](DPAN_Design_Document.md).

---

**Happy learning! Let DPAN grow with every conversation.**
