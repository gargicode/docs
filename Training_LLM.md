# Building an LLM: A Simple Example

Let's walk through how you'd build a tiny language model from scratch using a simple example!

## **The Training Data**

Imagine we have this tiny dataset (in reality, LLMs use trillions of words):

```
The cat sat on the mat.
The dog sat on the rug.
The bird flew over the tree.
The cat ate the fish.
```

---

## **Step 1: Tokenization (Breaking Text Into Pieces)**

### Python Code Example:
```python
text = "The cat sat on the mat."

# Simple tokenization - split by spaces
tokens = text.lower().split()
# Result: ['the', 'cat', 'sat', 'on', 'the', 'mat.']

# Build a vocabulary (all unique words)
vocabulary = {
    'the': 0,
    'cat': 1,
    'sat': 2,
    'on': 3,
    'mat': 4,
    'dog': 5,
    'rug': 6,
    'bird': 7,
    'flew': 8,
    'over': 9,
    'tree': 10,
    'ate': 11,
    'fish': 12
}
```

**What happens:** Every unique word gets assigned a number (ID). This is like creating a dictionary.

---

## **Step 2: Convert Words to Numbers**

```python
sentence = "The cat sat on the mat."
tokens = sentence.lower().split()

# Convert to IDs using our vocabulary
token_ids = [vocabulary[token.rstrip('.')] for token in tokens]
# Result: [0, 1, 2, 3, 0, 4]
```

**Database Structure:**
```
Table: vocabulary
+-------+----+
| word  | id |
+-------+----+
| the   | 0  |
| cat   | 1  |
| sat   | 2  |
| on    | 3  |
| mat   | 4  |
+-------+----+
```

---

## **Step 3: Create Embeddings (Vectors)**

This is where each word becomes a list of numbers. Initially, these are random!

```python
import numpy as np

# Let's use 4 dimensions for simplicity (real models use 1000+)
embedding_dim = 4
vocab_size = len(vocabulary)

# Initialize random embeddings
embeddings = np.random.randn(vocab_size, embedding_dim)

print(embeddings)
```

**Output Example:**
```
Word: "cat" (ID: 1)
Vector: [0.23, -0.45, 0.89, 0.12]

Word: "dog" (ID: 5)
Vector: [0.19, -0.52, 0.91, 0.08]

Word: "fish" (ID: 12)
Vector: [-0.82, 0.34, -0.15, 0.67]
```

**Database Structure:**
```
Table: embeddings
+----+--------+--------+--------+--------+
| id | dim_0  | dim_1  | dim_2  | dim_3  |
+----+--------+--------+--------+--------+
| 0  | 0.15   | -0.23  | 0.67   | 0.89   |
| 1  | 0.23   | -0.45  | 0.89   | 0.12   |
| 2  | -0.34  | 0.56   | -0.12  | 0.45   |
+----+--------+--------+--------+--------+
```

---

## **Step 4: Build the Neural Network**

The neural network is just a bunch of mathematical operations with **parameters** (weights).

```python
import numpy as np

class TinyLanguageModel:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        # These are our "parameters" - the knobs we'll tune
        self.embeddings = np.random.randn(vocab_size, embedding_dim)
        self.W1 = np.random.randn(embedding_dim, hidden_dim)  # First layer weights
        self.W2 = np.random.randn(hidden_dim, vocab_size)     # Output layer weights
        
    def forward(self, token_id):
        # Step 1: Get the embedding for this word
        x = self.embeddings[token_id]  # [4] dimensional vector
        
        # Step 2: First layer transformation
        hidden = np.dot(x, self.W1)  # Matrix multiplication
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Step 3: Output layer - predict next word
        output = np.dot(hidden, self.W2)
        
        # Step 4: Convert to probabilities
        probabilities = self.softmax(output)
        return probabilities
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# Initialize model
model = TinyLanguageModel(vocab_size=13, embedding_dim=4, hidden_dim=8)
```

**Database Structure for Parameters:**
```
Table: model_parameters
+---------------+-----------------+--------+
| layer_name    | parameter_name  | shape  | values (blob)     |
+---------------+-----------------+--------+-------------------+
| embeddings    | embedding_matrix| (13,4) | [0.23, -0.45, ...]|
| layer_1       | weights_W1      | (4,8)  | [0.15, 0.23, ...] |
| layer_2       | weights_W2      | (8,13) | [-0.12, 0.56, ...]|
+---------------+-----------------+--------+-------------------+
```

---

## **Step 5: Training - Learning Patterns**

Training is about adjusting those parameters to predict the next word correctly.

```python
# Training example: "The cat sat" → should predict "on"

# Input sequence: [the, cat, sat]
input_sequence = [0, 1, 2]  # Token IDs

# Target: "on" (ID: 3)
target = 3

# Forward pass - make a prediction
# After processing "the cat sat", what comes next?
prediction = model.forward(input_sequence[-1])  # Using last word

# prediction is a probability distribution:
# [0.02, 0.05, 0.01, 0.85, 0.03, 0.01, 0.01, 0.01, 0.00, 0.01]
#   ^     ^     ^     ^
#  'the' 'cat' 'sat' 'on' <- 85% probability! Good!

# Calculate loss (how wrong we were)
predicted_prob_for_target = prediction[target]
loss = -np.log(predicted_prob_for_target)  # Lower is better

# Backpropagation - adjust parameters to reduce loss
# (This is complex calculus - gradients flow backwards)
# Basically: "Make 'on' more likely next time"
```

**Training Loop:**
```python
def train(model, training_data, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        
        for sentence in training_data:
            token_ids = tokenize(sentence)
            
            # For each word, try to predict the next
            for i in range(len(token_ids) - 1):
                current_word = token_ids[i]
                next_word = token_ids[i + 1]
                
                # Predict
                prediction = model.forward(current_word)
                
                # Calculate loss
                loss = -np.log(prediction[next_word])
                total_loss += loss
                
                # Update parameters (gradient descent)
                model.update_parameters(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
```

---

## **Step 6: After Training - What Changed?**

After seeing millions of examples, the embeddings and weights have learned patterns:

**Before Training:**
```
"cat" vector: [0.23, -0.45, 0.89, 0.12]
"dog" vector: [-0.82, 0.34, -0.15, 0.67]
```

**After Training:**
```
"cat" vector: [0.45, -0.32, 0.91, 0.23]  <- Now similar to "dog"!
"dog" vector: [0.47, -0.30, 0.89, 0.25]  <- Close in vector space
```

The model learned that "cat" and "dog" are similar because they appear in similar contexts!

---

## **Step 7: Using the Trained Model**

```python
# Generate text: "The cat"
input_ids = [0, 1]  # "the cat"

# What comes next?
prediction = model.forward(input_ids[-1])

# Get the most likely next word
next_word_id = np.argmax(prediction)
next_word = inverse_vocabulary[next_word_id]

print(f"The cat {next_word}")  
# Output: "The cat sat" (because it learned this pattern!)
```

---

## **Complete System Architecture**

```
┌─────────────────────────────────────────┐
│         TRAINING PHASE                  │
├─────────────────────────────────────────┤
│                                         │
│  Raw Text Files                         │
│  ├─ document1.txt                       │
│  ├─ document2.txt                       │
│  └─ document3.txt                       │
│         ↓                               │
│  Python Preprocessing                   │
│  ├─ Tokenization                        │
│  ├─ Build Vocabulary                    │
│  └─ Convert to IDs                      │
│         ↓                               │
│  Database/Storage                       │
│  ├─ vocabulary.db                       │
│  └─ training_sequences.db               │
│         ↓                               │
│  Neural Network Training                │
│  ├─ Initialize random weights           │
│  ├─ Forward pass (predict)              │
│  ├─ Calculate loss                      │
│  ├─ Backpropagate                       │
│  └─ Update weights                      │
│         ↓                               │
│  Save Trained Model                     │
│  ├─ embeddings.npy                      │
│  ├─ layer1_weights.npy                  │
│  ├─ layer2_weights.npy                  │
│  └─ model_config.json                   │
│                                         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         INFERENCE PHASE                 │
├─────────────────────────────────────────┤
│                                         │
│  User Input: "What is the capital"      │
│         ↓                               │
│  Load Model from Storage                │
│  ├─ Load embeddings                     │
│  ├─ Load weights                        │
│  └─ Load vocabulary                     │
│         ↓                               │
│  Tokenize Input                         │
│  ["what", "is", "the", "capital"]       │
│         ↓                               │
│  Convert to Embeddings                  │
│  [[0.23,...], [0.45,...], ...]          │
│         ↓                               │
│  Pass Through Neural Network            │
│  Layer by layer transformation          │
│         ↓                               │
│  Generate Output Token by Token         │
│  "of" → "Canada" → "is" → "Ottawa"      │
│         ↓                               │
│  Return Response                        │
│  "The capital of Canada is Ottawa."     │
│                                         │
└─────────────────────────────────────────┘
```

---

## **Real-World Scale**

### Our Tiny Example:
- **Vocabulary:** 13 words
- **Embedding dimension:** 4
- **Training data:** 4 sentences
- **Parameters:** ~200 numbers

### Real LLM (like GPT-3):
- **Vocabulary:** 50,000+ tokens
- **Embedding dimension:** 12,288
- **Training data:** 500+ billion words
- **Parameters:** 175 billion numbers
- **Storage:** ~350 GB just for the weights!
- **Training time:** Months on thousands of GPUs

---

## **Key Storage Components**

```
project/
├── data/
│   ├── raw_text/              # Original training documents
│   │   ├── books.txt
│   │   ├── articles.txt
│   │   └── websites.txt
│   └── processed/
│       ├── vocabulary.json     # Word → ID mapping
│       └── tokenized.bin       # Preprocessed token IDs
│
├── model/
│   ├── embeddings.npy          # Word vectors
│   ├── layer_1_weights.npy     # Neural network weights
│   ├── layer_2_weights.npy
│   ├── layer_N_weights.npy
│   └── config.json             # Model architecture info
│
├── checkpoints/               # Saved during training
│   ├── epoch_100.ckpt
│   ├── epoch_200.ckpt
│   └── best_model.ckpt
│
└── train.py                   # Training script
└── inference.py               # Using the model
```

---

## **Simple Python Implementation**

Here's a complete minimal example you could actually run:

```python
import numpy as np

# Training data
texts = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ate the fish"
]

# Build vocabulary
vocab = {}
for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {vocab}")

# Create training pairs (current_word → next_word)
training_pairs = []
for text in texts:
    words = text.split()
    for i in range(len(words) - 1):
        current = vocab[words[i]]
        next_word = vocab[words[i + 1]]
        training_pairs.append((current, next_word))

print(f"Training pairs: {len(training_pairs)}")

# Initialize simple model
embedding_dim = 4
embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
W_output = np.random.randn(embedding_dim, vocab_size) * 0.1

# Training
learning_rate = 0.1
for epoch in range(1000):
    total_loss = 0
    
    for current, target in training_pairs:
        # Forward pass
        x = embeddings[current]
        logits = x @ W_output
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Loss
        loss = -np.log(probs[target] + 1e-10)
        total_loss += loss
        
        # Backward pass (simplified)
        grad_logits = probs.copy()
        grad_logits[target] -= 1
        
        # Update weights
        grad_W = np.outer(x, grad_logits)
        W_output -= learning_rate * grad_W
        
        grad_x = grad_logits @ W_output.T
        embeddings[current] -= learning_rate * grad_x
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Test
test_word = "cat"
test_id = vocab[test_word]
x = embeddings[test_id]
logits = x @ W_output
probs = np.exp(logits) / np.sum(np.exp(logits))

# Get most likely next word
next_id = np.argmax(probs)
inv_vocab = {v: k for k, v in vocab.items()}
next_word = inv_vocab[next_id]

print(f"\nAfter 'cat', most likely next word: '{next_word}'")
print(f"Probabilities: {probs}")
```

---

## **The Big Picture**

1. **Collect text** → millions of documents
2. **Tokenize** → break into pieces
3. **Create vocabulary** → assign IDs
4. **Initialize embeddings** → random vectors
5. **Build neural network** → layers of math operations
6. **Train** → adjust parameters to predict next word
7. **Save model** → store all the learned weights
8. **Use model** → load weights and generate text

The "intelligence" emerges from billions of parameters learning patterns from vast amounts of text data!
