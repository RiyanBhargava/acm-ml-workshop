# DAY 3-DEEP LEARNING & NLP WORKSHOP

## 🧠 Introduction to Deep Learning

**Deep Learning (DL)** is a subset of Machine Learning that uses **neural networks** with many layers to automatically learn complex patterns from data.

### 🧩 Analogy

| Human Brain | Deep Learning |
| --- | --- |
| Neurons in the brain | Artificial Neurons in a network |
| Learn from experience | Learn from data |
| Recognizes faces/voices | Recognizes patterns in data |

### 🔍 Example

- Traditional rule: “If price < 10, buy.”
- Deep Learning: *Learns* to buy/sell automatically from thousands of examples.

---

## ⚖️ Why Deep Learning over Traditional ML

| Aspect | Traditional Machine Learning | Deep Learning |
| --- | --- | --- |
| **Feature Engineering** | Manual — you select features | Automatic — model learns best features |
| **Data Requirement** | Works on small datasets | Requires large data, but performs better |
| **Performance** | Plateaus with complex data | Improves with more data & compute |
| **Examples** | Linear Regression, SVM, Decision Trees | CNN, LSTM, Transformers |
| **Applications** | Simple classification/regression | Image, speech, NLP, autonomous systems |

### 💡 Example

- ML: You manually count words and predict sentiment.
- DL: The model *understands meaning* (e.g., “not bad” = positive) automatically.

---

## 💬 What is NLP (Natural Language Processing)?

**NLP** helps computers understand and generate human language.

### 💡 Real-Life Examples

- ChatGPT (conversation)
- Google Translate (language conversion)
- Spam filters (email classification)
- Siri/Alexa (speech recognition)

---

## 🧹 Text Preprocessing Steps

Before giving text to a deep learning model, we clean and convert it into numbers.

| Step | What It Does | Example |
| --- | --- | --- |
| **Lowercasing** | Standardize text | “Harry” → “harry” |
| **Tokenization** | Split into words | “harry potter went” → [“harry”, “potter”, “went”] |
| **Stopword Removal** | Remove unimportant words | Remove “a”, “the”, “to”, etc. |
| **Stemming/Lemmatization** | Reduce to root form | “running” → “run” |
| **Vectorization** | Convert to numbers | “harry” → `[0.23, 0.89, -0.12, …]` |

---

## 🧩 Understanding Word Embeddings

**Word Embeddings** represent words as numerical vectors such that similar words are close in space.

### ✨ Example:

- “king” – “man” + “woman” ≈ “queen”
- “cat” and “dog” will be close in embedding space.

### 📊 Visualization

```markdown
          king
            \
             \    queen
              \  /
          man   woman

     cat -------- dog

```

> Embeddings capture semantic meaning, not just spelling.
> 

Popular embeddings:

- **Word2Vec**
- **GloVe**
- **FastText**

---

## ⚙️ Deep Learning Architectures for NLP

---

### 🌀 (A) LSTMs — Long Short-Term Memory Networks

LSTMs are good at learning **sequences** (like sentences or time series).

They solve a key problem — remembering **long-term context**.

### 🧩 Example

Sentence:

> “Harry looked at Ron and said he was __.”
> 

To predict the blank word (“angry”), the model must **remember earlier words** — that’s what LSTMs do.

### 🧠 Concept Diagram

```markdown
Input → [LSTM cell → LSTM cell → LSTM cell] → Output
            ↑ remembers previous words ↑

```

### 🧮 Applications

- Next word prediction
- Sentiment analysis
- Chatbots

---

### ⚡ (B) Transformers

Transformers are the **modern standard** for NLP.

Instead of reading one word at a time, they read the **whole sentence at once** and use **attention** to find relationships between words.

### 🧩 Example

Sentence:

> “The ball hit the boy because he was careless.”
> 

The model learns that **“he”** refers to **“boy”**, not “ball”.

### 🔍 Attention Mechanism

The model gives “attention scores” — how much each word relates to another.

| Word | Attends to | Importance |
| --- | --- | --- |
| he | boy | ⭐⭐⭐⭐ |
| ball | hit | ⭐⭐ |
| because | careless | ⭐⭐⭐ |

---

## 🤖 Modern Models: BERT & GPT

| Model | Direction | Main Purpose | Example Use |
| --- | --- | --- | --- |
| **BERT** | Bidirectional (reads both directions) | Understand text | Sentiment, Q&A |
| **GPT** | Unidirectional (left→right) | Generate text | Chatbots, writing |

---

## 🧪 Hands-On Project: Predicting Next Word (Harry Potter Corpus)

### 🧰 Setup

```bash
pip install transformers torch

```

### 🧩 Python Code

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Harry looked at Ron and said"
result = generator(prompt, max_length=25, num_return_sequences=1, temperature=0.9)

print(result[0]['generated_text'])

```

### 🧙‍♂️ Example Output

> Harry looked at Ron and said quietly, “We can’t let anyone know about this.” The wind howled through the castle halls...
> 

### 🔍 Try Custom Prompts

```python
prompts = [
    "Voldemort raised his wand and",
    "Hermione opened the book of spells and",
    "Hogwarts castle was silent until"
]

for p in prompts:
    print(generator(p, max_length=30, num_return_sequences=1)[0]['generated_text'])

```

### ⚙️ How It Works

1. Text → tokens → embeddings
2. Model predicts the **next likely word**
3. Adds it to text and repeats
4. Generates creative continuations

---

##