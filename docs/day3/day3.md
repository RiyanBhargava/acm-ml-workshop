# DAY 3-DEEP LEARNING & NLP WORKSHOP

## 🔹 1. Introduction to Deep Learning

Deep Learning (DL) is a subset of Machine Learning inspired by the structure of the human brain — the **Artificial Neural Network (ANN)**.

It enables models to automatically extract features and patterns from raw data such as images, audio, or text without manual feature engineering.

### Key Features:

- Learns from large amounts of data
- Uses multiple layers (hence *deep*) to model complex relationships
- Common frameworks: TensorFlow, PyTorch, Keras

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

## 🔹 2. What is NLP?

**Natural Language Processing (NLP)** is a field of AI that allows computers to understand, interpret, and generate human language.

It bridges **human communication** and **machine understanding**.

### Real-life Applications:

- Chatbots (e.g., ChatGPT, Alexa)
- Machine translation (e.g., Google Translate)
- Sentiment analysis (e.g., analyzing reviews)
- Text summarization, auto-completion, spam detection

---

## 🔹 3. NLP Workflow Overview

1. **Text Preprocessing:** Cleaning and preparing raw text
2. **Tokenization:** Splitting text into words or subwords
3. **Vectorization:** Converting text into numerical form
4. **Modeling:** Using DL models like LSTM, GRU, or Transformer
5. **Evaluation & Inference:** Predicting and generating new text

---

## 🔹 4. Basics of NLP Concepts

### 🧩 4.1 Text Preprocessing

Before feeding data to models, we clean it:

- **Lowercasing** text
- **Removing punctuation & stopwords**
- **Tokenization** (splitting into words)
- **Stemming/Lemmatization** (reducing words to base form)

**Example:**

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk, re
nltk.download('punkt'); nltk.download('stopwords')

text = "Harry Potter and Hermione went to Hogwarts! It's magical."
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
tokens = word_tokenize(text)
tokens = [t for t in tokens if t not in stopwords.words('english')]
print(tokens)

```

---

## 🔹 5. Word Embeddings

Traditional ML used “bag of words,” which ignored word meaning.

**Word embeddings** solve this — representing words as dense numerical vectors that capture **semantic meaning**.

### Example:

The embeddings make:

- `king - man + woman ≈ queen`

### Common Techniques:

| Technique | Description |
| --- | --- |
| **Word2Vec** | Learns embeddings using context windows |
| **GloVe** | Embeddings based on word co-occurrence |
| **FastText** | Considers subword information |
| **Transformer-based embeddings (BERT)** | Contextual — same word has different embeddings based on context |

**Word2Vec Example:**

```python
from gensim.models import Word2Vec
sentences = [["harry", "potter", "is", "a", "wizard"],
             ["hermione", "is", "brilliant"]]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)
print(model.wv.most_similar("harry"))

```

---

## 🔹 6. Deep Learning in NLP

Deep Learning models process text sequentially or contextually to capture patterns, syntax, and semantics.

Two major architectures are used:

---

### ⚙️ 6.1 LSTMs (Long Short-Term Memory Networks)

**Purpose:** Handle long-term dependencies in text sequences.

**Problem Solved:**

Traditional RNNs forget earlier information. LSTMs use **memory cells** and **gates** (input, output, forget) to retain or discard information as needed.

**Example use case:**

Predicting the next word in a sentence like *“Harry looked at Ron and said ___”*

**Simple LSTM Code Example:**

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=10),
    LSTM(100),
    Dense(5000, activation='softmax')
])
model.summary()

```

**Working:**

- Embedding layer converts words to vectors.
- LSTM learns temporal relationships.
- Dense layer outputs probability for the next word.

---

### ⚙️ 6.2 Transformers

Transformers revolutionized NLP by introducing **self-attention**, allowing the model to see **the entire sequence at once**, not just previous words.

**Key Components:**

- **Encoder** → Reads input text
- **Decoder** → Generates output text
- **Self-Attention Mechanism** → Learns which words relate to which others

**Advantages:**

- Parallel processing (faster training)
- Captures global context
- Forms the backbone of modern models (BERT, GPT, T5, etc.)

---

## 🔹 7. BERT and GPT Models

### 🧠 BERT (Bidirectional Encoder Representations from Transformers)

- Reads text **both directions (left → right and right → left)**
- Used for understanding-based tasks (like sentiment, classification)

### 🤖 GPT (Generative Pretrained Transformer)

- Reads text **left-to-right**
- Used for **text generation**, **next-word prediction**, **story completion**

---

## 🔹 8. Practical Implementation: Next-Word Prediction using Pre-Trained Model

We’ll use a **Transformer-based model (GPT-2)** to generate text continuation from the *Harry Potter corpus*.

---

### 🧩 Step 1: Install & Import Libraries

```python
!pip install transformers torch
from transformers import pipeline
```

---

### 🧩 Step 2: Load the Text Generation Pipeline

```python
generator = pipeline("text-generation", model="gpt2")
```

---

### 🧩 Step 3: Give a Harry Potter-style Prompt

```python
prompt = "Harry looked at Ron and said"
result = generator(prompt, max_length=30, num_return_sequences=1, temperature=0.9)
print(result[0]['generated_text'])
```

**Example Output:**

> “Harry looked at Ron and said he could feel something strange in the air, a whisper of magic that made the room glow faintly.”
> 

---

### 🧩 Step 4: Try Other Prompts

```python
prompts = [
    "Voldemort raised his wand and",
    "Hermione opened the book of spells and",
    "Hogwarts castle was quiet until"
]

for p in prompts:
    print(generator(p, max_length=25, num_return_sequences=1)[0]['generated_text'])
```

---

### 🧩 Step 5: Explain the Parameters

| Parameter | Description |
| --- | --- |
| **prompt** | Starting phrase for generation |
| **max_length** | Number of tokens to generate |
| **temperature** | Controls randomness (0.7 = focused, 1.0 = creative) |
| **num_return_sequences** | Number of different outputs to generate |

---

### 🧩 Step 6: Optional — Sentiment Analysis (bonus demo)

```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love Hogwarts but hate exams!"))
```

**Output:**

> [{'label': 'POSITIVE', 'score': 0.99}]
> 

---

## 🔹 9. Key Learnings

| Concept | Summary |
| --- | --- |
| **Word Embeddings** | Represent words as dense numerical vectors capturing meaning |
| **LSTM** | Sequential model that remembers long-term dependencies |
| **Transformer** | Parallel model that captures context using attention |
| **BERT** | Bidirectional model for understanding text |
| **GPT-2** | Generative model for predicting and generating next words |
| **Application** | Next-word prediction using pre-trained model (Harry Potter demo) |

---

##