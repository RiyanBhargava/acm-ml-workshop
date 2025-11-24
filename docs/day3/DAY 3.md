# DAY 3: DEEP LEARNING & NLP

---

# **1. Overview**

In this workshop you'll learn:

- **Deep Learning Basics:** Covers the fundamentals, starting from a single **neuron**, building up to **neural networks**, and explaining the "learning" process of **gradient descent** and **backpropagation**.
- **What is NLP? :** Introduces Natural Language Processing and its evolution from old **rules-based** systems to modern **Deep Learning** models.
- **Turning Words into Numbers:** Explains the critical step of **vectorization**, contrasting older methods like **Bag-of-Words (BoW)**, **TF-IDF**, and **One-Hot Encoding** with modern **Word Embeddings** (Word2Vec, GloVe, fastText) that capture meaning.
- **Understanding Sequence & Memory:** Describes why word order matters and how **RNNs** (Recurrent Neural Networks) and their powerful upgrade, **LSTMs** (Long Short-Term Memory), were created to process sequences.
- **The Modern Revolution (Transformers):** Details the breakthrough **Attention Mechanism** and the two dominant models it created: **BERT** (for understanding context) and **GPT** (for generating text).
- **Challenges & Applications:** Briefly touches on why human language is so hard for AI (like sarcasm and bias) and where NLP is used in the real world (e.g., finance, healthcare).

---

# **2. Workshop Resources**

### **(Make a copy and run the cells) :**

### **📓 Colab Notebook:**

[Open in Google Colab](https://colab.research.google.com/drive/1RGcpQuLJz-I7EYQPfaEYDR01m6IqEMfG?usp=sharing) 

### **📊 Dataset:**

[harry_potter_corpus.txt](../files/day3/harry_potter_corpus.txt)

---

# 3. Introduction to Deep Learning (How Computers "Learn")

Welcome! Before we teach a computer to read, we must first understand how a computer "learns" at all. The main idea is **Deep Learning (DL)**.

Imagine you want to teach a computer to recognize your handwriting. How would it do that? This is the core problem Deep Learning solves.

DL is a method inspired by the human brain. It's not that we're building a real brain, but we're borrowing the key idea: **a network of simple, interconnected units called neurons.**

## 3.1. Why Neural Networks Matter and Their Applications

Neural networks are central to modern AI because they **learn useful internal representations directly from data**, allowing them to capture complex, nonlinear structures that classical models miss. This core capability allows them to power a vast array of real-world AI systems across numerous domains.

Prominent applications include:

- **Computer Vision:** Convolutional Neural Networks (CNNs) are used for image recognition, medical imaging analysis, and powering autonomous vehicles.
- **Natural Language Processing:** Transformers are the basis for machine translation, advanced chatbots, and text summarization.
- **Speech Recognition:** Recurrent Neural Networks (RNNs) and other deep nets are used for transcription services and voice assistants.
- **Forecasting and Time Series:** They are applied to demand prediction, financial modeling, and weather forecasting.
- **Reinforcement Learning:** Neural networks act as function approximators in game-playing agents, such as DeepMind's AlphaGo.
- **Pattern Recognition:** They are highly effective at identifying fraud, detecting anomalies, and classifying documents.

## **3.2. Why Deep Learning over Traditional Machine Learning?**

1. **Automatic Feature Engineering:** This is the biggest advantage. Traditional ML (like Support Vector Machines or Random Forests) relies on *manual feature engineering*. A data scientist must spend significant time selecting and creating features (e.g., "word count" or "average pixel brightness"). Deep Learning models learn the best features *automatically* from the raw data.
2. **Performance with Scale:** Traditional ML models plateau in performance as you give them more data. Deep Learning models *continue to improve* as the volume of data increases.
3. **Handling Unstructured Data:** DL excels at complex, unstructured data like text, images, and audio, where traditional ML struggles.

While that framework is very powerful and versatile, it’s comes at the expense of *interpretability.* There’s often little, if any, intuitive explanation—beyond a raw mathematical one—for how the values of individual model parameters learned by a neural network reflect real-world characteristics of data. For that reason, deep learning models are often referred to as “black boxes,” especially when compared to traditional types of machine learning models.

## 3.3. The Building Block: The Artificial Neuron

Think of a single **neuron** as a tiny, simple decision-maker. It gets some inputs and decides how strongly to "fire" an output.

Here's its job, step-by-step:

1. **It Receives Inputs (X):** These are just numbers. For an image, this could be the brightness value (0-255) of a few pixels.
2. **It Has Weights (W):** Each input has a **weight**. This is the *most important concept*. A weight is just a number that represents **importance**. A high weight means "pay a lot of attention to this input!" A low weight means "this input doesn't matter much."
3. **It Has a Bias (b):** A **bias** is an extra "nudge." It's a number that helps the neuron decide how easy or hard it is to fire. (e.g., "Don't fire unless you are *really* sure").
4. **It Calculates an Output (Y):** The neuron multiplies each input by its weight, adds them all up, adds the bias, and then passes this total through an **Activation Function**. This function just squashes the number (e.g., to be between 0 and 1) to make it a clean, final output signal.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image.png)

## 3.4. Building a "Deep" Brain: The Neural Network

A "deep" network is just many layers of these neurons stacked together. This is where the magic happens!

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%201.png)

For example: Recognizing a handwritten digit

1. **Input Layer:** This layer just "receives" the raw data (e.g., all 784 pixels of a handwritten digit). It doesn't make any decisions.
2. **Hidden Layers:** This is the *real* "brain" of the network. The term "deep" comes from having *multiple* hidden layers. They perform **automatic feature learning**:
    - **Layer 1** might learn to find simple edges and lines.
    - **Layer 2** might combine those edges to find loops and curves.
    - **Layer 3** might combine those loops to recognize a full "8" or "9".
3. **Output Layer:** This layer gives the final answer (e.g., 10 neurons, one for each digit 0-9, where the "9" neuron fires the strongest).

![Screenshot 2025-11-15 at 9.16.03 PM.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/Screenshot_2025-11-15_at_9.16.03_PM.png)

## 3.5. How Does it Learn? (The Training Process)

The power of a neural network is its ability to find the optimal **weights** and **biases** that map inputs to correct outputs. It achieves this by iteratively "learning from its mistakes" through a process driven by **Backpropagation** and **Gradient Descent**.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%202.png)

This learning process is a four-step cycle:

**i. The Forward Pass (The Guess)**
First, the network makes a guess. Inputs (like an image of a "7") are fed *forward* through the network's layers. At each layer, the data is multiplied by the current weights, a bias is added, and it passes through a nonlinear activation function. This produces the network's initial, likely random, prediction (e.g., it guesses "3").

**ii. The Loss Calculation (The Mistake)**
Next, the network measures *how wrong* its guess was. A **Loss Function** (or Cost Function) compares the network's prediction to the true label. This calculation results in a single number, the "loss" or "mistake score," which quantifies the error. A high score means a bad guess; the goal is to get this score as low as possible.

**iii. The Backward Pass (Assigning Blame)**
This is the core of the learning mechanism, enabled by **Backpropagation** (short for "backward propagation of error").

- **Calculates Contribution:** Starting from the final loss score, the algorithm works *backward* through the network, layer by layer.
- **Uses Calculus:** Using the chain rule of calculus, it calculates the "gradient"—a derivative that precisely measures how much each individual weight and bias in the entire network contributed to the final error.
- **Finds Direction:** This gradient "blames" the parameters. It tells the network not only *who* was responsible for the mistake but also *which direction* to nudge each parameter to fix it.

**iv. The Weight Update (The Correction)**

Finally, the network applies the correction using **Gradient Descent**.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%203.png)

**A. Gradient Descent — What It Actually Does:**

- Think of all possible weight combinations as a giant, hilly landscape.
- The **height** of the landscape at any point is the **loss**.
- The goal: **reach the lowest valley** — the point where loss is minimum.

**B. Local Minima vs. Global Minimum:**

- A **global minimum** is the *absolute lowest* point in the entire landscape.
- A **local minimum** is a *small valley* that is lower than its surroundings but not the lowest overall.
- Gradient Descent follows the slope downward, so if the landscape is complex, it may get stuck in a **local minimum** instead of reaching the **global minimum**.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%204.png)

**C. Normal GD vs. Stochastic GD:**

**Stochastic Gradient Descent (SGD)** is a fast and noisy version of **Gradient Descent** used to train machine learning models—especially neural networks, using **ONE** **training example** at a time, instead of the entire dataset.

| Feature | **Normal / Batch Gradient Descent** | **Stochastic Gradient Descent (SGD)** |
| --- | --- | --- |
| **Data Used per Update** | Entire dataset | One data point (or small batch) |
| **Speed** | Very slow | Much faster |
| **Computational Cost** | Very high | Much cheaper |
| **Accuracy of Gradient** | Very accurate | Noisy updates |
| **Effect on Optimization** | Can get stuck in local minima | Noise helps escape local minima and find better solutions |

**Why SGD is used primarily?**

Because computing gradients for millions of samples every step is too expensive. SGD drastically reduces the number of computations and speeds up learning, making training modern neural networks feasible.

---

**v. The Training Loop:**

- This entire four-step cycle is repeated many times, showing the network thousands of data examples. Each full pass through the training dataset is called an **epoch**.
With each epoch, the weights and biases are nudged closer to their optimal values, the "mistake score" descends into the "valley," and the network's predictions become incrementally more accurate.

However, despite practitioners' effort to train high performing models, neural networks still face challenges similar to other machine learning models—most significantly, overfitting. When a neural network becomes overly complex with too many parameters, the model will overfit to the training data and predict poorly. Overfitting is a common problem in all kinds of neural networks, and paying close attention to bias-variance tradeoff is paramount to creating high-performing neural network models.  

## **3.6. Types of neural networks**

While multilayer perceptrons are the foundation, neural networks have evolved into specialized architectures suited for different domains:

- **Convolutional neural networks (CNNs or convnets)**: Designed for grid-like data such as images. CNNs excel at image recognition, computer vision and facial recognition thanks to convolutional filters that detect spatial hierarchies of features.
- **Recurrent neural networks (RNNs)**: Incorporate feedback loops that allow information to persist across time steps. RNNs are well-suited for speech recognition, time series forecasting and sequential data.
- **Transformers**: A modern architecture that replaced RNNs for many sequence tasks. Transformers leverage attention mechanisms to capture dependencies in natural language processing (NLP) and power state-of-the-art models like GPT.

These variations highlight the versatility of neural networks. Regardless of architecture, all rely on the same principles: artificial neurons, nonlinear activations and optimization algorithms.

## **3.7. Applying the Machine to Language**

Now we apply our "learning machine" to the messy, complex problem of human language.

Understanding Natural Language Processing(NLP)At its core, all modern NLP follows a three-step process:

1. **Step 1: Text to Numbers (Embedding):** We must convert raw text ("The quick brown fox...") into a numerical format (vectors) that a machine can understand. This is the most critical step.
2. **Step 2: Process the Numbers (The Model):** The numerical vectors are fed into a deep learning model (like an RNN or a Transformer). This "brain" processes the numbers to "understand" the patterns, context, and relationships.
3. **Step 3: Numbers to Output (The Task):** The model's final numerical output is converted into a human-usable result. This could be:
    - A single label (e.g., "Positive Sentiment").
    - A new sequence of text (e.g., a translation).
    - A specific word (e.g., an "autocomplete" suggestion).

Before deep learning, this process was much more manual.

---

# 4. The Evolution of NLP: Three Main Approaches

To understand language, NLP models have evolved over time. They started with strict, simple rules and grew into the powerful, flexible "learning" systems we have today.

You can think of this evolution in three main stages.

## 4.1. Before We Begin: Two Core Ideas

All NLP, from the simplest to the most complex, relies on two basic ways of analyzing language:

1. **Syntactical Analysis (Grammar):** This is the "rules" part. It focuses on the **structure and grammar** of a sentence. It checks if the word order is correct according to the rules of the language.
    - **Example:** "The cat sat on the mat" is **syntactically correct**.
    - **Example:** "Sat the on mat cat" is **syntactically incorrect**.
2. **Semantical Analysis (Meaning):** This is the "meaning" part. Once it knows the grammar is correct, this step tries to figure out the **meaning and intent** of the sentence.
    - **Example:** "The cat sat on the mat" and "The mat was sat on by the cat" have different *syntax* (structure) but the same *semantics* (meaning).

Now, let's look at how the models evolved.

## 4.2. Approach A: Rules-Based NLP (The "If-Then" Approach)

This was the earliest approach to NLP. It's based on **manually programmed, "if-then" rules**.

- **How it Worked:** A programmer had to sit down and write explicit rules for the computer to follow.
    - `IF` the user says "hello," `THEN` respond with "Hi, how can I help you?"
    - `IF` the user says "What are your hours?" `THEN` respond with "We are open 9 AM to 5 PM."
- **The Problem:** This approach is extremely **limited and not scalable**.
    - It has no "learning" or AI capabilities.
    - It breaks easily. If a user asks, "When are you guys open?" instead of "What are your hours?", the system would fail because it doesn't have a specific rule for that exact phrase.
- **Example:** Early automated phone menus (like Moviefone) that only understood specific commands.

## 4.3. Approach B: Statistical NLP (The "Probability" Approach)

This was the next big step, which introduced **machine learning**. Instead of relying on hard-coded rules, this approach "learns" from a large amount of text.

- **How it Worked:** The model analyzes data and assigns a **statistical likelihood (a probability)** to different word combinations.
    - For example, it learns that after the words "New York," the word "City" is *highly probable*, while the word "banana" is *very improbable*.
- **The Big Breakthrough: Vector Representation.** This approach introduced the essential technique of mapping words to **numbers (called "vectors")**. This allowed, for the first time, computers to perform mathematical and statistical calculations on words.
- **Examples:** Older spellcheckers (which suggest the *most likely* correct word) and T9 texting on old phones (which predicted the *most likely* word you were typing).

> A Quick Note on Training:
These models needed "labeled data"—data that a human had already manually annotated (e.t., "This is a noun," "This is a verb"). This was slow and expensive.
A key breakthrough called Self-Supervised Learning (SSL) allowed models to learn from unlabeled raw text, which is much faster and cheaper and a key reason why modern Deep Learning is so powerful.
> 

## 4.4. Approach C: Deep Learning NLP (The "Modern" Approach)

This is the dominant, state-of-the-art approach used today. It's an evolution of the statistical method but uses powerful, multi-layered **neural networks** to learn from *massive* volumes of unstructured, raw data.

These models are incredibly accurate because they can understand complex context and nuance. Several types of deep learning models are important:

- **Sequence-to-Sequence (Seq2Seq) Models:**
    - **What they do:** They are designed to transform an input sequence (like a sentence) into a *different* output sequence.
    - **Best for:** Machine Translation. (e.g., converting a German sentence into an English one).
- **Transformer Models:**
    - **What they do:** This is the *biggest breakthrough* in modern NLP. Transformers use a mechanism called **"self-attention"** to look at all the words in a sentence at once and calculate how *important* each word is to all the other words, no matter how far apart.
    - **Example:** Google's **BERT** model, which powers its search engine, is a famous transformer.
- **Autoregressive Models:**
    - **What they do:** This is a type of transformer model that is expertly trained to do one thing: **predict the next word in a sequence**. By doing this over and over, it can generate entire paragraphs of human-like text.
    - **Examples:** **GPT** (which powers ChatGPT), Llama, and Claude.
- **Foundation Models:**
    - **What they do:** These are *huge*, pre-trained "base" models (like **IBM's Granite** or OpenAI's GPT-4) that have a very broad, general understanding of language. They can then be quickly adapted for many specific tasks, from content generation to data extraction.

---

# **5. How NLP Works: The 4-Step Pipeline**

A computer can't just "read" a sentence. To get from raw human language to a useful insight, it follows a strict, step-by-step "assembly line."

## **5.1. Step 1: Text Preprocessing (The "Cleaning" Step)**

First, we clean up the raw text and turn it into a standardized format. This is the "prep work" in a kitchen—getting your ingredients (the words) ready before you start cooking (the analysis).

- **Tokenization:** Splitting a long string of text into smaller pieces, or "tokens."
    - *Example:* "The cat sat" becomes `["The", "cat", "sat"]`
- **Lowercasing:** Converting all characters to lowercase.
    - *Example:* "Apple" and "apple" both become `"apple"`.
- **Stop Word Removal:** Removing common "filler" words (like "is," "the," "a," "on") that add little unique meaning.
- **Stemming & Lemmatization:** Reducing words to their "root" form (e.g., "running," "ran," and "runs" all become "run").
- **Text Cleaning:** Removing punctuation, special characters (@, #), numbers, etc.

## **5.2. Step 2: Feature Extraction (The "Converting" Step)**

This is a critical step. **Computers do not understand words; they only understand numbers.** Feature extraction converts the clean text tokens into a numerical representation (a "vector") that a machine can actually analyze.

### 5.2.1. The "Old Way" (Statistical Counts)

Before we had powerful neural networks, we relied on **statistics and word counts**. These models were clever but lacked any *real* understanding.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%204.png)

- **Bag-of-Words (BoW):**
    - **How it Works:** The simplest method. It treats a sentence as a "bag" (a jumbled set) of words, ignoring order. It just *counts* how many times each word appears.
    - **Example:** To a BoW model, "The man bit the dog" and "The dog bit the man" are *exactly the same*. They both contain `{"the": 2, "man": 1, "bit": 1, "dog": 1}`.
    - **Limitation:** It has zero understanding of context or grammar.
- **TF-IDF (Term Frequency-Inverse Document Frequency):**
    - **How it Works:** A "smarter" version of BoW. It scores words not just on *frequency*, but on *importance*. A word gets a high score if it's frequent in *this* document but *rare* in all other documents.
    - **Example:** In a set of news articles, the word "the" is common everywhere (low score). The word "astrophysics" is rare, so in an article about space, it gets a *very high* score.
    - **Limitation:** It's great for search engines, but it still has no *semantic meaning*. It doesn't know that "cat" and "kitten" are related. To TF-IDF, they are just two different, meaningless tokens.
- **One-Hot Encoding**
    - **Idea:** Create a giant list (a vector) for your entire vocabulary (e.g., 50,000 words). Each word gets a vector of all zeros, except for a single "1" at its own position.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%205.png)

### **i. Bag-of-Words (BoW):**

- **How it Works:**
    
    Treats a sentence as a collection (“bag”) of words. Ignores grammar and word order.
    
- **Example Sentence:**
    
    **“The cat sat on the mat”**
    
- **Vocabulary (example):**
    
    `["the", "cat", "sat", "on", "mat"]`
    
- **BoW Vector:**
    
    Count how many times each word appears:
    
    | Word | the | cat | sat | on | mat |
    | --- | --- | --- | --- | --- | --- |
    | Count | 2 | 1 | 1 | 1 | 1 |
    
    **BoW Vector:** → **[2, 1, 1, 1, 1]**
    
- **Limitation:**
    
    “The cat chased the dog” and “The dog chased the cat” would look almost identical.
    

---

### **ii. TF-IDF (Term Frequency-Inverse Document Frequency):**

- **How it Works:**
    
    Gives high importance to words that are *frequent in this document* but *rare across all documents*.
    
- **Example Sentence:**
    
    **“The cat sat on the mat”**
    
- **Vocabulary:** same as before.
- **TF-IDF Vector (example values):**
    
    
    | Word | the | cat | sat | on | mat |
    | --- | --- | --- | --- | --- | --- |
    | TF-IDF | 0.0 | 0.52 | 0.64 | 0.48 | 0.64 |
    
    → **TF-IDF Vector:** **[0.0, 0.52, 0.64, 0.48, 0.64]**
    
    (“the” gets **0.0** because it appears everywhere in the dataset, so IDF ≈ 0)
    
- **Limitation:**
    
    Still no understanding of meaning (e.g., “cat” ≠ “kitten” to TF-IDF).
    

### **iii. One-Hot Encoding**

- **Idea:**
    
    Every word gets a giant vector full of zeros except **one 1** at its index in the global vocabulary.
    
- **Vocabulary Example (size = 5):**
    
    `["the", "cat", "sat", "on", "mat"]`
    

| Word | Vector |
| --- | --- |
| the | [1, 0, 0, 0, 0] |
| cat | [0, 1, 0, 0, 0] |
| sat | [0, 0, 1, 0, 0] |
| on | [0, 0, 0, 1, 0] |
| mat | [0, 0, 0, 0, 1] |
- **Sentence Representation:**
    
    Usually stored as 5 separate one-hot vectors (one per word), e.g.:
    
    **“The cat sat on the mat” →**
    
    [1,0,0,0,0]
    [0,1,0,0,0]
    [0,0,1,0,0]
    [0,0,0,1,0]
    [1,0,0,0,0]
    [0,0,0,0,1]
    
- **Problems:**
    - If vocab = 50,000 → each word is a 50,000-length vector
    - No semantic meaning at all
    - “cat”, “dog”, and “car” all look equally unrelated

### **5.2.2. The "Modern Way" (Contextual Embeddings)**

Instead of *counting*, modern NLP systems *learn* the **meaning** of words by training neural networks on billions of sentences.

- **Word2Vec (Word to Vector):**
    - **How it Works:** We train a simple neural network on a "fake" task: "Given a word (like *fox*), predict the words around it (*quick, brown, jumps, over*)." We train this on billions of sentences.
    - **The "Aha!" Moment:** We don't care about the network's predictions. We *steal its weights*. This learned weight matrix becomes a lookup table where each word has its own 300-dimension vector (its "embedding").
    
    ![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%206.png)
    
    - **Advantages:**
        - This was the first method to capture **semantic meaning**.
        - It created the famous analogy: `Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")`.
        
        ![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%207.png)
        
    - **Limitations:**
        - **Out-of-Vocabulary (OOV):** If a word like "brunchfast" wasn't in its training data, Word2Vec has no vector for it.
        - **Polysemy (Many Meanings):** The word "bank" (river bank vs. money bank) has *only one vector*. That vector is a blurry, "average" of all its meanings.

Word2Vec was just the start. Other models iterated on this idea.

- **GloVe (Global Vectors):**
    - **How it Works:** Word2Vec learns from "local" windows (a few words at a time). GloVe learns from "global" statistics. It first builds a giant co-occurrence matrix of *how often every word appears near every other word* in the entire corpus. It then uses a technique (matrix factorization) to "compress" this giant matrix down into the same kind of word vectors.
    - **Advantages:** Often performs better at capturing global relationships and analogies than Word2Vec.
    - **Limitations:** Same as Word2Vec. It still has the **OOV problem** and the **polysemy ("bank") problem**.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%208.png)

- **fastText (from Facebook):**
    - **How it Works:** This model's insight was brilliant. It *doesn't* learn vectors for words. It learns vectors for **character n-grams** (sub-word pieces).
    - **Example:** The vector for "brunch" is the *sum* of the vectors for its parts (e.g., `<br`, `bru`, `run`, `unc`, `nch`, `ch>`).
    
    ![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%209.png)
    
    - **Advantages:**
        1. **Solves the OOV problem:** It can create a vector for *any* word, even misspelled ones ("brunchfastly"), by summing its sub-word parts.
        2. **Understands Morphology:** It knows "run" and "running" are related because they share many character n-grams.
    - **Limitations:**
        1. **Still has the polysemy ("bank") problem.** (This isn't solved until Transformers).
        2. **Storage:** The dictionary of all n-grams is *massive*, making the model files very large.

## **6.3. Step 3: Text Analysis (The "Understanding" Step)**

Now that our text is in a clean, numerical format, the real work can begin. This step involves feeding the numerical data into a **model architecture** (the "brain") to interpret and extract meaningful information.

### 5.3.1. Traditional Analysis Tasks

This is *what* we want the model to do:

- **Part-of-Speech (POS) Tagging:** Identifying nouns, verbs, adjectives, etc.
- **Named Entity Recognition (NER):** Finding people, places, and organizations.
- **Sentiment Analysis:** Determining if the tone is positive or negative.
- **Topic Modeling:** Finding the main themes in a document.

### **5.3.2. Modern Model Architectures (The "Brain")**

A standard ANN has no memory. If you give it:

```
Input 1: “how”
Input 2: “are”
```

It completely forgets “how” when it receives “are”.

But language is **sequential**. Meaning depends on *order*.

So we need models that can “remember” previous inputs.

---

### **i. Recurrent Neural Networks (RNNs)**

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%2010.png)

An RNN adds a **loop** so that information can be passed from one step to the next.

```
Input x₁ → (RNN) → h₁ → output
Input x₂ → (RNN) → h₂ → output
Input x₃ → (RNN) → h₃ → output
```

Where:

- **xₜ = word embedding at time t**
- **hₜ = hidden state (RNN’s memory)**

The hidden state is updated as:

```
hₜ = tanh(Wx * xₜ + Wh * hₜ₋₁)
```

### **Example: Encode a 4-word sentence step-by-step**

Sentence: **“I love deep learning”**

Assume each word is turned into a 4-dimensional vector (tiny example to understand the process).

Let the embeddings be:

```
I          → [1, 0, 0, 0]
love       → [0, 1, 0, 0]
deep       → [0, 0, 1, 0]
learning   → [0, 0, 0, 1]
```

Start with **h₀ = [0,0,0,0]** (zero memory).

---

**Step 1: word = “I”**

```
x₁ = [1, 0, 0, 0]
h₁ = tanh(Wx*x₁ + Wh*h₀)
```

Hidden state might become something like:

```
h₁ = [0.6, -0.1, 0.2, 0.0]
```

---

**Step 2: word = “love”**

```
x₂ = [0, 1, 0, 0]
h₂ = tanh(Wx*x₂ + Wh*h₁)
```

Now memory includes both **I** + **love**:

```
h₂ = [0.40, 0.55, -0.10, 0.20]
```

---

**Step 3: word = “deep”**

```
h₃ = tanh(Wx*x₃ + Wh*h₂)
```

Memory grows again:

```
h₃ = [0.10, 0.62, 0.33, 0.45]
```

---

**Step 4: word = “learning”**

```
h₄ = tanh(Wx*x₄ + Wh*h₃)
```

Final encoding of the entire sentence:

```
h₄ = [0.21, 0.70, 0.55, 0.63]
```

**This final `h₄` is the vector representation of the whole sentence.**

This is how RNNs represent sequences.

---

**Why RNNs Struggle:** They only have **one state (`hₜ`)**. Over many steps, early information fades → **vanishing gradient problem**.

---

### **ii. Long Short-Term Memory (LSTMs)**

**LSTMs** are a specialized, more advanced type of RNN, designed specifically to solve the long-term memory problem.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%2011.png)

An LSTM doesn't just have a simple loop; it has a complex internal structure based on a "cell state" and three "gates":

- **Cell State:** A "conveyor belt" that carries relevant information through the entire sequence.
- **Forget Gate:** A "doorman" that looks at the new input and decides what old information (if any) to *remove* from the cell state.
- **Input Gate:** Decides what *new* information from the current input is important enough to *add* to the cell state.
- **Output Gate:** Decides what part of the cell state to *use* to make the final prediction for the current step.

By using these gates, an LSTM can learn to "remember" important information from long ago (e.g., the subject of a sentence) and "forget" irrelevant details.

---

### **iii. The Modern Revolution (The Transformer)**

Even LSTMs struggle with very long sentences, and their sequential nature (processing one word at a time) makes them slow to train. The **Transformer** architecture solved this.

### **a) Encoder-Decoder Models**

This architecture is key to tasks like machine translation.

1. **Encoder:** An "encoder" (which could be an RNN) reads the entire input sentence (e.g., "How are you?") and compresses its full meaning into a single vector (a "context vector").
2. **Decoder:** A "decoder" (another RNN) takes that *one* vector and "decodes" it into the output sentence (e.g., "¿Cómo estás?").
- **The Problem:** This single context vector is a **bottleneck**. It's hard to cram the entire meaning of a 50-word sentence into one vector.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%2012.png)

### **b) The Breakthrough: The Attention Mechanism**

**Attention** solved the bottleneck. Instead of forcing the decoder to rely on *one* vector, it allows the decoder to "look back" at *all* the encoder's outputs from the *entire* input sentence at every step.

It learns to "pay attention" to the specific input words that are most relevant for generating the *current* output word. This was a massive leap in performance.

- **Advantage:** It's **highly parallelizable** (much faster to train) and can capture *extremely* long-range dependencies, making it the new state-of-the-art.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%2013.png)

### **iv. Modern Models: BERT & GPT**

These are the two most famous models built on the Transformer architecture.

### **a) BERT (Bidirectional Encoder Representations from Transformers)**

- **What it is:** An **Encoder-only** Transformer.
- **How it Learns:** It's trained by taking a sentence, "masking" (hiding) 15% of the words, and then trying to predict those hidden words.
- **Key Feature:** It's **bidirectional**. To predict a masked word, it looks at *both* the words that come *before* it and the words that come *after* it.
- **Best For:** **Understanding** tasks. It builds a deep understanding of context, making it perfect for sentiment analysis, question answering, and text classification.

### **b) GPT (Generative Pre-trained Transformer)**

- **What it is:** A **Decoder-only** Transformer.
- **How it Learns:** It's trained as a "language model," meaning it simply tries to predict the *very next word* in a sentence, given all the words that came before it.
- **Key Feature:** It's **auto-regressive** (one-way). It only looks *backward* (at the words that came before).
- **Best For:** **Generation** tasks. Because it's trained to "predict the next word," it is exceptional at writing essays, holding conversations, summarizing text, and generating creative content.

![image.png](../assets/DAY%203%20DEEP%20LEARNING%20&%20NLP/image%2014.png)

---

## **5.4. Step 4: Model Training (The "Learning" Step)**

This step is the *process* that "teaches" the model architectures from Step 3.

This is where the model "learns" by looking for patterns and relationships within the data.

1. **Feed Data:** The model (e.g., BERT) is fed the numerical data from Step 2.
2. **Make Prediction:** It makes a prediction (e.g., "I think this movie review is positive").
3. **Check Answer:** It checks its prediction against the right answer (the "label").
4. **Measure Error:** It measures how "wrong" it was (this is called the "loss").
5. **Adjust:** It slightly adjusts its internal parameters (weights) to be "less wrong" next time.

This process is repeated millions or even billions of times. Once "trained," this model can be saved and used in Step 3 to make predictions on new, unseen data.

---

# **6. Why is NLP So Hard?**

Human language is incredibly complex and messy. Even the best NLP models struggle with the same things humans do. These "ambiguities" are the biggest challenge.

- **Biased Training Data:** If the data used to train a model is biased (e.g., pulled from biased parts of the web), the model's answers will also be biased. This is a major risk, especially in sensitive fields like healthcare or HR.
- **Misinterpretation ("Garbage In, Garbage Out"):** A model can easily get confused by messy, real-world language, including:
    - Slang, idioms, or fragments
    - Mumbled words or strong dialects
    - Bad grammar or misspellings
    - Homonyms (e.g., "bear" the animal vs. "bear" the burden)
- **Tone of Voice & Sarcasm:** The *way* something is said can change its meaning completely. Models struggle to detect sarcasm or exaggeration, as they often only "read" the words, not the intent.
- **New and Evolving Language:** New words are invented all the time ("rizz," "skibidi"), and grammar rules evolve. Models can't keep up unless they are constantly retrained.

---

# **7. Where is NLP Used?**

You can find NLP applications in almost every major industry.

- **Finance:** NLP models instantly read financial reports, news articles, and social media to help make split-second trading decisions.
- **Healthcare:** NLP analyzes millions of medical records and research papers at once, helping doctors detect diseases earlier or find new insights.
- **Insurance:** Models analyze insurance claims to spot patterns (like potential fraud) and help automate the claims process.
- **Legal:** Instead of lawyers manually reading millions of documents for a case, NLP can automate "legal discovery" by scanning and finding all relevant information.

A computer can't just "read" a sentence. To get from raw human language to a useful insight, it follows a strict "assembly line" process.

---

# **8. Practical Implementation: Next-Word Prediction using Pre-Trained Model**

## Fine-Tuning BERT on Harry Potter Corpus

**Open the Colab Link, Make a Copy and Upload the dataset on Colab**

**📓 Colab Notebook:**

[Open in Google Colab](https://colab.research.google.com/drive/1RGcpQuLJz-I7EYQPfaEYDR01m6IqEMfG?usp=sharing) 

**📊 Dataset:**  

[harry_potter_corpus.txt](../files/day3/harry_potter_corpus.txt)

---

# 9. Summary

- Covered **Deep Learning basics**, including artificial neurons, neural networks, and how models learn using **forward pass, loss, backpropagation, and gradient descent**.
- Explored **why deep learning outperforms traditional ML**, handling unstructured data and learning features automatically.
- Introduced **NLP**, its evolution from **rules-based** to **statistical** to **deep learning approaches**.
- Learned **text preprocessing**, feature extraction, and vectorization methods: **BoW, TF-IDF, One-Hot, Word2Vec, GloVe, fastText**.
- Studied **sequence models**: RNNs, LSTMs, and the **Transformer architecture** with **attention mechanism**.
- Covered modern NLP models: **BERT for understanding** and **GPT for text generation**, and their real-world applications.
- Discussed **challenges in NLP**, like ambiguity, sarcasm, bias, evolving language, and applications in finance, healthcare, insurance, legal, and more.

## See you next week! 🚀

---
