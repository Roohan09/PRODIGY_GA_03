> *Python • Natural Language Processing • Markov Chains*
**Implement a text generation algorithm using Markov chains — a statistical model that predicts the probability of a word or character based on the previous one(s).**

---

## 📂 Repository Structure

```
PRODIGY_GA_03/
├── markov_text_generation.py    # Main script (word-level, char-level, markovify)
├── Markov_Text_Generation.ipynb # Interactive Jupyter notebook
├── train_data.txt               # Training corpus (AI/ML domain)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/Roohan09/PRODIGY_GA_03.git
cd PRODIGY_GA_03

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Demo (runs all variants)
```bash
python markov_text_generation.py demo
```

### Word-Level Generation
```bash
python markov_text_generation.py generate \
    --train_file train_data.txt \
    --order 2 \
    --seed "Artificial intelligence is" \
    --max_words 100 \
    --num_sequences 3
```

### Character-Level Generation
```bash
python markov_text_generation.py char \
    --train_file train_data.txt \
    --order 4 \
    --max_chars 500 \
    --num_sequences 2
```

### markovify Library
```bash
python markov_text_generation.py markovify \
    --train_file train_data.txt \
    --state_size 2 \
    --num_sentences 5
```

---

## 📓 Notebook

Open `Markov_Text_Generation.ipynb` in Jupyter or Google Colab for an interactive walkthrough including:

- Word-level Markov chains (order 1, 2, and 3)
- Transition probability analysis
- Character-level Markov chains
- markovify library demo
- Interactive generation cell

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Roohan09/PRODIGY_GA_03/blob/main/Markov_Text_Generation.ipynb)

---

## 🔑 Key Concepts

| Concept | Description |
|---|---|
| **Markov Chain** | Statistical model where the next state depends only on the current state |
| **Order** | Number of previous words/chars used as context (higher = more coherent) |
| **State** | A tuple of N consecutive words or a string of N characters |
| **Transition** | The word or character that follows a given state |
| **Word-level** | Operates on words as tokens — grammatically coherent output |
| **Char-level** | Operates on characters — can invent new words |
| **markovify** | Python library for Markov chain text generation with sentence awareness |

---

## 📊 Order Comparison

| Order | Context | Output Quality |
|---|---|---|
| 1 | 1 word | Random, incoherent |
| 2 | 2 words | Readable phrases |
| 3 | 3 words | Coherent sentences |
| 4+ | 4+ words | Closely mirrors training data |

---

## 🧠 How It Works

```
Training:
"The cat sat on the mat"
State ("The", "cat") → ["sat"]
State ("cat", "sat") → ["on"]
State ("sat", "on")  → ["the"]
...

Generation:
Start: ("The", "cat")
→ pick random from ["sat"] → "sat"
→ new state: ("cat", "sat")
→ pick random from ["on"] → "on"
→ Output: "The cat sat on ..."
```

---

## 📚 References

- [Text Generation with Markov Chains — Towards Data Science](https://towardsdatascience.com/text-generation-with-markov-chains-an-introduction-to-using-markovify-742e6680dc33)
- [Predictive Text and Text Generation — Notebook](https://github.com/aparrish/predictive-text-and-text-generation/blob/master/predictive-text-and-text-generation.ipynb)
- [markovify Documentation](https://github.com/jsvine/markovify)

---

## 📋 LinkedIn Post Template

> Just completed **Task 03** of my Generative AI Internship at **ProDigy Infotech**! 🚀
>
> 🔗 **Project:** Text Generation with Markov Chains
>
> In this task, I:
> ✅ Built a word-level Markov chain text generator from scratch
> ✅ Implemented character-level generation for creative text synthesis
> ✅ Compared chain orders (1, 2, 3) and their effect on output quality
> ✅ Analysed transition probabilities to understand the statistical model
> ✅ Used the markovify library for sentence-aware generation
>
> 💡 Key learnings: n-gram language models, probabilistic state transitions, the trade-off between order and overfitting, and how classical statistical methods compare to modern neural approaches.
>
> 🔗 GitHub: github.com/Roohan09/PRODIGY_GA_03
>
> #GenerativeAI #MarkovChains #NLP #MachineLearning #ProdigyInfotech #Internship #AI #Python

---

*Made with ❤️ during the ProDigy Infotech Generative AI Internship*
