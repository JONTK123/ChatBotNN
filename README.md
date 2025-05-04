# 🧠 Token-Predict-API – Neural‑Network TokenPredictor

A **tiny, visual LLM playground** built with **PyTorch**, **FastAPI** and vanilla **JavaScript**.  
It *predicts the next token (word)* given the **last 4 tokens** of a sentence and streams all training internals to the browser in real‑time.

> **Educational only** – perfect for demos, live coding, and understanding how language‑model pieces fit together.

---

## 🎯  Purpose

*   De‑mystify how *small* LLMs learn token‑by‑token  
*   Visualise **embeddings, activations, weight deltas** while training  
*   Mix **hand‑made tokenisation** with a real **GPT‑2 tokenizer**  
*   Provide a hackable code‑base for experiments and blog posts  

---

## 📦  Features

| Category | Details                                                                        |
|----------|--------------------------------------------------------------------------------|
| **Model** | Custom `MiniGPT` (embeddings → 1 or 2 hidden layers → vocab logits)            |
| **Visuals** | Live loss line‑chart, 3‑D loss surface, confusion matrix, top‑error tokens     |
| **Frontend** | Plain HTML+ JS (Plotly, Cytoscape.js, WebSockets)                              |
| **Training** | Adam+CrossEntropy, configurable epochs, live batch streaming                   |
| **Tokeniser** | GPT‑2 tokenizer (**HuggingFace**) + manual split for comparison                |
| **API** | FastAPI endpoints `/treinar`, `/tokenizar`, `/completar`, plus WebSocket `/ws` |

---

## ⚠️  Limitations

* Predicts **only the next token** – no full text generation  
* Context window hard‑coded to **4 tokens**  
* Needs **≥ 50 phrases** and around **100‑150 epochs** for stable learning  
* Frontend is deliberately **minimal** (study‑oriented)  
* Not production‑ready; expect mistakes for rare / unseen tokens  

---

## 💡  Ideas for Improvement

*   Extend context to 6‑8 tokens  
*   Wider embeddings / hidden layers  
*   Larger, more diverse training corpora  
*   Replace FF‑layers with GRU / Transformer blocks  
*   Generate full sentences instead of token‑by‑token  

---

## 🧭  System Flow

1. Paste training phrases → click **“Tokenise!”**  
2. App shows both **split tokens** and **GPT‑2 tokens**  
3. Click **“Train”** → live graphs & activations stream via WebSocket  
4. When done, type a 4‑token prompt and press **“Complete”**  
5. API returns the most probable next token, rendered instantly  

---

## ✨  Example Prompts

| Prompt (`last 4 tokens`) | Likely Prediction |
|--------------------------|-------------------|
| `eu gosto de comer`      | `maçã`, `banana`, … |
| `hoje o céu está`        | `limpo`, `nublado`, … |
| `você precisa estudar`   | `mais`, `agora`, … |
| `ela gosta de pintar`    | `quadros`, `paredes` |

---

## 📂  Mini Dataset (you can swap your own - at any language!!!!!!!)

```python
frases = [
    "eu gosto de comer maçã",
    "ela gosta de pintar quadros",
    "nós vamos ao parque amanhã",
    "ele vai sair hoje à tarde",
    "você precisa estudar agora",
    "hoje o céu está limpo",
    "amanhã o tempo estará frio",
    "nós queremos comprar uma casa",
    "você quer comprar um celular",
    "eles vão ao mercado cedo",
]
```

## 🛠Installation & Usage

###1–Clone

```python
git clone https://github.com/JONTK123/Token-Predict-API.git
cd ChatBotNN
```

### 2 – Dependencies

```python
python -m venv .venv
# Windows  →  .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 3 – Start the API

```python
uvicorn backend.nn:app --reload
```

### 4 – Open the Front‑End

Simply open frontend/index.html in your browser.
Paste sentences → Tokenise → Train → play with predictions

###5 - Requirements.txt

```python
fastapi
uvicorn
torch
transformers
scikit-learn
matplotlib
numpy
anyio
```

## 🔍How It Works

1. Tokenisation – GPT‑2 tokenizer (HuggingFace) + naive .split() for side‑by‑side comparison
2. Pair generation – for every sentence:
Input = first N tokens → Label = the next token
3. Model – embeddings → hidden layer(s) → vocab‑size logits 
4. Training loop – Adam + CrossEntropy, live progress pushed via WebSocket
5. Inference – give 4 tokens, API returns argmax(logits) as the predicted next token

## 🧠 About

Built by [@Thiago / JONTK123] to learn & teach:
Linkedin -> https://www.linkedin.com/in/thiago-luiz-fossa-26b440276/?locale=pt_BR

- Neural‑network fundamentals
- Embedding layers and tokenisation quirks
- PyTorch training pipelines
- Mini‑LLM behaviour on tiny datasets
## 🚧 Disclaimer

This repository **is not production-ready**.  
Predictions may be wrong or biased, especially for unseen words.  
It exists purely for learning, experimentation, and visual intuition.

Pull requests, issues, and discussions are welcome – have fun.