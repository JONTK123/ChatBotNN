# 🧠 Token-Predict-API – Preditor de Tokens de Rede Neural

Um **pequeno playground visual de LLM** construído com **PyTorch**, **FastAPI** e **JavaScript puro**.  
Ele *prediz o próximo token (palavra)* dado os **últimos 4 tokens** de uma frase e transmite todos os detalhes do treinamento para o navegador em tempo real.

> **Apenas educacional** – perfeito para demonstrações, codificação ao vivo e para entender como as partes de um modelo de linguagem se encaixam.

---

## 🎯  Objetivo

*   Desmistificar como os *pequenos* LLMs aprendem token‑por‑token  
*   Visualizar **embeddings, ativações, deltas de pesos** durante o treinamento  
*   Misturar **tokenização feita manualmente** com um **tokenizador GPT‑2 real**  
*   Fornecer uma base de código hackeável para experimentos e posts de blog  

---

## 📦  Funcionalidades

| Categoria | Detalhes                                                                      |
|-----------|--------------------------------------------------------------------------------|
| **Modelo** | `MiniGPT` customizado (embeddings → 1 ou 2 camadas ocultas → logits de vocabulário) |
| **Visuais** | Gráfico de perda ao vivo, superfície de perda 3D, matriz de confusão, tokens com mais erros |
| **Frontend** | HTML + JS simples (Plotly, Cytoscape.js, WebSockets)                           |
| **Treinamento** | Adam + CrossEntropy, épocas configuráveis, transmissão ao vivo de batches    |
| **Tokenizador** | Tokenizador GPT‑2 (**HuggingFace**) + split manual para comparação          |
| **API** | Endpoints FastAPI `/treinar`, `/tokenizar`, `/completar`, além de WebSocket `/ws` |

---

## ⚠️  Limitações

* Prediz **apenas o próximo token** – sem geração de texto completo  
* Janela de contexto fixa para **4 tokens**  
* Requer **≥ 50 frases** e cerca de **100‑150 épocas** para aprendizado estável  
* Frontend é deliberadamente **mínimo** (voltado para estudo)  
* Não está pronto para produção; espere erros para tokens raros ou desconhecidos  

---

## 💡  Ideias para Melhoria

*   Estender o contexto para 6‑8 tokens  
*   Embeddings / camadas ocultas maiores  
*   Corpus de treinamento maior e mais diversificado  
*   Substituir camadas FF por blocos GRU / Transformer  
*   Gerar frases completas em vez de token‑por‑token  

---

## 🧭  Fluxo do Sistema

1. Cole frases de treinamento → clique em **“Tokenizar!”**  
2. O app exibe tanto os **tokens separados** quanto os **tokens GPT‑2**  
3. Clique em **“Treinar”** → gráficos ao vivo e ativações são transmitidos via WebSocket  
4. Quando terminar, digite um prompt de 4 tokens e pressione **“Completar”**  
5. A API retorna o próximo token mais provável, renderizado instantaneamente  

---

## ✨  Exemplos de Prompts

| Prompt (`últimos 4 tokens`) | Previsão Provável |
|-----------------------------|-------------------|
| `eu gosto de comer`         | `maçã`, `banana`, … |
| `hoje o céu está`           | `limpo`, `nublado`, … |
| `você precisa estudar`      | `mais`, `agora`, … |
| `ela gosta de pintar`       | `quadros`, `paredes` |

---

## 📂  Mini Dataset (você pode trocar o seu próprio – em qualquer idioma!!!!!!!)

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
## 🛠Instalação & Uso

### 1–Clonar

```python
git clone https://github.com/JONTK123/Token-Predict-API.git
cd ChatBotNN
```

### 2 – Dependências

```python
python -m venv .venv
# Windows  →  .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 3 – Iniciar a API

```python
uvicorn backend.nn:app --reload
```

### 4 – Abrir o Front‑End

Basta abrir frontend/index.html no seu navegador.
Cole frases → Tokenize → Treine → brinque com as previsões

### 5 - Requirements.txt

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

## 🔍Como Funciona

1. Tokenização – Tokenizador GPT‑2 (HuggingFace) + .split() ingênuo para comparação lado a lado
2. Geração de pares – para cada frase: Entrada = primeiros N tokens → Rótulo = o próximo token
3. Modelo – embeddings → camada(s) oculta(s) → logits de tamanho do vocabulário
4. Loop de treinamento – Adam + CrossEntropy, progresso ao vivo enviado via WebSocket
5. Inferência – forneça 4 tokens, a API retorna argmax(logits) como o próximo token previsto

## 🧠 Sobre

Construído por [@Thiago / JONTK123] para aprender e ensinar:
Linkedin -> https://www.linkedin.com/in/thiago-luiz-fossa-26b440276/?locale=pt_BR
- Fundamentos de redes neurais
- Camadas de embedding e peculiaridades da tokenização
- Pipelines de treinamento PyTorch
- Comportamento de Mini‑LLM em conjuntos de dados pequenos
- Tokenização e embeddings
- Visualização de ativações e pesos
- Interpretação de matrizes de confusão
- Análise de erros e tokens mais prováveis
- Uso de WebSockets para streaming em tempo real
- Integração de front-end e back-end com FastAPI
- Criação de gráficos interativos com Plotly
- Manipulação de dados com NumPy e Matplotlib
- Uso de bibliotecas de aprendizado de máquina como Scikit-learn
- Criação de APIs RESTful com FastAPI
- Desenvolvimento de aplicações web com HTML e JavaScript

## 🚧 Aviso

Este repositório **não está pronto para produção**.
As previsões podem estar erradas ou tendenciosas, especialmente para palavras desconhecidas.
Ele existe puramente para aprendizado, experimentação e intuição visual.
Pull requests, problemas e discussões são bem-vindos – divirta-se.


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