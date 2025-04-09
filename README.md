# ChatBotNN

## English

### Overview

**ChatBotNN** is a simple neural network-based chatbot prototype focused on natural language understanding within the context of hotel services.  
It uses a small, manually crafted dataset of hotel-related phrases and demonstrates basic tokenization, dataset preparation, and training of a minimal language model using **PyTorch**.

This project is designed for **educational and experimental purposes**, aiming to explain the inner workings of tokenization and language model training.

---

### Dataset

A basic dataset composed of hotel-related sentences in Portuguese:

```python
frases = [
    "o hotel oferece wi-fi gratuito",
    "o check-in começa às 14h",
    "o café da manhã é servido até as 10h",
    "a recepção funciona 24 horas",
    "o check-out é até meio-dia",
    "para ligar nao perturbe, toque no interruptor 2",
    "o hotel tem piscina aquecida",
    "o hotel tem academia",
    "o hotel tem estacionamento gratuito",
    "o hotel tem serviço de lavanderia",
    "o hotel tem serviço de quarto 24 horas",
]
```

---

### Key Components

#### Tokenization (Manual and via Transformers)

- Manual token-to-ID mapping
- Tokenization using HuggingFace's `AutoTokenizer` (GPT-2)

#### Training Sample Generation

- Pairs of (partial sentence → next word)
- Prepared both manually and using tokenizer-based IDs

#### Model Architecture

A small neural network (`MiniGPT`) with:

- Embedding layer
- Hidden dense layer (ReLU)
- Output layer predicting vocabulary tokens

#### Training

- Custom PyTorch `Dataset` and `DataLoader`
- CrossEntropy loss
- `Adam` optimizer
- 50 training epochs

#### Inference

- A function to generate the next word based on an input sentence

---

### Libraries Used

- `torch`
- `transformers` (HuggingFace)
- `torch.nn`, `torch.utils.data`

---

## Português

### Visão Geral

**ChatBotNN** é um protótipo simples de chatbot baseado em rede neural, com foco na compreensão de linguagem natural em contexto hoteleiro.  
Utiliza um dataset básico de frases relacionadas a serviços de hotel e demonstra, de forma didática, os processos de tokenização, preparação de dados e treinamento de um modelo de linguagem minimalista com **PyTorch**.

Este projeto tem fins **educacionais e exploratórios**, buscando mostrar o funcionamento interno de um modelo de linguagem.

---

### Dataset

Um dataset simples com frases sobre serviços de hotel:

```python
frases = [
    "o hotel oferece wi-fi gratuito",
    "o check-in começa às 14h",
    "o café da manhã é servido até as 10h",
    "a recepção funciona 24 horas",
    "o check-out é até meio-dia",
    "para ligar nao perturbe, toque no interruptor 2",
    "o hotel tem piscina aquecida",
    "o hotel tem academia",
    "o hotel tem estacionamento gratuito",
    "o hotel tem serviço de lavanderia",
    "o hotel tem serviço de quarto 24 horas",
]
```

---

### Componentes Principais

#### Tokenização (Manual e via Transformers)

- Mapeamento manual de palavras para IDs
- Tokenização usando `AutoTokenizer` (GPT-2) da HuggingFace

#### Geração de Dados de Treino

- Pares de (frase parcial → próxima palavra)
- Gerados manualmente e via IDs tokenizados

#### Arquitetura do Modelo

Rede neural simples (`MiniGPT`) com:

- Camada de embeddings
- Camada oculta densa (ReLU)
- Camada de saída para prever o próximo token

#### Treinamento

- `Dataset` e `DataLoader` com PyTorch
- Função de perda CrossEntropy
- Otimizador `Adam`
- 50 épocas de treino

#### Inferência

- Função para prever a próxima palavra com base em uma frase de entrada

---

### Bibliotecas Utilizadas

- `torch`
- `transformers` (HuggingFace)
- `torch.nn`, `torch.utils.data`

---

### Objetivo

Demonstrar, de forma prática e acessível, como funcionam os fundamentos de um modelo de linguagem neural simples — desde o pré-processamento até o treinamento e a geração de texto.
