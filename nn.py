# DataSet simples

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

# Tokenizando o dataset ( sem biblioteca )
# ----------------------------------------------------------------------------------------

# Junta frases em uma unica string e separa as palavras
palavras = set(" ".join(frases).split())

# Cria dicionario de palavras para indices, chaves sao as palavras e valores sao os indices
word2id = {w:i for i, w in enumerate(palavras)}
#ou
# for i, w in enumerate(palavras):
#     word2id[w] = i

# Cria dicionario de indices para palavras, chaves sao os indices e valores sao as palavras
id2word = {i:w for w, i in word2id.items()}
# ou
# for w, i in word2id.items():
#   id2word[i] = w

# 1. Esse código:
# word2id = {w: i for i, w in enumerate(palavras)}
# Faz o seguinte:
# Cria um dicionário
# Onde:
# w é a chave (a palavra)
# i é o valor (o índice)
# Ele percorre a lista palavras com enumerate, ou seja, tem acesso tanto ao índice quanto à palavra.

# Esse código:
# id2word = {i: w for w, i in word2id.items()}
# Faz o inverso:
# Cria um novo dicionário
# Onde:
# i é a chave (o índice)
# w é o valor (a palavra)
# Ele percorre o dicionário word2id, usando .items() — que dá acesso a pares (chave, valor) → (w, i)

# Usamos enumerate para ter acesso ao index e seu conteudo do array de strings
# Exemplo array = ["oi", "queijo"] [1,2] -> i=1 -> "oi" e i=2 -> "queijo"
# Podemos tb fazer um for:
# for i in range(len(palavras)):
#     print(i, palavras[i])

# e isso é equivalente a:

# for (int i = 0; i < palavras.length; i++) {
#     System.out.println("Índice: " + i + " Palavra: " + palavras[i]);
# }

# em java

# Para iterar somente sobre o resultado podemos fazer, em python:
# for palavra in palavras:
#     print(palavra)

# e se quiser em java:
# for (String palavra: palavras) {
#     System.out.println(palavra);

print(f"Palavras: {palavras}")
print(f"Word2ID: {word2id}")
print(f"ID2Word: {id2word}")
print("\n")

# Tokeinzando o dataset ( com biblioteca )
# ----------------------------------------------------------------------------------------
from transformers import AutoTokenizer

# Usando tokenizer do GPT-2 (diversos bancos de tokens disponiveis)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

for i, frase in enumerate(frases):
    # Tokeniza a frase
    tokens = tokenizer.tokenize(frase)

    # Converte os tokens para ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Converte os ids para tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Converte os ids para string
    frase_tokenizada = tokenizer.decode(token_ids)
    print(f"Frase: {frase}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Frase tokenizada: {frase_tokenizada}")
    print("\n")

# Tokenizacao-dupla dos dados ( sem biblioteca )
# ----------------------------------------------------------------------------------------

treino = []

for frase in frases:
    palavras_frase = frase.split() # Separa as palavras da frase com " " por padrao
    for i in range(1, len(palavras_frase)): # Array de subtrings
        entrada = palavras_frase[:i] # Todas palavras antes de i
        saida = palavras_frase[i] # Ultima palavra, i ( proxima )

        # Converte as palavras para tokens
        entrada_ids = [word2id[w] for w in entrada] # Encontra o token da palavra
        saida_id = word2id[saida] # Encontra palavra do token

        treino.append((entrada_ids, saida_id)) # Arrays para prever proxima palavra com base nas ultimas

print(f"Treino: {treino}")

# Tokenizacao-dupla dos dados ( com biblioteca )
# ----------------------------------------------------------------------------------------

treino2 = []

for frase in frases:
    # Separa a frase em palavras e ja da token
    token_ids = tokenizer.encode(frase, add_special_tokens=False)

    # Gera pares (entrada parcial → próximo token)
    for i in range(1, len(token_ids)):
        entrada_ids = token_ids[:i]
        saida_id = token_ids[i]
        treino2.append((entrada_ids, saida_id))

print(f"Treino: {treino2}")


# Exemplo

# # Iteração 1
# i = 1
# entrada = palavras_frase[:i]  # ["o"]
# saida = palavras_frase[i]     # "hotel"
# entrada_ids = [word2id[w] for w in entrada]  # [id de "o"]
# saida_id = word2id[saida]  # id de "hotel"
# treino.append((entrada_ids, saida_id))
#
# # Iteração 2
# i = 2
# entrada = palavras_frase[:i]  # ["o", "hotel"]
# saida = palavras_frase[i]     # "oferece"
# entrada_ids = [word2id[w] for w in entrada]  # [id de "o", id de "hotel"]
# saida_id = word2id[saida]  # id de "oferece"
# treino.append((entrada_ids, saida_id))

# Criando rede neural e preparando dados
# ----------------------------------------------------------------------------------------

from torch.utils.data import Dataset

# Transforma cada par em tensores PyTorch
class ChatDataset(Dataset):
    def _init_(self, dados):
        self.dados = dados  # recebe uma lista de pares (entrada_ids, saida_id)

    def _len_(self):
        return len(self.dados)  # número total de exemplos

    # Retorna o proximo token com base nos anteriores ( Uma funcao faz isso ) com base no dataSet fornecido, no caso Treinos1 e 2
    def _getitem_(self, idx):
        # torch.tensor(..., dtype=torch.long) -> Isso acontece internamente (biblioteca). Resumindo transforma os tokens de entrada e saida para o tipo tensor para o modelo usar
        entrada, saida = self.dados[idx]
        return {
            "input_ids": torch.tensor(entrada, dtype=torch.long),
            "label": torch.tensor(saida, dtype=torch.long)
        }

# Transforma seus dados (pares entrada → saída) num formato que o PyTorch entende
# Permite que o modelo leia 1 exemplo por vez (ou em batches)

# Biblioteca PyTorch é um conjunto de códigos prontos e reutilizáveis que te ajudam a:
# Criar redes neurais
# Fazer treino automático
# Calcular gradientes
# Processar dados (dataset, loader, etc.)
# Usar GPU se disponível

# Sem biblioteca, você teria que:
# Escrever manualmente as operações de álgebra linear (somas, multiplicações)
# Implementar derivadas (cálculo!) para o backpropagation
# Criar camadas, otimizadores e funções de

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def _init_(self, vocab_size, embed_dim=32, hidden_dim=64):
        super()._init_()
        self.embed = nn.Embedding(vocab_size, embed_dim) # Transforma cada token em um vetor de ( nesse caso ) 32 dimensões
        self.ff1 = nn.Linear(embed_dim, hidden_dim) # vetor -> apresentacao abstrata
        self.out = nn.Linear(hidden_dim, vocab_size) # Pontuacao com base na entrada

    def forward(self, input_ids):
        emb = self.embed(input_ids)              # [batch, seq_len, embed_dim]
        x = emb.mean(dim=1)                      # média dos embeddings, faz uma media vetorial entre todos vetores, resume frase inteira como um vetor so
        x = F.relu(self.ff1(x))                  # camada oculta
        logits = self.out(x)                     # saída final (pontuação por token)
        return logits

# Treinando
# ----------------------------------------------------------------------------------------

# Dataset e carregador de dados
dataset2 = ChatDataset(treino2)
# dataset1 = ChatDataset(treino1)

loader = DataLoader(dataset2, batch_size=1, shuffle=True)

# Inicializa o modelo
vocab_size = tokenizer.vocab_size  # pega o tamanho do vocabulário usado
modelo = MiniGPT(vocab_size)

# Função de erro e otimizador
loss_fn = nn.CrossEntropyLoss()  # compara previsões com o rótulo correto
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)

# Treinamento
for epoca in range(50):  # número de vezes que o modelo vê todos os dados
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"]     # sequência de entrada
        label = batch["label"]             # próximo token esperado

        logits = modelo(input_ids)         # saída do modelo (pontuações)
        loss = loss_fn(logits, label)      # calcula o erro

        optimizer.zero_grad()              # limpa os gradientes anteriores
        loss.backward()                    # faz backpropagation
        optimizer.step()                   # atualiza os pesos do modelo

        total_loss += loss.item()

    if epoca % 10 == 0:
        print(f"\nÉpoca {epoca} - Perda: {total_loss:.4f}")

# testar geração de texto com o modelo treinado:
# Função para prever o próximo token baseado em uma frase de entrada
def gerar_proxima_palavra(frase):
    modelo.eval()  # modo de avaliação (desativa dropout, etc.)

    # Tokeniza a frase de entrada
    input_ids = tokenizer.encode(frase, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        logits = modelo(input_ids)
        predicted_id = torch.argmax(logits, dim=1).item()

    # Converte ID de volta pra palavra
    predicted_token = tokenizer.decode([predicted_id])
    print(f"Frase: '{frase}' → Próxima palavra prevista: '{predicted_token}'")

gerar_proxima_palavra("o hotel")
gerar_proxima_palavra("o check-in")
gerar_proxima_palavra("a recepção")
gerar_proxima_palavra("o hotel tem")
