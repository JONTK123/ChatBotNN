# ==============================================================#
#  Mini-NN API
#  --------------------------------------------------------------#
#  • Treina uma mini-rede neural (prevê o próximo token)         #
#  • Faz streaming das métricas por WebSocket                    #
#  • Gera PNGs e os serve em /static/…                           #
# ==============================================================#

# ------------------------------#
#  Imports
# ------------------------------#
import json
import pathlib
import shutil
import tempfile
from collections import Counter
from datetime import datetime as dt

import anyio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ------------------------------#
#  Constantes
# ------------------------------#
EPOCHS       = 50
BATCH_SIZE   = 1
LR           = 0.01
DELTA_TOPK   = 10             # quantos pesos delta serão enviados
STATIC_DIR   = pathlib.Path("static_outputs")
STATIC_DIR.mkdir(exist_ok=True)
ENVIAR_TOPK = True
# ENVIAR_TOPK = False

# ------------------------------#
#  Estado do servidor
# ------------------------------#
estado = {"treinando": False, "loss_final": None, "pngs": {}}
subscribers: set[WebSocket] = set()

# ==============================================================#
#  Função de Treinamento
# ==============================================================#
def treinar_rede(frases: list[str], out_dir: pathlib.Path) -> dict:
    """Treina a mini-rede e grava métricas/gráficos no diretório out_dir."""
    # --- Tokenização e pares (entrada → saída) ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    treino_pairs = [
        (ids[:i], ids[i])
        for frase in frases
        for ids in [tokenizer.encode(frase, add_special_tokens=False)]
        for i in range(1, len(ids))
    ]

    # --- Dataset / DataLoader ---
    class ChatDataset(Dataset):
        def __init__(self, pares): self.pares = pares
        def __len__(self):         return len(self.pares)
        def __getitem__(self, i):
            x, y = self.pares[i]
            return {"input_ids": torch.tensor(x), "label": torch.tensor(y)}

    loader = DataLoader(ChatDataset(treino_pairs), batch_size=BATCH_SIZE, shuffle=True)

    # --- Modelo simplificado ---
    class MiniGPT(nn.Module):
        def __init__(self, vocab, emb=32, hid=64):
            super().__init__()
            self.embed = nn.Embedding(vocab, emb)
            self.ff1   = nn.Linear(emb, hid)
            self.out   = nn.Linear(hid, vocab)

        def forward(self, ids):
            x = self.embed(ids).mean(dim=1)
            return self.out(F.relu(self.ff1(x)))

    model  = MiniGPT(tokenizer.vocab_size)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.CrossEntropyLoss()

    # --- Inicializa pesos anteriores para deltas ---
    with torch.no_grad():
        pesos_anteriores = model.embed.weight.flatten().clone()

    # --- Históricos de métricas ---
    hist_loss, hist_acc, hist_ppl = [], [], []
    hist_prec, hist_rec, hist_f1  = [], [], []
    loss_batches: list[list[float]] = []
    erros_tokens: Counter = Counter()

    logs_epocas = []

    # --- Loop de épocas ---
    for ep in range(EPOCHS):
        tot_loss, tot_ok = 0.0, 0
        batch_losses, y_true, y_pred = [], [], []

        for step, batch in enumerate(loader, 1):
            ids, label = batch["input_ids"], batch["label"]

            # forward / backward
            logits = model(ids)
            loss   = lossfn(logits, label)
            optim.zero_grad(); loss.backward(); optim.step()

            # métricas de batch
            tot_loss += loss.item()
            pred = torch.argmax(logits, 1)
            tot_ok += (pred == label).sum().item()
            batch_losses.append(loss.item())

            # coleta para métricas globais
            y_true.extend(label.tolist())
            y_pred.extend(pred.tolist())
            for p, t in zip(pred, label):
                if p != t:
                    erros_tokens[int(t)] += 1

            # --- Calcula delta de pesos e envia ao front ---
            with torch.no_grad():
                pesos_atuais = model.embed.weight.flatten()

                if ENVIAR_TOPK:
                    diferencas = (pesos_atuais - pesos_anteriores).abs()
                    topk = torch.topk(diferencas, k=min(DELTA_TOPK, diferencas.numel()))
                    pesos_delta = [
                        (int(i), float(pesos_anteriores[i]), float(pesos_atuais[i]))
                        for i in topk.indices
                    ]
                else:
                    pesos_delta = [
                        (i, float(pesos_anteriores[i]), float(pesos_atuais[i]))
                        for i in range(len(pesos_atuais))
                        if float(pesos_anteriores[i]) != float(pesos_atuais[i])
                    ]

                pesos_anteriores = pesos_atuais.clone()

            _broadcast_progress({
                "epoca":        ep,
                "batch":        step,
                "loss":         loss.item(),
                "weights_delta": pesos_delta
            })

        # métricas ao fim da época
        hist_loss.append(tot_loss)
        hist_acc .append(tot_ok / len(loader))
        hist_ppl .append(np.exp(tot_loss / len(loader)))
        loss_batches.append(batch_losses)

        hist_prec.append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        hist_rec .append(recall_score  (y_true, y_pred, average="macro", zero_division=0))
        hist_f1  .append(f1_score      (y_true, y_pred, average="macro", zero_division=0))

        if ep % 10 == 0:
            log_msg = f"[{dt.now():%H:%M:%S}] época {ep} • loss {tot_loss:.4f}"
            print(log_msg)
            logs_epocas.append(log_msg)

    # --- Geração de PNGs de todas as métricas ---
    def save(fig_name: str):
        plt.savefig(out_dir / fig_name)
        plt.close()

    # Loss / Acc / PPL por época
    for serie, title, nome in [
        (hist_loss, "Loss",          "grafico_loss_epoca.png"),
        (hist_acc , "Acurácia",      "grafico_acuracia.png"),
        (hist_ppl , "Perplexidade",  "grafico_perplexidade.png"),
    ]:
        plt.figure(); plt.plot(serie)
        plt.title(f"{title} por época"); plt.xlabel("Época"); plt.grid(True)
        save(nome)

    # Precision / Recall / F1
    plt.figure()
    plt.plot(hist_prec, label="Precision")
    plt.plot(hist_rec , label="Recall")
    plt.plot(hist_f1  , label="F1-Score")
    plt.legend(); plt.grid(True)
    plt.title("Precision / Recall / F1 por época")
    save("grafico_prf1.png")

    # Top-10 tokens com mais erros
    tok, val = zip(*erros_tokens.most_common(10)) if erros_tokens else ([], [])
    plt.figure(); plt.bar([tokenizer.decode([i]) for i in tok], val)
    plt.title("Top-10 tokens com mais erros"); plt.xticks(rotation=45); plt.ylabel("Erros")
    save("grafico_erros.png")

    # Loss por batch em 3D
    max_b = max(len(l) for l in loss_batches)
    X, Y  = np.meshgrid(range(EPOCHS), range(max_b))
    Z     = np.zeros_like(X, dtype=float)
    for e, lista in enumerate(loss_batches):
        Z[:len(lista), e] = lista
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Época"); ax.set_ylabel("Batch"); ax.set_zlabel("Loss")
    save("grafico_mapa3d.png")

    # Matriz de Confusão normalizada
    cm   = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="viridis", ax=ax, values_format=".2f")
    plt.title("Matriz de Confusão (normalizada)")
    plt.xticks(rotation=90)
    save("grafico_confusao.png")

    # copia todos os PNGs para a pasta estática
    pngs_dict = {}
    for img in out_dir.glob("*.png"):
        dst = STATIC_DIR / img.name
        shutil.copyfile(img, dst)
        pngs_dict[img.stem] = f"/static/{dst.name}"

    loss_inicial = hist_loss[0]
    loss_final = hist_loss[-1]

    return {
        "loss_inicial": loss_inicial,
        "loss_final": loss_final,
        "logs": logs_epocas,
        "pngs": pngs_dict
    }

    # ==============================================================#
#  FastAPI setup
# ==============================================================#
app = FastAPI(title="Mini-NN API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve PNGs em /static/…
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# broadcast utilitário
def _broadcast_progress(payload: dict):
    txt = json.dumps(payload, default=float)
    for ws in set(subscribers):
        try:
            anyio.from_thread.run(ws.send_text, txt)
        except RuntimeError:
            subscribers.discard(ws)

# WebSocket endpoint
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        subscribers.discard(ws)

# input schema
class FrasesInput(BaseModel):
    frases: list[str]

# background task
def _treino_bg(frases: list[str]):
    tmp = pathlib.Path(tempfile.mkdtemp())
    estado["treinando"] = True
    resultado = treinar_rede(frases, tmp)
    estado.update(resultado)
    estado["treinando"] = False
    _broadcast_progress({"done": True, **resultado})

# rota de início de treino
@app.post("/treinar")
def treinar(req: FrasesInput, bg: BackgroundTasks):
    if estado["treinando"]:
        return {"msg": "Já existe treino em andamento."}
    bg.add_task(_treino_bg, req.frases)
    return {"msg": "Treino iniciado!"}

# rota de status
@app.get("/status")
def status():
    return estado

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
