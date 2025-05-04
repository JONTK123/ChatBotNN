const API = "http://127.0.0.1:8000";  // Verifique se este é o endpoint correto
let usar2Camadas = false;
let tokensMostrados = false;

// Ação ao clicar no botão
document.getElementById("bt").onclick = async () => {
  const frases = document.getElementById("txt").value
    .split("\n")
    .filter(l => l.trim());

  usar2Camadas = document.getElementById("modo2camadas").checked;

  if (frases.length < 5) {
    alert("Digite pelo menos 5 frases");
    return;
  }

  if (!tokensMostrados) {
    // PRIMEIRO CLIQUE: mostrar tokens
    const tokenPreview = await fetch(API + "/tokenizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(frases)
    });

    const tokenData = await tokenPreview.json();
    const tokenBox = document.getElementById("tokensDebug");
    tokenBox.innerHTML = "";

    tokenData.forEach(({ frase, tokens_simples, tokens_transformer }, i) => {
      tokenBox.innerHTML += `
        <div style="text-align:left; margin-bottom:1em;">
          <b>Frase ${i + 1}:</b> <code>${frase}</code><br>
          <b>Tokens (split):</b> <code>${tokens_simples.join(" | ")}</code><br>
          <b>Tokens (transformers):</b> <code>${tokens_transformer.join(" | ")}</code>
        </div>
      `;
    });

    document.getElementById("bt").textContent = "🚀 Iniciar Treinamento";
    tokensMostrados = true;
    return;
  }

  // SEGUNDO CLIQUE: inicia o treino
  document.getElementById("loading").style.display = "block";
  document.getElementById("bt").style.display = "none";
  document.getElementById("status").textContent = "{}";
  document.getElementById("finalImgs").innerHTML = "";
  document.getElementById("treinoInfo")?.remove();

  cy.elements().remove();
  cy.add(buildMiniGraph());
  cy.layout({ name: "preset" }).run();

  // Inicia o treinamento
  await fetch(API + "/treinar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frases, usar2Camadas })
  });
};

// WebSocket – stream de atualizações do treinamento
let ws;
connectStream();
function connectStream() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");  // Garantir que a URL seja correta
  ws.onopen = () => console.log("✅ WS conectado");
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.loss !== undefined) updateCharts(msg);
    if (msg.weights_delta) updateGraph(msg);
    if (msg.activations) updateActivations(msg.activations);
    if (msg.done) mostrarResumoFinal(msg);
    showStatus(msg);
  };
  ws.onclose = () => setTimeout(connectStream, 2000);  // Tenta reconectar em caso de desconexão
}

// Plotly – gráfico de loss
window.onload = () => {
  Plotly.newPlot("liveLoss", [{
    x: [], y: [], mode: "lines", name: "Loss", line: { color: "#4cc9f0" }
  }], {
    margin: { t: 30 },
    paper_bgcolor: "#1a1a1a",
    plot_bgcolor: "#1a1a1a",
    xaxis: { title: "Passo (batches)", color: "#ccc" },
    yaxis: { title: "Loss", color: "#ccc" }
  });
};

// Atualiza gráfico de loss durante o treinamento
let passo = 0;
function updateCharts({ loss }) {
  passo++;
  Plotly.extendTraces("liveLoss", {
    x: [[passo]], y: [[loss]]
  }, [0], 500);
}

// Cytoscape – rede neural visual
const cy = cytoscape({
  container: document.querySelector("#liveNet"),
  elements: [],
  layout: { name: "preset" },
  style: [
    {
      selector: "node",
      style: {
        "background-color": "#333",
        "border-width": 2,
        "border-color": "#aaa",
        "label": "data(id)",
        "color": "#fff",
        "text-outline-color": "#000",
        "text-outline-width": 2,
        "text-valign": "center",
        "font-size": 13,
        "width": 55,
        "height": 55
      }
    },
    {
      selector: 'node[id^="Out_TOP_"]',
      style: {
        "label": "data(label)",
        "width": 90,
        "height": 90,
        "font-size": 16,
        "color": "#fff",
        "text-outline-color": "#000",
        "text-outline-width": 2
      }
    },
    {
      selector: "edge",
      style: {
        "width": 2,
        "line-color": "#888",
        "curve-style": "bezier"
      }
    }
  ]
});

// Função para construir o gráfico da rede
function buildMiniGraph() {
  const els = [], totalHidden = 6, y0 = 20, dy = 80;

  for (let i = 0; i < 4; i++) {
    const y = y0 + i * dy;
    els.push({ data: { id: `In_${i}` }, position: { x: 20, y } });
    els.push({ data: { id: `in_conn_${i}`, source: `In_${i}`, target: `E` } });
  }

  els.push({ data: { id: "E" }, position: { x: 100, y: 150 } });

  for (let i = 0; i < totalHidden; i++) {
    const y = y0 + i * dy;
    els.push({ data: { id: `H1_${i}` }, position: { x: 250, y } });
    els.push({ data: { id: `eh${i}`, source: "E", target: `H1_${i}` } });
  }

  if (usar2Camadas) {
    for (let i = 0; i < totalHidden; i++) {
      const y = y0 + i * dy;
      els.push({ data: { id: `H2_${i}` }, position: { x: 400, y } });
      els.push({ data: { id: `h1_${i}_h2`, source: `H1_${i}`, target: `H2_${i}` } });
      els.push({ data: { id: `h2_${i}_o`, source: `H2_${i}`, target: `Out` } });
    }
  } else {
    for (let i = 0; i < totalHidden; i++) {
      els.push({ data: { id: `h${i}o`, source: `H1_${i}`, target: "Out" } });
    }
  }

  els.push({ data: { id: "Out", label: "Out" }, position: { x: usar2Camadas ? 500 : 400, y: 150 } });

  return els;
}

// Atualiza o gráfico de pesos (delta)
function updateGraph({ weights_delta }) {
  weights_delta.forEach(([idx, w0, w1]) => {
    const edgeId = `eh${idx}`;
    const e = cy.getElementById(edgeId);
    if (!e) return;
    const delta = w1 - w0;
    const color = delta > 0 ? "dodgerblue" : "red";
    const width = 2 + 8 * Math.min(1, Math.abs(delta));
    e.style({ "line-color": color, width });
  });
}

// Atualiza ativações
function updateActivations(act) {
  act.input?.forEach((vetor, i) => {
    const label = vetor.map(x => x.toFixed(2)).join(", ");
    const node = cy.getElementById(`In_${i}`);
    if (node) node.style("label", label);
  });

  act.hid1?.forEach((v, i) => {
    const node = cy.getElementById(`H1_${i}`);
    if (node) {
      node.style("background-color", calcularCor(v));
      node.style("label", v.toFixed(2));
    }
  });

  act.hid2?.forEach((v, i) => {
    const node = cy.getElementById(`H2_${i}`);
    if (node) {
      node.style("background-color", calcularCor(v));
      node.style("label", v.toFixed(2));
    }
  });

  cy.nodes().filter(n => n.id().startsWith("Out_TOP_")).remove();
  act.top_tokens?.forEach(([id, val, txt], i) => {
    const y = 100 + i * 90;
    const nodeId = `Out_TOP_${i}`;
    cy.add([
      { data: { id: nodeId, label: `${txt} (${val.toFixed(2)})` }, position: { x: 620, y } },
      { data: { id: `conn_${nodeId}`, source: "Out", target: nodeId } }
    ]);
  });
}

// Função para calcular cor das ativações
function calcularCor(valor) {
  const escala = Math.tanh(valor);
  const abs = Math.abs(escala);
  const intensidade = Math.round(abs * 255);

  if (escala > 0) return `rgb(${255 - intensidade}, ${255 - intensidade}, 255)`;
  if (escala < 0) return `rgb(255, ${255 - intensidade}, ${255 - intensidade})`;
  return 'rgb(200, 200, 200)';
}

// Mostra status do treinamento
function showStatus(msg) {
  const { pngs, logs, ...rest } = msg;
  document.getElementById("status").textContent = JSON.stringify(rest, null, 2);
}

// Exibe o resumo final
function mostrarResumoFinal(msg) {
  const info = document.getElementById("treinoInfo") || document.createElement("div");
  info.id = "treinoInfo";
  info.style = "margin:1em 0;font-size:1rem;color:#ddd";
  info.innerHTML = `
    ✅ <b>Treinamento finalizado!</b><br>
    Loss inicial: <code>${msg.loss_inicial.toFixed(4)}</code><br>
    Loss final:   <code>${msg.loss_final.toFixed(4)}</code>
  `;
  document.querySelector("#liveLoss").before(info);

  document.getElementById("tokenInfo").style.display = "none";

  if (!document.getElementById("predictBox")) {
    const div = document.createElement("div");
    div.id = "predictBox";
    div.style = "margin:2em 0;text-align:center";

    div.innerHTML = `
      <h2>Teste o modelo</h2>
      <input id="promptInp" placeholder="Comece digitando uma frase do treino…" 
             style="width:60%;max-width:500px;padding:.5em;border-radius:4px;border:1px solid #555;background:#1e1e1e;color:#eee">
      <button id="btnPred" style="margin-left:.5em">Completar</button>
      <div id="predOut" style="margin-top:1em;color:#8be9fd"></div>
    `;
    info.after(div);

    document.getElementById("btnPred").onclick = async () => {
      document.getElementById("predOut").textContent = "";
      const prompt = document.getElementById("promptInp").value.trim();
      if (!prompt) return;
      const r = await fetch(API + "/completar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const { continuacao, erro } = await r.json();
      document.getElementById("predOut").textContent = erro || continuacao;
    };
  }

  if (msg.pngs) showPngs(msg.pngs);

  // Reinicia o botão de treino
  const old = document.getElementById("bt");
  if (old) {
    const p = old.parentElement;
    const pos = Array.from(p.children).indexOf(old);
    old.remove();
    const btn = document.createElement("button");
    btn.id = "bt";
    btn.textContent = "🔄 Reiniciar";
    btn.onclick = () => {
      document.getElementById("tokenInfo").style.display = "block";
      location.reload();
    };
    p.insertBefore(btn, p.children[pos] || null);
  }

  document.getElementById("loading").style.display = "none";
}

// Exibe os gráficos finais
function showPngs(pngs) {
  const div = document.getElementById("finalImgs");
  div.innerHTML = "";

  const titulos = {
    grafico_acuracia:       "Acurácia",
    grafico_confusao:       "Matriz de Confusão",
    grafico_erros:          "Top‑10 Tokens com mais erros",
    grafico_loss_epoca:     "Loss por Época",
    grafico_mapa3d:         "Loss por Batch (3D)",
    grafico_perplexidade:   "Perplexidade",
    grafico_prf1:           "Precision / Recall / F1"
  };

  const legendas = {
    grafico_acuracia:       "Mostra quantas previsões foram corretas. Ideal é próximo de 100%.",
    grafico_confusao:       "Mostra onde a rede mais acerta ou erra. Ideal é a diagonal estar destacada.",
    grafico_erros:          "Tokens que a rede mais errou ao longo do treino. Útil para analisar dificuldades.",
    grafico_loss_epoca:     "Erro total da rede em cada época. Quanto menor, melhor.",
    grafico_mapa3d:         "Valor do loss por batch em todas as épocas. Picos indicam instabilidade.",
    grafico_perplexidade:   "Mede o quão “surpreso” o modelo está. Quanto menor, mais confiante.",
    grafico_prf1:           "Métricas clássicas de classificação. Mostram quão equilibrado está o desempenho."
  };

  for (const [nome, url] of Object.entries(pngs)) {
    const titulo = titulos[nome] || nome;
    const legenda = legendas[nome] || "";

    div.innerHTML += `
      <h3 style="margin-top:2em">${titulo}</h3>
      <img src="${API + url}" alt="${nome}">
      <div class="legend" style="max-width:700px;margin:0 auto 1em;font-size:.95rem;color:#bbb">
        ${legenda}
      </div>
    `;
  }
}
