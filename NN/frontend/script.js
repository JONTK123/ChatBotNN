const API = "http://127.0.0.1:8000";
let usar2Camadas = false;

document.getElementById("bt").onclick = async () => {
  const frases = document.getElementById("txt").value
    .split("\n")
    .filter(l => l.trim());
  usar2Camadas = document.getElementById("modo2camadas").checked;

  if (frases.length < 5) {
    alert("Digite pelo menos 5 linhas");
    return;
  }

  document.getElementById("loading").style.display = "block";
  document.getElementById("bt").style.display = "none";
  document.getElementById("status").textContent = "{}";
  document.getElementById("finalImgs").innerHTML = "";

  cy.elements().remove();
  cy.add(buildMiniGraph());
  cy.layout({ name: "preset" }).run();

  await fetch(API + "/treinar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frases, usar2Camadas })
  });
};

// WebSocket – stream
let ws;
connectStream();
function connectStream() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");
  ws.onopen = () => console.log("✅ WS conectado");
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.loss !== undefined) updateCharts(msg);
    if (msg.weights_delta) updateGraph(msg);
    if (msg.activations) updateActivations(msg.activations);
    if (msg.done) mostrarResumoFinal(msg);
    showStatus(msg);
  };
  ws.onclose = () => setTimeout(connectStream, 2000);
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

let passo = 0;
function updateCharts({ loss }) {
  passo++;
  Plotly.extendTraces("liveLoss", {
    x: [[passo]], y: [[loss]]
  }, [0], 500);
}

// Cytoscape – rede neural
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

function updateGraph({ weights_delta }) {
  weights_delta.forEach(([idx, w0, w1]) => {
    const e = cy.getElementById(`eh${idx}`);
    if (!e) return;
    const delta = w1 - w0;
    const color = delta > 0 ? "red" : "dodgerblue";
    const width = 2 + 8 * Math.min(1, Math.abs(delta));
    e.style({ "line-color": color, width });
  });
}

function updateActivations(act) {
  // Entradas → 4 vetores, formatados
  act.input?.forEach((vetor, i) => {
    const label = vetor.map(x => x.toFixed(2)).join(", ");
    const node = cy.getElementById(`In_${i}`);
    if (node) node.style("label", label);
  });

  // Hidden 1
  act.hid1?.forEach((v, i) => {
    const node = cy.getElementById(`H1_${i}`);
    if (node) {
      node.style("background-color", calcularCor(v));
      node.style("label", v.toFixed(2));
    }
  });

  // Hidden 2
  act.hid2?.forEach((v, i) => {
    const node = cy.getElementById(`H2_${i}`);
    if (node) {
      node.style("background-color", calcularCor(v));
      node.style("label", v.toFixed(2));
    }
  });

  // Saídas
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

function calcularCor(valor) {
  const escala = Math.tanh(valor);
  const abs = Math.abs(escala);
  const intensidade = Math.round(abs * 255);

  if (escala > 0) return `rgb(${255 - intensidade}, ${255 - intensidade}, 255)`;
  if (escala < 0) return `rgb(255, ${255 - intensidade}, ${255 - intensidade})`;
  return 'rgb(200, 200, 200)';
}

function showStatus(msg) {
  const { pngs, logs, ...rest } = msg;
  document.getElementById("status").textContent = JSON.stringify(rest, null, 2);
}

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

  const old = document.getElementById("bt");
  if (old) {
    const p = old.parentElement;
    const pos = Array.from(p.children).indexOf(old);
    old.remove();
    const btn = document.createElement("button");
    btn.id = "bt";
    btn.textContent = "🔄 Reiniciar";
    btn.onclick = () => location.reload();
    p.insertBefore(btn, p.children[pos] || null);
  }

  if (msg.pngs) showPngs(msg.pngs);
  document.getElementById("loading").style.display = "none";
}

function showPngs(pngs) {
  const div = document.getElementById("finalImgs");
  div.innerHTML = "";
  for (const [k, u] of Object.entries(pngs)) {
    const t = {
      loss_epoca: "📉 Loss por Época",
      acuracia: "✅ Acurácia",
      perplexidade: "🧠 Perplexidade",
      prf1: "🎯 Precision/Recall/F1",
      erros: "❌ Top-10 Erros",
      mapa3d: "🌌 Mapa 3D",
      confusao: "🧮 Matriz de Confusão"
    }[k] || k;
    const h = document.createElement("h3");
    h.textContent = t;
    h.style.color = "#fff";
    const img = document.createElement("img");
    img.src = API + u;
    img.alt = k;
    div.appendChild(h);
    div.appendChild(img);
  }
}
