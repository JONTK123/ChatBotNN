/* =========================================================
 *  Mini-NN – front-end
 *  --------------------------------------------------------
 *  • Envia frases para treino (POST /treinar)
 *  • Recebe métricas em tempo-real por WebSocket (/ws)
 *  • Desenha:
 *      – gráfico de loss (Plotly)
 *      – mudanças de pesos (Cytoscape)
 *      – PNGs finais
 *      – resumo com loss inicial / final
 * ======================================================= */

const API = 'http://127.0.0.1:8000';

/* -------------------------------------------------------
 *  1. Botão “Treinar!”
 * ----------------------------------------------------- */
document.getElementById('bt').onclick = async () => {
  const frases = document
    .getElementById('txt')
    .value.split('\n')
    .filter(l => l.trim());

  if (frases.length < 5) {
    alert('Digite pelo menos 5 linhas');
    return;
  }

  // UI
  document.getElementById('loading').style.display = 'block';
  document.getElementById('bt').style.display = 'none';
  document.getElementById('finalImgs').innerHTML = '';
  document.getElementById('status').textContent = '{}';

  await fetch(API + '/treinar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frases }),
  });
};

/* -------------------------------------------------------
 * 2. WebSocket (stream ao vivo)
 * ----------------------------------------------------- */
let ws;
connectStream();

function connectStream() {
  ws = new WebSocket('ws://127.0.0.1:8000/ws');

  ws.onopen = () => console.log('✅ WebSocket conectado');

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    console.log('📨', msg);

    if (msg.loss !== undefined) updateCharts(msg);
    if (msg.weights_delta) updateGraph(msg);
    if (msg.done) mostrarResumoFinal(msg);

    showStatus(msg);
  };

  ws.onclose = () => {
    console.log('🔌 WebSocket fechado – reconectando…');
    setTimeout(connectStream, 2_000);
  };
}

/* -------------------------------------------------------
 * 3.  Resumo final + botão reiniciar + PNGs
 * ----------------------------------------------------- */
function mostrarResumoFinal(msg) {
  /* --- resumo de perdas --- */
  const info =
    document.getElementById('treinoInfo') || document.createElement('div');
  info.id = 'treinoInfo';
  info.style.margin = '1em 0';
  info.style.fontSize = '16px';
  info.innerHTML = `
    ✅ <b>Treinamento finalizado!</b><br>
    Loss inicial: <code>${msg.loss_inicial.toFixed(4)}</code><br>
    Loss final:   <code>${msg.loss_final.toFixed(4)}</code>
  `;
  document
    .getElementById('liveLoss')
    .parentElement.insertBefore(info, document.getElementById('liveLoss'));

  /* --- exibe PNGs --- */
  if (msg.pngs) showPngs(msg.pngs);

  /* --- troca botão --- */
  const antigo = document.getElementById('bt');
  if (antigo) {
    const pai = antigo.parentElement;
    antigo.remove();

    const reiniciar = document.createElement('button');
    reiniciar.id = 'bt-reiniciar';
    reiniciar.textContent = '🔄 Reiniciar';
    reiniciar.onclick = () => location.reload();
    pai.appendChild(reiniciar);
  }

  document.getElementById('loading').style.display = 'none';
}

/* -------------------------------------------------------
 * 4. Plotly – gráfico de loss ao vivo
 * ----------------------------------------------------- */
const trace = { x: [], y: [], mode: 'lines', name: 'loss' };

Plotly.newPlot('liveLoss', [trace], {
  margin: { t: 30 },
  xaxis: { title: 'batches' },
  yaxis: { title: 'loss' },
});

let passo = 0;
function updateCharts({ loss }) {
  passo++;
  Plotly.extendTraces(
    'liveLoss',
    { x: [[passo]], y: [[loss]] },
    [0],
    500 /* keep last 500 points */
  );
}

/* -------------------------------------------------------
 * 5. Cytoscape – mini-rede + deltas de pesos
 * ----------------------------------------------------- */
const cy = cytoscape({
  container: document.querySelector('#liveNet'),
  elements: buildMiniGraph(),
  layout: { name: 'grid' },
  style: [
    { selector: 'edge', style: { width: 2, 'line-color': '#888' } },
    {
      selector: 'node',
      style: {
        'background-color': '#222',
        label: 'data(id)',
        color: '#fff',
        'font-size': 8,
        'text-valign': 'center',
      },
    },
  ],
});

function buildMiniGraph() {
  const els = [{ data: { id: 'E' } }];
  for (let i = 0; i < 4; i++) {
    els.push({ data: { id: 'H' + i } });
    els.push({ data: { id: 'eh' + i, source: 'E', target: 'H' + i } });
    els.push({ data: { id: 'h' + i + 'o', source: 'H' + i, target: 'Out' } });
  }
  els.push({ data: { id: 'Out' } });
  return els;
}

function updateGraph({ weights_delta }) {
  weights_delta.forEach(([idx, oldW, newW]) => {
    const e = cy.edges()[idx];
    if (!e) return;
    const v = Math.tanh(newW - oldW);
    e.style('width', 1 + 5 * Math.abs(v));
    e.style('line-color', v > 0 ? 'red' : 'blue');
  });
}

/* -------------------------------------------------------
 * 6.  JSON “status” cru   (ESCONDE pngs e logs)
 * ----------------------------------------------------- */
function showStatus(msg) {
  const { pngs, logs, ...visivel } = msg;
  document.getElementById('status').textContent =
    JSON.stringify(visivel, null, 2);
}

/* -------------------------------------------------------
 * 7.  PNGs finais (opcional)
 * ----------------------------------------------------- */
function showPngs(pngs) {
  const nomes = {
    loss_epoca: '📉 Loss por Época',
    acuracia: '✅ Acurácia por Época',
    perplexidade: '🧠 Perplexidade por Época',
    prf1: '🎯 Precision / Recall / F1',
    erros: '❌ Top-10 Tokens com Erros',
    mapa3d: '🌌 Mapa 3D (Época × Batch)',
    confusao: '🧮 Matriz de Confusão',
  };

  const div = document.getElementById('finalImgs');
  div.innerHTML = '';

  for (const [nome, url] of Object.entries(pngs)) {
    const h3 = document.createElement('h3');
    h3.textContent = nomes[nome.replace('grafico_', '')] || nome;
    const img = document.createElement('img');
    img.src = API + url;
    img.alt = nome;
    div.appendChild(h3);
    div.appendChild(img);
  }
}
