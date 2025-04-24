const API = "http://127.0.0.1:8000";
let usar2Camadas = false;

// Botão “Treinar!”
document.getElementById("bt").onclick = async () => {
  const frases = document.getElementById("txt").value
    .split("\n")
    .filter(l => l.trim());
  usar2Camadas = document.getElementById("modo2camadas").checked;

  if (frases.length < 5) {
    alert("Digite pelo menos 5 linhas");
    return;
  }

  // UI
  document.getElementById("loading").style.display = "block";
  document.getElementById("bt").style.display = "none";
  document.getElementById("status").textContent = "{}";
  document.getElementById("finalImgs").innerHTML = "";

  // Reconstrói o grafo
  cy.elements().remove();
  cy.add(buildMiniGraph());
  cy.layout({ name: "preset" }).run();

  await fetch(API + "/treinar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frases, usar2Camadas })
  });
};

// WebSocket – stream ao vivo
let ws;
connectStream();
function connectStream() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");
  ws.onopen = () => console.log("✅ WS conectado");
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.loss      !== undefined) updateCharts(msg);
    if (msg.weights_delta)              updateGraph(msg);
    if (msg.activations)                updateActivations(msg.activations);
    if (msg.done)                       mostrarResumoFinal(msg);
    showStatus(msg);
  };
  ws.onclose = () => setTimeout(connectStream,2000);
}

// Plotly – perda ao vivo
Plotly.newPlot("liveLoss", [{ x:[], y:[], mode:"lines", name:"loss" }], {
  margin:{t:30},
  xaxis:{title:"Passo (batches)"},
  yaxis:{title:"Loss"}
});
let passo = 0;
function updateCharts({ loss }) {
  passo++;
  Plotly.extendTraces("liveLoss", { x:[[passo]], y:[[loss]] }, [0], 500);
}

// Cytoscape – visual da rede
const cy = cytoscape({
  container: document.querySelector("#liveNet"),
  elements: [],
  layout: { name: "preset" },
  style: [
    {
      selector: "node",
      style: {
        "background-color":"#333",
        "border-width":2,
        "border-color":"#aaa",
        label:"data(id)",
        color:"#fff",
        "text-valign":"center",
        "font-size":14,
        width:55,
        height:55
      }
    },
    {
      selector: "edge",
      style: {
        width:2,
        "line-color":"#888",
        "curve-style":"bezier"
      }
    }
  ]
});

function buildMiniGraph() {
  const els = [], totalHidden=64, y0=20, dy=15;
  // Entrada
  els.push({ data:{id:"E"}, position:{x:50,y:300} });
  // Camada 1
  for(let i=0;i<totalHidden;i++){
    const y=y0+i*dy;
    els.push({ data:{id:`H1_${i}`}, position:{x:200,y} });
    els.push({ data:{id:`eh${i}`, source:"E", target:`H1_${i}`}});
  }
  if(usar2Camadas){
    // Camada 2
    for(let i=0;i<totalHidden;i++){
      const y=y0+i*dy;
      els.push({ data:{id:`H2_${i}`}, position:{x:350,y} });
      els.push({ data:{id:`h1_${i}_h2`, source:`H1_${i}`, target:`H2_${i}`}});
      els.push({ data:{id:`h2_${i}_o`, source:`H2_${i}`, target:"Out"}});
    }
  } else {
    // Saída direta
    for(let i=0;i<totalHidden;i++){
      els.push({ data:{id:`h${i}o`, source:`H1_${i}`, target:"Out"}});
    }
  }
  // Saída
  els.push({ data:{id:"Out"}, position:{x:usar2Camadas?500:400,y:300} });
  return els;
}

// Atualiza cores e espessuras das arestas
function updateGraph({ weights_delta }) {
  weights_delta.forEach(([idx,w0,w1]) => {
    const e = cy.edges()[idx];
    if(!e) return;
    const delta = w1 - w0;
    const color = delta>0 ? "red" : "dodgerblue";
    const width = 2 + 8 * Math.min(1, Math.abs(delta));
    e.style({ "line-color": color, width });
  });
}

// Atualiza ativação dos nós
function updateActivations(act) {
  act.hid1?.forEach((v,i)=>{
    const n=cy.getElementById(`H1_${i}`);
    if(n){ n.style("background-color", calcularCor(v)); n.style("label",v.toFixed(2)); }
  });
  act.hid2?.forEach((v,i)=>{
    const n=cy.getElementById(`H2_${i}`);
    if(n){ n.style("background-color", calcularCor(v)); n.style("label",v.toFixed(2)); }
  });
}

// escala de cor azul→vermelho
function calcularCor(valor) {
  const escala = Math.tanh(valor);  // Normaliza para [-1, 1]
  const abs = Math.abs(escala);
  const intensidade = Math.round(abs * 255);

  if (escala > 0) {
    // positivo → azul
    return `rgb(${255 - intensidade}, ${255 - intensidade}, 255)`;
  } else if (escala < 0) {
    // negativo → vermelho
    return `rgb(255, ${255 - intensidade}, ${255 - intensidade})`;
  } else {
    // zero → cinza claro
    return 'rgb(200, 200, 200)';
  }
}

// exibe JSON status
function showStatus(msg){
  const {pngs,logs,...rest}=msg;
  document.getElementById("status").textContent=JSON.stringify(rest,null,2);
}

// ao final: resumo + troca botão + mostra PNGs
function mostrarResumoFinal(msg){
  const info = document.getElementById("treinoInfo")||document.createElement("div");
  info.id="treinoInfo";
  info.style="margin:1em 0;font-size:1rem;color:#ddd";
  info.innerHTML=`
    ✅ <b>Treinamento finalizado!</b><br>
    Loss inicial: <code>${msg.loss_inicial.toFixed(4)}</code><br>
    Loss final:   <code>${msg.loss_final.toFixed(4)}</code>
  `;
  document.querySelector("#liveLoss").before(info);

  // reutiliza mesmo lugar do botão
  const old = document.getElementById("bt");
  if(old){
    const p = old.parentElement;
    const pos = Array.from(p.children).indexOf(old);
    old.remove();
    const btn = document.createElement("button");
    btn.id="bt"; btn.textContent="🔄 Reiniciar";
    btn.onclick=()=>location.reload();
    p.insertBefore(btn,p.children[pos]||null);
  }

  if(msg.pngs) showPngs(msg.pngs);
  document.getElementById("loading").style.display="none";
}

// mostra PNGs finais
function showPngs(pngs){
  const div=document.getElementById("finalImgs");
  div.innerHTML="";
  for(const [k,u] of Object.entries(pngs)){
    const t={
      loss_epoca:"📉 Loss por Época",
      acuracia:"✅ Acurácia",
      perplexidade:"🧠 Perplexidade",
      prf1:"🎯 Precision/Recall/F1",
      erros:"❌ Top-10 Erros",
      mapa3d:"🌌 Mapa 3D",
      confusao:"🧮 Matriz de Confusão"
    }[k]||k;
    const h=document.createElement("h3");
    h.textContent=t; h.style.color="#fff";
    const img=document.createElement("img");
    img.src=API+u; img.alt=k;
    div.appendChild(h);
    div.appendChild(img);
  }
}
