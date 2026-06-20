/* ═══════════════════════════════════════════════════════════
   APP STATE
   ═══════════════════════════════════════════════════════════ */
let dataset = [];          // [{x1, x2, label}]
let currentModel = 'tree';
let trainedTree    = null;
let trainedNeuron  = null;
let isTrained = false;

/* ─── PRESETS ───────────────────────────────────────────── */
const PRESETS = {
  and:  [[0,0,0],[0,1,0],[1,0,0],[1,1,1]],
  or:   [[0,0,0],[0,1,1],[1,0,1],[1,1,1]],
  xor:  [[0,0,0],[0,1,1],[1,0,1],[1,1,0],
         [0.1,0.1,0],[0.9,0.1,1],[0.1,0.9,1],[0.9,0.9,0]],
  iris: [
    [5.1,3.5,0],[4.9,3.0,0],[4.7,3.2,0],[4.6,3.1,0],[5.0,3.6,0],
    [5.4,3.9,0],[4.6,3.4,0],[5.0,3.4,0],[4.4,2.9,0],[4.9,3.1,0],
    [7.0,3.2,1],[6.4,3.2,1],[6.9,3.1,1],[5.5,2.3,1],[6.5,2.8,1],
    [5.7,2.8,1],[6.3,3.3,1],[4.9,2.4,1],[6.6,2.9,1],[5.2,2.7,1],
  ]
};

/* ═══════════════════════════════════════════════════════════
   DATA MANAGEMENT
   ═══════════════════════════════════════════════════════════ */
function addDataPoint() {
  const x1 = parseFloat(document.getElementById('inX1').value);
  const x2 = parseFloat(document.getElementById('inX2').value);
  const label = parseInt(document.getElementById('inClass').value);
  if (isNaN(x1) || isNaN(x2)) { toast('Ingresa valores numéricos válidos', 'error'); return; }
  dataset.push({ x1, x2, label });
  document.getElementById('inX1').value = '';
  document.getElementById('inX2').value = '';
  refreshTable();
  resetTrained();
  toast(`Dato (${x1}, ${x2}) → Clase ${label} agregado`, 'success');
}

function removePoint(idx) {
  dataset.splice(idx, 1);
  refreshTable();
  resetTrained();
}

function clearData() {
  dataset = [];
  refreshTable();
  resetTrained();
  toast('Datos eliminados', 'info');
}

function loadPreset(name) {
  const rows = PRESETS[name];
  dataset = rows.map(([x1, x2, label]) => ({ x1, x2, label }));
  refreshTable();
  resetTrained();
  toast(`Preset "${name.toUpperCase()}" cargado (${dataset.length} puntos)`, 'info');
}

function refreshTable() {
  const body = document.getElementById('dataBody');
  const empty = document.getElementById('emptyState');
  const count = document.getElementById('dataCount');
  count.textContent = dataset.length;

  if (!dataset.length) {
    body.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  body.innerHTML = dataset.map((d, i) => `
    <tr>
      <td class="font-mono">${d.x1}</td>
      <td class="font-mono">${d.x2}</td>
      <td class="${d.label === 0 ? 'class-0' : 'class-1'}">${d.label}</td>
      <td><button class="del-btn" onclick="removePoint(${i})">✕</button></td>
    </tr>
  `).join('');
}

/* ═══════════════════════════════════════════════════════════
   MODEL SWITCH
   ═══════════════════════════════════════════════════════════ */
function switchModel(m) {
  currentModel = m;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.model === m));
  if (m === 'tree') {
    document.getElementById('pageTitle').textContent  = '🌲 Árbol de Decisión';
    document.getElementById('modelBadge').textContent = 'CART — Gini';
  } else {
    document.getElementById('pageTitle').textContent  = '🧠 Neurona (Perceptrón)';
    document.getElementById('modelBadge').textContent = 'Sigmoid + SGD';
  }
}

/* ═══════════════════════════════════════════════════════════
   TRAINING
   ═══════════════════════════════════════════════════════════ */
function trainModel() {
  if (dataset.length < 4) { toast('Necesitas al menos 4 datos para entrenar', 'error'); return; }

  const X = dataset.map(d => [d.x1, d.x2]);
  const y = dataset.map(d => d.label);

  // Normalize
  const mins = [Math.min(...X.map(r => r[0])), Math.min(...X.map(r => r[1]))];
  const maxs = [Math.max(...X.map(r => r[0])), Math.max(...X.map(r => r[1]))];
  const norm = X.map(r => r.map((v, j) => {
    const range = maxs[j] - mins[j];
    return range === 0 ? 0 : (v - mins[j]) / range;
  }));

  window._normParams = { mins, maxs };

  // ── Train Decision Tree ──
  const depth = parseInt(document.getElementById('maxDepth').value);
  trainedTree = new DecisionTree(depth);
  trainedTree.fit(norm, y);
  const treeAcc = trainedTree.accuracy(norm, y);

  document.getElementById('treeAcc').textContent   = (treeAcc * 100).toFixed(1) + '%';
  document.getElementById('treeNodes').textContent = trainedTree.nodeCount;
  document.getElementById('treeDepth').textContent = trainedTree.treeDepth;
  document.getElementById('treeMetrics').style.display = 'flex';
  document.getElementById('treeIdle').style.display    = 'none';
  document.getElementById('treeCanvas').style.display  = 'block';

  drawTree(trainedTree.root);

  // ── Train Perceptron ──
  const epochs = parseInt(document.getElementById('epochs').value);
  const lr     = parseFloat(document.getElementById('lr').value);
  trainedNeuron = new Perceptron(lr, epochs);
  trainedNeuron.fit(norm, y);
  const neuronAcc = trainedNeuron.accuracy(norm, y);

  document.getElementById('neuronAcc').textContent   = (neuronAcc * 100).toFixed(1) + '%';
  document.getElementById('neuronW1').textContent    = trainedNeuron.weights[0].toFixed(3);
  document.getElementById('neuronW2').textContent    = trainedNeuron.weights[1].toFixed(3);
  document.getElementById('neuronBias').textContent  = trainedNeuron.bias.toFixed(3);
  document.getElementById('neuronEpochs').textContent = epochs;
  document.getElementById('neuronMetrics').style.display = 'flex';
  document.getElementById('neuronIdle').style.display    = 'none';
  document.getElementById('neuronCanvas').style.display  = 'block';
  document.getElementById('lossSection').style.display   = 'block';

  drawNeuron(trainedNeuron);
  drawLoss(trainedNeuron.lossHistory);
  drawBoundary();

  // Status
  isTrained = true;
  document.getElementById('statusDot').className    = 'status-dot trained';
  document.getElementById('statusText').textContent = 'Entrenado ✓';
  document.getElementById('predictBtn').disabled    = false;
  document.getElementById('boundaryPanel').style.display = 'block';

  toast(`Modelos entrenados · Árbol: ${(treeAcc*100).toFixed(1)}% · Neurona: ${(neuronAcc*100).toFixed(1)}%`, 'success');
}

function resetTrained() {
  isTrained = false;
  trainedTree = trainedNeuron = null;
  document.getElementById('statusDot').className    = 'status-dot';
  document.getElementById('statusText').textContent = 'Sin entrenar';
  document.getElementById('predictBtn').disabled    = true;
  document.getElementById('predResult').style.display = 'none';
  document.getElementById('treeMetrics').style.display   = 'none';
  document.getElementById('neuronMetrics').style.display = 'none';
  document.getElementById('lossSection').style.display   = 'none';
  document.getElementById('boundaryPanel').style.display = 'none';
  document.getElementById('treeIdle').style.display   = 'flex';
  document.getElementById('neuronIdle').style.display = 'flex';
  document.getElementById('treeCanvas').style.display   = 'none';
  document.getElementById('neuronCanvas').style.display = 'none';
}

/* ═══════════════════════════════════════════════════════════
   PREDICTION
   ═══════════════════════════════════════════════════════════ */
function predict() {
  if (!isTrained) return;
  const x1r = parseFloat(document.getElementById('predX1').value);
  const x2r = parseFloat(document.getElementById('predX2').value);
  if (isNaN(x1r) || isNaN(x2r)) { toast('Ingresa valores numéricos para predecir', 'error'); return; }

  const { mins, maxs } = window._normParams;
  const xn = [
    maxs[0] - mins[0] === 0 ? 0 : (x1r - mins[0]) / (maxs[0] - mins[0]),
    maxs[1] - mins[1] === 0 ? 0 : (x2r - mins[1]) / (maxs[1] - mins[1])
  ];

  const treePred  = trainedTree.predictOne(xn);
  const neuronPred = trainedNeuron.predictOne(xn);
  const prob = trainedNeuron.predictProb(xn);

  const el = document.getElementById('predResult');
  el.style.display = 'block';
  el.innerHTML = `
    <div style="display:flex;gap:20px;flex-wrap:wrap">
      <div>
        <span style="color:var(--muted);font-size:10px">ÁRBOL DE DECISIÓN</span><br>
        <span style="font-size:18px;color:${treePred===1?'var(--teal)':'var(--orange)'}">Clase ${treePred}</span>
      </div>
      <div>
        <span style="color:var(--muted);font-size:10px">NEURONA</span><br>
        <span style="font-size:18px;color:${neuronPred===1?'var(--teal)':'var(--orange)'}">Clase ${neuronPred}</span>
        <span style="font-size:11px;color:var(--muted)"> (p=${prob.toFixed(3)})</span>
      </div>
    </div>
  `;
}

/* ═══════════════════════════════════════════════════════════
   VISUALIZATIONS
   ═══════════════════════════════════════════════════════════ */

// ── DECISION TREE CANVAS ──────────────────────────────────
function drawTree(root) {
  const canvas = document.getElementById('treeCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 500;
  const H = Math.max(220, (trainedTree.treeDepth + 1) * 80 + 40);
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  function nodeRadius(depth) { return Math.max(14, 22 - depth * 2); }

  function draw(node, x, y, span) {
    if (!node) return;
    const r = nodeRadius(node.depth);
    const isLeaf = node.prediction !== null;
    const childY = y + 75;

    if (node.left) {
      const lx = x - span / 2;
      ctx.beginPath();
      ctx.moveTo(x, y + r);
      ctx.lineTo(lx, childY - nodeRadius(node.depth + 1));
      ctx.strokeStyle = '#30363D'; ctx.lineWidth = 1.5;
      ctx.stroke();
      // edge label
      ctx.fillStyle = '#7D8590'; ctx.font = '9px JetBrains Mono';
      ctx.fillText('≤', x - span/4 - 8, (y + childY)/2);
      draw(node.left, lx, childY, span / 2);
    }
    if (node.right) {
      const rx = x + span / 2;
      ctx.beginPath();
      ctx.moveTo(x, y + r);
      ctx.lineTo(rx, childY - nodeRadius(node.depth + 1));
      ctx.strokeStyle = '#30363D'; ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#7D8590'; ctx.font = '9px JetBrains Mono';
      ctx.fillText('>', x + span/4, (y + childY)/2);
      draw(node.right, rx, childY, span / 2);
    }

    // Node circle
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    if (isLeaf) {
      const clr = node.prediction === 1 ? '#00D4AA' : '#FF6B35';
      grad.addColorStop(0, clr + '33');
      grad.addColorStop(1, clr + '11');
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = grad; ctx.fill();
      ctx.strokeStyle = clr; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = clr; ctx.font = `bold ${Math.max(9,r-4)}px JetBrains Mono`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(node.prediction, x, y);
    } else {
      grad.addColorStop(0, '#1C2330');
      grad.addColorStop(1, '#0D1117');
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = grad; ctx.fill();
      ctx.strokeStyle = '#00D4AA'; ctx.lineWidth = 1.5; ctx.stroke();
      // feature label
      ctx.fillStyle = '#00D4AA'; ctx.font = `bold ${Math.max(8,r-6)}px JetBrains Mono`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`X${node.feature + 1}`, x, y - 4);
      ctx.fillStyle = '#7D8590'; ctx.font = `${Math.max(7,r-8)}px JetBrains Mono`;
      ctx.fillText(`≤${node.threshold.toFixed(2)}`, x, y + 6);
    }
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  }

  draw(root, W / 2, 30, W * 0.85);
}

// ── NEURON DIAGRAM ────────────────────────────────────────
function drawNeuron(neuron) {
  const canvas = document.getElementById('neuronCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 500;
  const H = 180;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2;
  const inputX = W * 0.18, outputX = W * 0.82;
  const inputYs = [cy - 50, cy + 50];
  const labels  = ['X₁', 'X₂'];
  const weights = neuron.weights;

  // Draw weight connections
  inputYs.forEach((iy, j) => {
    const w = weights[j];
    const alpha = Math.min(0.9, Math.abs(w) * 0.5 + 0.2);
    const clr = w >= 0 ? `rgba(0,212,170,${alpha})` : `rgba(255,107,53,${alpha})`;
    ctx.beginPath(); ctx.moveTo(inputX + 14, iy); ctx.lineTo(cx - 28, cy);
    ctx.strokeStyle = clr; ctx.lineWidth = Math.max(1, Math.abs(w) * 2 + 1); ctx.stroke();
    // weight label
    const mx = (inputX + cx) / 2, my = (iy + cy) / 2;
    ctx.fillStyle = clr; ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`w${j+1}=${w.toFixed(2)}`, mx, my - 4);
  });

  // Output connection
  ctx.beginPath(); ctx.moveTo(cx + 28, cy); ctx.lineTo(outputX - 14, cy);
  ctx.strokeStyle = '#FF6B35'; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = '#FF6B35'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
  ctx.fillText('ŷ', (cx + outputX) / 2, cy - 8);

  // Input nodes
  inputYs.forEach((iy, j) => {
    ctx.beginPath(); ctx.arc(inputX, iy, 14, 0, Math.PI * 2);
    const g = ctx.createRadialGradient(inputX, iy, 0, inputX, iy, 14);
    g.addColorStop(0, '#1C2330'); g.addColorStop(1, '#0D1117');
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = '#00D4AA'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = '#00D4AA'; ctx.font = 'bold 11px JetBrains Mono';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(labels[j], inputX, iy);
  });

  // Neuron body
  ctx.beginPath(); ctx.arc(cx, cy, 28, 0, Math.PI * 2);
  const ng = ctx.createRadialGradient(cx, cy, 0, cx, cy, 28);
  ng.addColorStop(0, 'rgba(0,212,170,0.2)'); ng.addColorStop(1, 'rgba(0,212,170,0.03)');
  ctx.fillStyle = ng; ctx.fill();
  ctx.strokeStyle = '#00D4AA'; ctx.lineWidth = 2; ctx.stroke();
  // Σ + sigmoid label
  ctx.fillStyle = '#00D4AA'; ctx.font = 'bold 14px JetBrains Mono';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('Σ', cx, cy - 7);
  ctx.fillStyle = '#7D8590'; ctx.font = '9px JetBrains Mono';
  ctx.fillText('σ(z)', cx, cy + 9);

  // Bias label
  ctx.fillStyle = '#7D8590'; ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(`b=${neuron.bias.toFixed(2)}`, cx, cy + 52);

  // Output node
  ctx.beginPath(); ctx.arc(outputX, cy, 14, 0, Math.PI * 2);
  const og = ctx.createRadialGradient(outputX, cy, 0, outputX, cy, 14);
  og.addColorStop(0, 'rgba(255,107,53,0.2)'); og.addColorStop(1, 'rgba(255,107,53,0.03)');
  ctx.fillStyle = og; ctx.fill();
  ctx.strokeStyle = '#FF6B35'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = '#FF6B35'; ctx.font = 'bold 11px JetBrains Mono';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('ŷ', outputX, cy);

  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
}

// ── LOSS CHART ────────────────────────────────────────────
function drawLoss(history) {
  const canvas = document.getElementById('lossCanvas');
  const W = canvas.parentElement.clientWidth - 40 || 400;
  canvas.width = W; canvas.height = 80;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H = 80);

  if (!history.length) return;
  const maxL = Math.max(...history);
  const minL = Math.min(...history);
  const range = maxL - minL || 1;
  const pad = 10;

  ctx.beginPath();
  history.forEach((v, i) => {
    const x = pad + (i / (history.length - 1 || 1)) * (W - 2 * pad);
    const y = H - pad - ((v - minL) / range) * (H - 2 * pad);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#FF6B35'); grad.addColorStop(1, '#00D4AA');
  ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.stroke();

  // Labels
  ctx.fillStyle = '#7D8590'; ctx.font = '9px JetBrains Mono';
  ctx.fillText(`${maxL.toFixed(3)}`, 2, 14);
  ctx.fillText(`${minL.toFixed(3)}`, 2, H - 4);
}

// ── DECISION BOUNDARY ─────────────────────────────────────
function drawBoundary() {
  const canvas = document.getElementById('boundaryCanvas');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const RES = 80;
  const cw = W / RES, ch = H / RES;

  for (let i = 0; i < RES; i++) {
    for (let j = 0; j < RES; j++) {
      const xn = i / (RES - 1), yn = 1 - j / (RES - 1);
      const treePred  = trainedTree.predictOne([xn, yn]);
      const neuronPred = trainedNeuron.predictOne([xn, yn]);
      const agree = treePred === neuronPred;
      const c1 = treePred === 1 ? 'rgba(0,212,170,' : 'rgba(255,107,53,';
      ctx.fillStyle = c1 + (agree ? '0.18' : '0.06') + ')';
      ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(48,54,61,0.5)'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    ctx.beginPath(); ctx.moveTo(i * W / 5, 0); ctx.lineTo(i * W / 5, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * H / 5); ctx.lineTo(W, i * H / 5); ctx.stroke();
  }

  // Data points
  const { mins, maxs } = window._normParams;
  dataset.forEach(d => {
    const xn = maxs[0] - mins[0] === 0 ? 0.5 : (d.x1 - mins[0]) / (maxs[0] - mins[0]);
    const yn = maxs[1] - mins[1] === 0 ? 0.5 : (d.x2 - mins[1]) / (maxs[1] - mins[1]);
    const px = xn * W, py = (1 - yn) * H;
    const clr = d.label === 1 ? '#00D4AA' : '#FF6B35';

    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fillStyle = clr + '33'; ctx.fill();
    ctx.strokeStyle = clr; ctx.lineWidth = 2; ctx.stroke();

    ctx.fillStyle = '#0D1117'; ctx.font = 'bold 8px JetBrains Mono';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(d.label, px, py);
  });

  // Axis labels
  ctx.fillStyle = '#7D8590'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
  ctx.fillText('X₁ →', W / 2, H - 4);
  ctx.save(); ctx.translate(10, H / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('X₂ →', 0, 0); ctx.restore();

  // Legend
  ctx.textAlign = 'left';
  [['Clase 1', '#00D4AA'], ['Clase 0', '#FF6B35']].forEach(([lbl, clr], i) => {
    ctx.beginPath(); ctx.arc(12 + i * 80, 14, 5, 0, Math.PI * 2);
    ctx.fillStyle = clr; ctx.fill();
    ctx.fillStyle = clr; ctx.font = '10px JetBrains Mono';
    ctx.fillText(lbl, 22 + i * 80, 18);
  });
}

/* ═══════════════════════════════════════════════════════════
   TOAST
   ═══════════════════════════════════════════════════════════ */
let _toastTimer;
function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 3000);
}

/* ═══════════════════════════════════════════════════════════
   SIDEBAR TOGGLE
   ═══════════════════════════════════════════════════════════ */
document.getElementById('sidebarToggle').addEventListener('click', () => {
  const sb = document.getElementById('sidebar');
  const btn = document.getElementById('sidebarToggle');
  sb.classList.toggle('collapsed');
  btn.textContent = sb.classList.contains('collapsed') ? '›' : '‹';
});

/* ═══════════════════════════════════════════════════════════
   KEYBOARD SHORTCUT: Enter en inputs
   ═══════════════════════════════════════════════════════════ */
['inX1','inX2','inClass'].forEach(id => {
  document.getElementById(id)?.addEventListener('keydown', e => {
    if (e.key === 'Enter') addDataPoint();
  });
});
['predX1','predX2'].forEach(id => {
  document.getElementById(id)?.addEventListener('keydown', e => {
    if (e.key === 'Enter') predict();
  });
});

/* ─── INIT ───────────────────────────────────────────────── */
refreshTable();
