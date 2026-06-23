/* =====================================================
   ESTADO
   ===================================================== */
let regDataset    = [];   // {x, y}
let neuronDataset = [];   // {x1, x2, label}

let regModel    = null;
let treeModel   = null;
let neuronModel = null;

let regTrained    = false;
let treeTrained   = false;
let neuronTrained = false;

let activeTab = 'regression';

/* =====================================================
   PRESETS
   ===================================================== */
const REG_PRESETS = {
  ventas:      [[1,20],[2,35],[3,50],[4,60],[5,80],[6,95],[7,110],[8,125],[9,140],[10,160]],
  temperatura: [[10,8],[12,10],[15,14],[18,20],[20,25],[22,30],[25,38],[27,42],[30,50],[32,55]]
};

const NEURON_PRESETS = {
  and: [[0,0,0],[0,1,0],[1,0,0],[1,1,1],[0.1,0.1,0],[0.9,0.9,1],[0.8,0.9,1],[0.1,0.9,0]],
  or:  [[0,0,0],[0,1,1],[1,0,1],[1,1,1],[0.1,0.1,0],[0.9,0.1,1],[0.1,0.8,1],[0.9,0.9,1]]
};

/* =====================================================
   TABS
   ===================================================== */
function switchTab(tab, btn) {
  activeTab = tab;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  document.getElementById('tabRegression').style.display = tab === 'regression' ? '' : 'none';
  document.getElementById('tabTree').style.display       = tab === 'tree'       ? '' : 'none';
  document.getElementById('tabNeuron').style.display     = tab === 'neuron'     ? '' : 'none';

  document.getElementById('sidebarReg').style.display    = tab === 'neuron' ? 'none' : '';
  document.getElementById('sidebarNeuron').style.display = tab === 'neuron' ? ''     : 'none';

  const titles = { regression: 'Regresion Lineal', tree: 'Arbol de Decision', neuron: 'Neurona' };
  document.getElementById('modelTitle').textContent = titles[tab];

  updateStatus();
}

function updateStatus() {
  const trained = activeTab === 'regression' ? regTrained
                : activeTab === 'tree'       ? treeTrained
                :                              neuronTrained;
  document.getElementById('statusText').textContent = trained ? 'Entrenado' : 'Sin entrenar';
}

/* =====================================================
   DATOS — REGRESION Y ARBOL (comparten dataset)
   ===================================================== */
function addRegPoint() {
  const xId = activeTab === 'tree' ? 'treeInX' : 'regInX';
  const yId = activeTab === 'tree' ? 'treeInY' : 'regInY';
  const x = parseFloat(document.getElementById(xId).value);
  const y = parseFloat(document.getElementById(yId).value);
  if (isNaN(x) || isNaN(y)) { toast('Ingresa valores numericos validos', 'error'); return; }
  regDataset.push({ x, y });
  document.getElementById(xId).value = '';
  document.getElementById(yId).value = '';
  refreshRegTable();
  regTrained = false; treeTrained = false;
  updateStatus();
}

function removeRegPoint(i) {
  regDataset.splice(i, 1);
  refreshRegTable();
  regTrained = false; treeTrained = false;
  updateStatus();
}

function clearRegData() {
  regDataset = [];
  refreshRegTable();
  regTrained = false; treeTrained = false;
  resetRegVisuals(); resetTreeVisuals();
  updateStatus();
}

function loadRegPreset(name) {
  regDataset = REG_PRESETS[name].map(([x, y]) => ({ x, y }));
  refreshRegTable();
  regTrained = false; treeTrained = false;
  resetRegVisuals(); resetTreeVisuals();
  updateStatus();
  toast('Datos de ejemplo cargados');
}

function refreshRegTable() {
  const body  = document.getElementById('regDataBody');
  const empty = document.getElementById('regEmptyMsg');
  document.getElementById('regDataCount').textContent = regDataset.length;
  if (!regDataset.length) { body.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  body.innerHTML = regDataset.map((d, i) => `
    <tr>
      <td>${d.x}</td><td>${d.y}</td>
      <td><button class="del-btn" onclick="removeRegPoint(${i})">x</button></td>
    </tr>
  `).join('');
}

/* =====================================================
   DATOS — NEURONA
   ===================================================== */
function addNeuronPoint() {
  const x1    = parseFloat(document.getElementById('nInX1').value);
  const x2    = parseFloat(document.getElementById('nInX2').value);
  const label = parseInt(document.getElementById('nInClass').value);
  if (isNaN(x1) || isNaN(x2)) { toast('Ingresa valores numericos validos', 'error'); return; }
  neuronDataset.push({ x1, x2, label });
  document.getElementById('nInX1').value = '';
  document.getElementById('nInX2').value = '';
  refreshNeuronTable();
  neuronTrained = false;
  updateStatus();
}

function removeNeuronPoint(i) {
  neuronDataset.splice(i, 1);
  refreshNeuronTable();
  neuronTrained = false;
  updateStatus();
}

function clearNeuronData() {
  neuronDataset = [];
  refreshNeuronTable();
  neuronTrained = false;
  resetNeuronVisuals();
  updateStatus();
}

function loadNeuronPreset(name) {
  neuronDataset = NEURON_PRESETS[name].map(([x1, x2, label]) => ({ x1, x2, label }));
  refreshNeuronTable();
  neuronTrained = false;
  resetNeuronVisuals();
  updateStatus();
  toast('Datos de ejemplo cargados');
}

function refreshNeuronTable() {
  const body  = document.getElementById('neuronDataBody');
  const empty = document.getElementById('neuronEmptyMsg');
  document.getElementById('neuronDataCount').textContent = neuronDataset.length;
  if (!neuronDataset.length) { body.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  body.innerHTML = neuronDataset.map((d, i) => `
    <tr>
      <td>${d.x1}</td><td>${d.x2}</td><td>${d.label}</td>
      <td><button class="del-btn" onclick="removeNeuronPoint(${i})">x</button></td>
    </tr>
  `).join('');
}

/* =====================================================
   ENTRENAR
   ===================================================== */
function trainActive() {
  if (activeTab === 'regression') trainRegression();
  else if (activeTab === 'tree')  trainTree();
  else                            trainNeuron();
}

function trainRegression() {
  if (regDataset.length < 3) { toast('Necesitas al menos 3 datos', 'error'); return; }
  const X = regDataset.map(d => d.x);
  const y = regDataset.map(d => d.y);

  regModel = new LinearRegression();
  regModel.fit(X, y);

  document.getElementById('regM').textContent   = regModel.m.toFixed(3);
  document.getElementById('regB').textContent   = regModel.b.toFixed(3);
  document.getElementById('regR2').textContent  = regModel.r2(X, y).toFixed(3);
  document.getElementById('regMAE').textContent = regModel.mae(X, y).toFixed(2);
  document.getElementById('regMetrics').style.display = 'flex';
  document.getElementById('regIdle').style.display    = 'none';
  document.getElementById('regPredBtn').disabled      = false;

  drawRegression(X, y);
  regTrained = true;
  updateStatus();
  toast('Regresion lineal entrenada', 'ok');
}

function trainTree() {
  if (regDataset.length < 3) { toast('Necesitas al menos 3 datos', 'error'); return; }
  const X = regDataset.map(d => d.x);
  const y = regDataset.map(d => d.y);
  const depth = parseInt(document.getElementById('maxDepth').value);

  treeModel = new DecisionTreeRegressor(depth);
  treeModel.fit(X, y);

  document.getElementById('treeMAE').textContent   = treeModel.mae(X, y).toFixed(2);
  document.getElementById('treeR2').textContent    = treeModel.r2(X, y).toFixed(3);
  document.getElementById('treeNodes').textContent = treeModel.nodeCount;
  document.getElementById('treeMetrics').style.display = 'flex';
  document.getElementById('treeIdle').style.display    = 'none';
  document.getElementById('treePredBtn').disabled      = false;

  drawTree(treeModel.root, X, y);
  treeTrained = true;
  updateStatus();
  toast('Arbol de decision entrenado', 'ok');
}

function trainNeuron() {
  if (neuronDataset.length < 4) { toast('Necesitas al menos 4 datos', 'error'); return; }
  const X = neuronDataset.map(d => [d.x1, d.x2]);
  const y = neuronDataset.map(d => d.label);
  const epochs = parseInt(document.getElementById('epochs').value);
  const lr     = parseFloat(document.getElementById('lr').value);

  neuronModel = new Perceptron(lr, epochs);
  neuronModel.fit(X, y);

  const acc = neuronModel.accuracy(X, y);
  document.getElementById('nAcc').textContent  = (acc * 100).toFixed(1) + '%';
  document.getElementById('nW1').textContent   = neuronModel.w[0].toFixed(3);
  document.getElementById('nW2').textContent   = neuronModel.w[1].toFixed(3);
  document.getElementById('nBias').textContent = neuronModel.bias.toFixed(3);
  document.getElementById('neuronMetrics').style.display = 'flex';
  document.getElementById('neuronPredBtn').disabled      = false;
  neuronTrained = true;
  updateStatus();
  toast('Neurona entrenada', 'ok');
}

/* =====================================================
   PREDECIR
   ===================================================== */
function predictReg() {
  const x = parseFloat(document.getElementById('regPredX').value);
  if (isNaN(x)) { toast('Ingresa un valor numerico', 'error'); return; }
  const pred = regModel.predict(x);
  const el = document.getElementById('regPredResult');
  el.style.display = '';
  el.innerHTML = `Para X = <strong>${x}</strong> &rarr; Y estimada = <strong>${pred.toFixed(2)}</strong>`;
}

function predictTree() {
  const x = parseFloat(document.getElementById('treePredX').value);
  if (isNaN(x)) { toast('Ingresa un valor numerico', 'error'); return; }
  const pred = treeModel.predictOne(x);
  const el = document.getElementById('treePredResult');
  el.style.display = '';
  el.innerHTML = `Para X = <strong>${x}</strong> &rarr; Y estimada = <strong>${pred.toFixed(2)}</strong>`;
}

function predictNeuron() {
  const x1 = parseFloat(document.getElementById('nPredX1').value);
  const x2 = parseFloat(document.getElementById('nPredX2').value);
  if (isNaN(x1) || isNaN(x2)) { toast('Ingresa valores numericos', 'error'); return; }
  const clase = neuronModel.predictOne([x1, x2]);
  const prob  = neuronModel.predictProb([x1, x2]);
  const el = document.getElementById('neuronPredResult');
  el.style.display = '';
  el.innerHTML = `Para X1 = <strong>${x1}</strong>, X2 = <strong>${x2}</strong> &rarr; Clase = <strong>${clase}</strong> (probabilidad: ${prob.toFixed(3)})`;
}

/* =====================================================
   RESET VISUALS
   ===================================================== */
function resetRegVisuals() {
  document.getElementById('regMetrics').style.display = 'none';
  document.getElementById('regIdle').style.display    = '';
  document.getElementById('regPredBtn').disabled      = true;
  document.getElementById('regPredResult').style.display = 'none';
  const c = document.getElementById('regCanvas');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function resetTreeVisuals() {
  document.getElementById('treeMetrics').style.display = 'none';
  document.getElementById('treeIdle').style.display    = '';
  document.getElementById('treePredBtn').disabled      = true;
  document.getElementById('treePredResult').style.display = 'none';
  const c = document.getElementById('treeCanvas');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function resetNeuronVisuals() {
  document.getElementById('neuronMetrics').style.display = 'none';
  document.getElementById('neuronPredBtn').disabled      = true;
  document.getElementById('neuronPredResult').style.display = 'none';
}

/* =====================================================
   GRAFICA — REGRESION LINEAL
   ===================================================== */
function drawRegression(X, y) {
  const canvas = document.getElementById('regCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 400;
  const H = 220;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const pad = { top: 16, right: 16, bottom: 30, left: 44 };
  const iW = W - pad.left - pad.right;
  const iH = H - pad.top  - pad.bottom;

  const xMin = Math.min(...X), xMax = Math.max(...X);
  const allY = [...y, regModel.predict(xMin), regModel.predict(xMax)];
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;

  const px = v => pad.left + ((v - xMin) / xR) * iW;
  const py = v => pad.top  + iH - ((v - yMin) / yR) * iH;

  // ejes
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + iH);
  ctx.lineTo(pad.left + iW, pad.top + iH);
  ctx.stroke();

  // grid y etiquetas
  ctx.fillStyle = '#aaa'; ctx.font = '10px Arial'; ctx.textAlign = 'right';
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const v  = yMin + t * yR;
    const yy = py(v);
    ctx.fillText(v.toFixed(1), pad.left - 4, yy + 3);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + iW, yy); ctx.stroke();
  });
  ctx.textAlign = 'center'; ctx.strokeStyle = '#bbb';
  [0, 0.5, 1].forEach(t => {
    const v = xMin + t * xR;
    ctx.fillText(v.toFixed(1), px(v), pad.top + iH + 14);
  });

  // linea regresion
  ctx.beginPath();
  ctx.moveTo(px(xMin), py(regModel.predict(xMin)));
  ctx.lineTo(px(xMax), py(regModel.predict(xMax)));
  ctx.strokeStyle = '#2c3e50'; ctx.lineWidth = 2; ctx.stroke();

  // puntos
  X.forEach((xi, i) => {
    ctx.beginPath(); ctx.arc(px(xi), py(y[i]), 4, 0, Math.PI * 2);
    ctx.fillStyle = '#2980b9'; ctx.fill();
  });

  // formula
  ctx.fillStyle = '#444'; ctx.font = '11px Arial'; ctx.textAlign = 'left';
  ctx.fillText('y = ' + regModel.m.toFixed(2) + 'x + ' + regModel.b.toFixed(2), pad.left + 6, pad.top + 13);
}

/* =====================================================
   GRAFICA — ARBOL DE DECISION
   ===================================================== */
function drawTree(root, X, y) {
  const canvas = document.getElementById('treeCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 400;

  function treeDepth(node) {
    if (!node || node.value !== null) return 0;
    return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
  }
  const d = treeDepth(root);
  const H = Math.max(180, (d + 1) * 72 + 30);
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const R = 18;

  function draw(node, x, y, span) {
    if (!node) return;
    const isLeaf = node.value !== null;
    const childY = y + 68;

    if (node.left) {
      const lx = x - span / 2;
      ctx.beginPath(); ctx.moveTo(x, y + R); ctx.lineTo(lx, childY - R);
      ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = '#999'; ctx.font = '9px Arial'; ctx.textAlign = 'center';
      ctx.fillText('<= ' + node.threshold.toFixed(2), x - span / 4, (y + childY) / 2);
      draw(node.left, lx, childY, span / 2);
    }
    if (node.right) {
      const rx = x + span / 2;
      ctx.beginPath(); ctx.moveTo(x, y + R); ctx.lineTo(rx, childY - R);
      ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = '#999'; ctx.font = '9px Arial'; ctx.textAlign = 'center';
      ctx.fillText('> ' + node.threshold.toFixed(2), x + span / 4, (y + childY) / 2);
      draw(node.right, rx, childY, span / 2);
    }

    ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fillStyle   = isLeaf ? '#d5e8d4' : '#dae8fc';
    ctx.strokeStyle = isLeaf ? '#82b366' : '#6c8ebf';
    ctx.lineWidth = 1.5; ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#333'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    if (isLeaf) {
      ctx.font = 'bold 10px Arial';
      ctx.fillText(node.value.toFixed(1), x, y);
    } else {
      ctx.font = '9px Arial';
      ctx.fillText('X', x, y - 5);
      ctx.fillStyle = '#555';
      ctx.fillText('<=' + node.threshold.toFixed(1), x, y + 6);
    }
    ctx.textBaseline = 'alphabetic';
  }

  draw(root, W / 2, 24, W * 0.8);
}

/* =====================================================
   DIAGRAMA — NEURONA
   ===================================================== */
function drawNeuron(model) {
  const canvas = document.getElementById('neuronCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 400;
  const H = 170;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2;
  const inX = W * 0.18, outX = W * 0.82;
  const inYs = [cy - 45, cy + 45];

  // conexiones con pesos
  inYs.forEach((iy, j) => {
    const w = model.w[j];
    const alpha = Math.min(0.9, Math.abs(w) * 0.4 + 0.25);
    ctx.beginPath(); ctx.moveTo(inX + 14, iy); ctx.lineTo(cx - 24, cy);
    ctx.strokeStyle = `rgba(44,62,80,${alpha})`; ctx.lineWidth = Math.max(1, Math.abs(w) + 1); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
    ctx.fillText('w' + (j+1) + '=' + w.toFixed(2), (inX + cx) / 2, (iy + cy) / 2 - 4);
  });

  // salida
  ctx.beginPath(); ctx.moveTo(cx + 24, cy); ctx.lineTo(outX - 14, cy);
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1.5; ctx.stroke();

  // nodos de entrada
  inYs.forEach((iy, j) => {
    ctx.beginPath(); ctx.arc(inX, iy, 14, 0, Math.PI * 2);
    ctx.fillStyle = '#dae8fc'; ctx.fill();
    ctx.strokeStyle = '#6c8ebf'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = '#333'; ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('X' + (j+1), inX, iy);
  });

  // cuerpo de la neurona
  ctx.beginPath(); ctx.arc(cx, cy, 24, 0, Math.PI * 2);
  ctx.fillStyle = '#fff3cd'; ctx.fill();
  ctx.strokeStyle = '#d6a020'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = '#333'; ctx.font = 'bold 11px Arial';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('S', cx, cy - 6);
  ctx.fillStyle = '#777'; ctx.font = '9px Arial';
  ctx.fillText('sigmoid', cx, cy + 7);

  // bias
  ctx.fillStyle = '#888'; ctx.font = '10px Arial'; ctx.textBaseline = 'alphabetic';
  ctx.fillText('b=' + model.bias.toFixed(2), cx - 14, cy + 44);

  // nodo de salida
  ctx.beginPath(); ctx.arc(outX, cy, 14, 0, Math.PI * 2);
  ctx.fillStyle = '#d5e8d4'; ctx.fill();
  ctx.strokeStyle = '#82b366'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = '#333'; ctx.font = 'bold 10px Arial';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('y', outX, cy);

  ctx.textBaseline = 'alphabetic';
}

/* =====================================================
   CURVA DE PERDIDA
   ===================================================== */
function drawLoss(history) {
  const canvas = document.getElementById('lossCanvas');
  const W = canvas.parentElement.clientWidth - 32 || 400;
  const H = 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (history.length < 2) return;

  const maxL = Math.max(...history), minL = Math.min(...history);
  const rng  = maxL - minL || 1;
  const pad  = 12;

  ctx.beginPath();
  history.forEach((v, i) => {
    const x = pad + (i / (history.length - 1)) * (W - 2 * pad);
    const y = H - pad - ((v - minL) / rng) * (H - 2 * pad);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#2c3e50'; ctx.lineWidth = 1.5; ctx.stroke();

  ctx.fillStyle = '#aaa'; ctx.font = '9px Arial'; ctx.textAlign = 'left';
  ctx.fillText(maxL.toFixed(3), 2, 12);
  ctx.fillText(minL.toFixed(3), 2, H - 3);
}

/* =====================================================
   TOAST
   ===================================================== */
let _t;
function toast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  clearTimeout(_t);
  _t = setTimeout(() => el.classList.remove('show'), 2800);
}

/* =====================================================
   ENTER EN INPUTS
   ===================================================== */
['regInX','regInY'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') addRegPoint(); })
);
['treeInX','treeInY'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') addRegPoint(); })
);
['nInX1','nInX2'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') addNeuronPoint(); })
);
document.getElementById('regPredX').addEventListener('keydown',  e => { if (e.key === 'Enter') predictReg(); });
document.getElementById('treePredX').addEventListener('keydown', e => { if (e.key === 'Enter') predictTree(); });
document.getElementById('nPredX1').addEventListener('keydown',   e => { if (e.key === 'Enter') predictNeuron(); });
document.getElementById('nPredX2').addEventListener('keydown',   e => { if (e.key === 'Enter') predictNeuron(); });

/* INIT */
refreshRegTable();
refreshNeuronTable();
