/* =====================================================
   ESTADO
   ===================================================== */
let regDataset  = [];
let regModel    = null;
let regTrained  = false;

let treeModelTubo      = null;
let treeModelMortalidad = null;
let treeTrained        = false;

let activeTab = 'regression';

const REG_PRESETS = {
  ventas:      [[1,20],[2,35],[3,50],[4,60],[5,80],[6,95],[7,110],[8,125],[9,140],[10,160]],
  temperatura: [[10,8],[12,10],[15,14],[18,20],[20,25],[22,30],[25,38],[27,42],[30,50],[32,55]]
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
  document.getElementById('sidebarReg').style.display    = tab === 'regression' ? '' : 'none';
  document.getElementById('sidebarTree').style.display   = tab === 'tree'       ? '' : 'none';

  const titles = { regression: 'Regresion Lineal', tree: 'Arbol de Decision — COVID-19' };
  document.getElementById('modelTitle').textContent = titles[tab];

  updateStatus();
}

function updateStatus() {
  const trained = activeTab === 'regression' ? regTrained : treeTrained;
  document.getElementById('statusText').textContent = trained ? 'Entrenado' : 'Sin entrenar';
}

/* =====================================================
   DATOS REGRESION
   ===================================================== */
function addRegPoint() {
  const x = parseFloat(document.getElementById('regInX').value);
  const y = parseFloat(document.getElementById('regInY').value);
  if (isNaN(x) || isNaN(y)) { toast('Ingresa valores numericos validos', 'error'); return; }
  regDataset.push({ x, y });
  document.getElementById('regInX').value = '';
  document.getElementById('regInY').value = '';
  refreshRegTable();
  regTrained = false;
  updateStatus();
}

function removeRegPoint(i) {
  regDataset.splice(i, 1);
  refreshRegTable();
  regTrained = false;
  updateStatus();
}

function clearRegData() {
  regDataset = [];
  refreshRegTable();
  regTrained = false;
  resetRegVisuals();
  updateStatus();
}

function loadRegPreset(name) {
  regDataset = REG_PRESETS[name].map(([x, y]) => ({ x, y }));
  refreshRegTable();
  regTrained = false;
  resetRegVisuals();
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
   ENTRENAR
   ===================================================== */
function trainActive() {
  if (activeTab === 'regression') trainRegression();
  else trainTree();
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
  // Preparar datos desde COVID_DATA
  // columnas: edad(0) genero(1) dx(2) intervenciones(3)
  //           hospitalizado(4) urgencias(5) gravedad(6)
  //           tubo(7) estancia(8) mortalidad(9)
  const depth = parseInt(document.getElementById('maxDepth').value);

  const X        = COVID_DATA.map(r => [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]);
  const yTubo    = COVID_DATA.map(r => r[7]);
  const yEstancia = COVID_DATA.map(r => r[8]);
  const yMort    = COVID_DATA.map(r => r[9]);

  // Arbol para tubo (usa mortalidad como target secundario para estancia en hojas)
  treeModelTubo = new DecisionTree(depth);
  treeModelTubo.fit(X, yTubo, yEstancia);

  treeModelMortalidad = new DecisionTree(depth);
  treeModelMortalidad.fit(X, yMort, yEstancia);

  const accTubo = treeModelTubo.accuracy(X, yTubo);

  document.getElementById('treeAcc').textContent      = (accTubo * 100).toFixed(1) + '%';
  document.getElementById('treeRecs').textContent     = COVID_DATA.length;
  document.getElementById('treeNodes').textContent    = treeModelTubo.nodeCount;
  document.getElementById('treeDepthVal').textContent = treeModelTubo.realDepth;
  document.getElementById('treeMetrics').style.display = 'flex';
  document.getElementById('treeIdle').style.display    = 'none';
  document.getElementById('covidPredBtn').disabled     = false;

  // Actualizar sidebar
  document.getElementById('treeDataInfo').innerHTML =
    `Registros cargados: <strong style="color:#aaa">${COVID_DATA.length}</strong><br>
     Variables: edad, genero, dx, intervenciones, hospitalizado, urgencias, gravedad<br><br>
     Predice: tubo, estancia, mortalidad`;

  treeTrained = true;
  updateStatus();
  toast('Modelo COVID entrenado con ' + COVID_DATA.length + ' registros', 'ok');
}

/* =====================================================
   PREDECIR REGRESION
   ===================================================== */
function predictReg() {
  const x = parseFloat(document.getElementById('regPredX').value);
  if (isNaN(x)) { toast('Ingresa un valor numerico', 'error'); return; }
  const pred = regModel.predict(x);
  const el = document.getElementById('regPredResult');
  el.style.display = '';
  el.innerHTML = `Para X = <strong>${x}</strong> &rarr; Y estimada = <strong>${pred.toFixed(2)}</strong>`;
}

/* =====================================================
   PREDECIR COVID
   ===================================================== */
function predictCovid() {
  if (!treeTrained) return;

  const edad           = parseFloat(document.getElementById('covidEdad').value);
  const genero         = parseInt(document.getElementById('covidGenero').value);
  const dx             = parseInt(document.getElementById('covidDx').value);
  const intervenciones = parseInt(document.getElementById('covidIntervenciones').value);
  const hospitalizado  = document.getElementById('covidHospitalizado').checked ? 1 : 0;
  const urgencias      = document.getElementById('covidUrgencias').checked ? 1 : 0;
  const gravedad       = parseInt(document.getElementById('covidGravedad').value);

  if (isNaN(edad) || isNaN(intervenciones)) {
    toast('Ingresa edad e intervenciones medicas', 'error'); return;
  }

  const x = [edad, genero, dx, intervenciones, hospitalizado, urgencias, gravedad];

  const resTubo   = treeModelTubo.predictOne(x);
  const resMort   = treeModelMortalidad.predictOne(x);
  const estancia  = Math.round(resTubo.estancia);

  // Mostrar resultados
  const elTubo = document.getElementById('resTubo');
  elTubo.textContent = resTubo.clase === 1 ? 'SI' : 'NO';
  elTubo.className   = 'result-val ' + (resTubo.clase === 1 ? 'riesgo-alto' : 'riesgo-bajo');

  document.getElementById('resEstancia').textContent = estancia + ' dias';

  const elMort = document.getElementById('resMortalidad');
  let nivelMort = '';
  if (resMort.clase === 1 && gravedad >= 4) {
    nivelMort = 'ALTO RIESGO'; elMort.className = 'result-val riesgo-alto';
  } else if (resMort.clase === 1 || gravedad >= 3) {
    nivelMort = 'RIESGO MEDIO'; elMort.className = 'result-val riesgo-medio';
  } else {
    nivelMort = 'BAJO RIESGO'; elMort.className = 'result-val riesgo-bajo';
  }
  elMort.textContent = nivelMort;

  // Alerta
  const elAlerta = document.getElementById('resAlerta');
  if (resMort.clase === 1 && gravedad >= 4) {
    elAlerta.style.display = '';
    elAlerta.className = 'alerta';
    elAlerta.textContent = 'ALERTA: El modelo clasifica al paciente con probabilidad de deceso. Se sugiere monitoreo intensivo.';
  } else if (resMort.clase === 0 && gravedad <= 2) {
    elAlerta.style.display = '';
    elAlerta.className = 'alerta info';
    elAlerta.textContent = 'El modelo indica bajo riesgo de mortalidad. Se recomienda seguimiento estandar.';
  } else {
    elAlerta.style.display = 'none';
  }

  document.getElementById('covidResult').style.display = '';
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

/* =====================================================
   GRAFICA REGRESION LINEAL
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

  // grid y etiquetas Y
  ctx.fillStyle = '#aaa'; ctx.font = '10px Arial'; ctx.textAlign = 'right';
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const v  = yMin + t * yR;
    const yy = py(v);
    ctx.fillText(v.toFixed(1), pad.left - 4, yy + 3);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + iW, yy); ctx.stroke();
  });

  // etiquetas X
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
['regInX', 'regInY'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') addRegPoint(); })
);
document.getElementById('regPredX').addEventListener('keydown', e => { if (e.key === 'Enter') predictReg(); });

/* INIT */
refreshRegTable();
