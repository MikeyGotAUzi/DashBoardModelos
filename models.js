/* =====================================================
   REGRESION LINEAL SIMPLE  (minimos cuadrados)
   ===================================================== */
class LinearRegression {
  constructor() { this.m = 0; this.b = 0; }

  fit(X, y) {
    const n    = X.length;
    const sumX  = X.reduce((a, v) => a + v, 0);
    const sumY  = y.reduce((a, v) => a + v, 0);
    const sumXY = X.reduce((a, v, i) => a + v * y[i], 0);
    const sumX2 = X.reduce((a, v) => a + v * v, 0);
    this.m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    this.b = (sumY - this.m * sumX) / n;
  }

  predict(x) { return this.m * x + this.b; }

  r2(X, y) {
    const yMean = y.reduce((a, v) => a + v, 0) / y.length;
    const ssTot = y.reduce((a, v) => a + (v - yMean) ** 2, 0);
    const ssRes = X.reduce((a, v, i) => a + (y[i] - this.predict(v)) ** 2, 0);
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }

  mae(X, y) {
    return X.reduce((a, v, i) => a + Math.abs(y[i] - this.predict(v)), 0) / X.length;
  }
}

/* =====================================================
   ARBOL DE DECISION  (clasificacion — Gini)
   Para COVID: predice tubo, estancia, mortalidad
   ===================================================== */
class TreeNode {
  constructor() {
    this.feature   = null;
    this.threshold = null;
    this.left      = null;
    this.right     = null;
    this.value     = null;   // hoja: clase mayoritaria
    this.avgVal    = null;   // hoja: promedio (para estancia)
    this.depth     = 0;
    this.samples   = 0;
  }
}

class DecisionTree {
  constructor(maxDepth = 4) {
    this.maxDepth  = maxDepth;
    this.root      = null;
    this.nodeCount = 0;
    this.realDepth = 0;
  }

  gini(labels) {
    if (!labels.length) return 0;
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    let imp = 1;
    for (const k in counts) imp -= (counts[k] / labels.length) ** 2;
    return imp;
  }

  majority(labels) {
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    return +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }

  mean(arr) { return arr.reduce((a, v) => a + v, 0) / arr.length; }

  bestSplit(X, y) {
    let bestGain = -Infinity, bestFeat = null, bestThresh = null;
    const n = X.length;
    const parentG = this.gini(y);

    for (let f = 0; f < X[0].length; f++) {
      const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        const lY = [], rY = [];
        X.forEach((r, idx) => r[f] <= thresh ? lY.push(y[idx]) : rY.push(y[idx]));
        if (!lY.length || !rY.length) continue;
        const gain = parentG - (lY.length / n) * this.gini(lY) - (rY.length / n) * this.gini(rY);
        if (gain > bestGain) { bestGain = gain; bestFeat = f; bestThresh = thresh; }
      }
    }
    return { feature: bestFeat, threshold: bestThresh, gain: bestGain };
  }

  build(X, y, yReg, depth) {
    this.nodeCount++;
    this.realDepth = Math.max(this.realDepth, depth);
    const node = new TreeNode();
    node.depth = depth; node.samples = X.length;

    const unique = [...new Set(y)];
    if (unique.length === 1 || depth >= this.maxDepth || X.length < 4) {
      node.value  = this.majority(y);
      node.avgVal = this.mean(yReg);
      return node;
    }

    const { feature, threshold, gain } = this.bestSplit(X, y);
    if (feature === null || gain <= 0) {
      node.value  = this.majority(y);
      node.avgVal = this.mean(yReg);
      return node;
    }

    node.feature = feature; node.threshold = threshold;

    const lIdx = X.map((r, i) => r[feature] <= threshold ? i : -1).filter(i => i >= 0);
    const rIdx = X.map((r, i) => r[feature] >  threshold ? i : -1).filter(i => i >= 0);

    node.left  = this.build(lIdx.map(i => X[i]), lIdx.map(i => y[i]),    lIdx.map(i => yReg[i]), depth + 1);
    node.right = this.build(rIdx.map(i => X[i]), rIdx.map(i => y[i]),    rIdx.map(i => yReg[i]), depth + 1);
    return node;
  }

  fit(X, y, yReg) {
    this.nodeCount = 0; this.realDepth = 0;
    this.root = this.build(X, y, yReg, 0);
  }

  predictOne(x, node = this.root) {
    if (node.value !== null) return { clase: node.value, estancia: node.avgVal };
    return x[node.feature] <= node.threshold
      ? this.predictOne(x, node.left)
      : this.predictOne(x, node.right);
  }

  accuracy(X, y) {
    return X.filter((x, i) => this.predictOne(x).clase === y[i]).length / X.length;
  }
}
