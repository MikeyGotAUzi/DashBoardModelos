/* =====================================================
   REGRESION LINEAL SIMPLE  (minimos cuadrados)
   ===================================================== */
class LinearRegression {
  constructor() { this.m = 0; this.b = 0; }

  fit(X, y) {
    const n = X.length;
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
   ARBOL DE DECISION  (regresion — varianza minima)
   ===================================================== */
class TreeNode {
  constructor() {
    this.threshold = null;
    this.left      = null;
    this.right     = null;
    this.value     = null;
    this.depth     = 0;
    this.samples   = 0;
  }
}

class DecisionTreeRegressor {
  constructor(maxDepth = 3) {
    this.maxDepth  = maxDepth;
    this.root      = null;
    this.nodeCount = 0;
  }

  variance(arr) {
    if (!arr.length) return 0;
    const mean = arr.reduce((a, v) => a + v, 0) / arr.length;
    return arr.reduce((a, v) => a + (v - mean) ** 2, 0) / arr.length;
  }

  mean(arr) { return arr.reduce((a, v) => a + v, 0) / arr.length; }

  bestSplit(X, y) {
    let bestGain = -Infinity, bestThresh = null;
    const parentVar = this.variance(y);
    const n = X.length;
    const vals = [...new Set(X)].sort((a, b) => a - b);

    for (let i = 0; i < vals.length - 1; i++) {
      const thresh = (vals[i] + vals[i + 1]) / 2;
      const lY = [], rY = [];
      X.forEach((v, idx) => v <= thresh ? lY.push(y[idx]) : rY.push(y[idx]));
      if (!lY.length || !rY.length) continue;
      const gain = parentVar
        - (lY.length / n) * this.variance(lY)
        - (rY.length / n) * this.variance(rY);
      if (gain > bestGain) { bestGain = gain; bestThresh = thresh; }
    }
    return { threshold: bestThresh, gain: bestGain };
  }

  build(X, y, depth) {
    this.nodeCount++;
    const node = new TreeNode();
    node.depth = depth; node.samples = X.length;

    if (depth >= this.maxDepth || X.length < 2) {
      node.value = this.mean(y); return node;
    }

    const { threshold, gain } = this.bestSplit(X, y);
    if (threshold === null || gain <= 0) {
      node.value = this.mean(y); return node;
    }

    node.threshold = threshold;
    const lIdx = X.map((v, i) => v <= threshold ? i : -1).filter(i => i >= 0);
    const rIdx = X.map((v, i) => v >  threshold ? i : -1).filter(i => i >= 0);
    node.left  = this.build(lIdx.map(i => X[i]), lIdx.map(i => y[i]), depth + 1);
    node.right = this.build(rIdx.map(i => X[i]), rIdx.map(i => y[i]), depth + 1);
    return node;
  }

  fit(X, y) { this.nodeCount = 0; this.root = this.build(X, y, 0); }

  predictOne(x, node = this.root) {
    if (node.value !== null) return node.value;
    return x <= node.threshold
      ? this.predictOne(x, node.left)
      : this.predictOne(x, node.right);
  }

  predict(X) { return X.map(x => this.predictOne(x)); }

  r2(X, y) {
    const preds = this.predict(X);
    const yMean = y.reduce((a, v) => a + v, 0) / y.length;
    const ssTot = y.reduce((a, v) => a + (v - yMean) ** 2, 0);
    const ssRes = preds.reduce((a, p, i) => a + (y[i] - p) ** 2, 0);
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }

  mae(X, y) {
    const preds = this.predict(X);
    return preds.reduce((a, p, i) => a + Math.abs(y[i] - p), 0) / X.length;
  }
}

/* =====================================================
   PERCEPTRON  (neurona — clasificacion binaria)
   Activacion: Sigmoid   Error: entropia cruzada
   ===================================================== */
class Perceptron {
  constructor(lr = 0.1, epochs = 100) {
    this.lr = lr;
    this.epochs = epochs;
    this.w = [];
    this.bias = 0;
    this.lossHistory = [];
  }

  sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  fit(X, y) {
    const nFeats = X[0].length;
    this.w = new Array(nFeats).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.bias = 0;
    this.lossHistory = [];

    for (let e = 0; e < this.epochs; e++) {
      let loss = 0;
      for (let i = 0; i < X.length; i++) {
        const z    = X[i].reduce((s, xi, j) => s + xi * this.w[j], this.bias);
        const pred = this.sigmoid(z);
        const err  = pred - y[i];
        loss += -y[i] * Math.log(pred + 1e-9) - (1 - y[i]) * Math.log(1 - pred + 1e-9);
        for (let j = 0; j < nFeats; j++) this.w[j] -= this.lr * err * X[i][j];
        this.bias -= this.lr * err;
      }
      const interval = Math.max(1, Math.floor(this.epochs / 60));
      if (e % interval === 0) this.lossHistory.push(loss / X.length);
    }
  }

  predictProb(x) {
    const z = x.reduce((s, xi, j) => s + xi * this.w[j], this.bias);
    return this.sigmoid(z);
  }

  predictOne(x) { return this.predictProb(x) >= 0.5 ? 1 : 0; }

  accuracy(X, y) {
    return X.filter((x, i) => this.predictOne(x) === y[i]).length / X.length;
  }
}
