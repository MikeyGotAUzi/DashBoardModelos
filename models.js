/* ═══════════════════════════════════════════════════════════
   DECISION TREE  (CART — Gini impurity)
   ═══════════════════════════════════════════════════════════ */
class DecisionTreeNode {
  constructor() {
    this.feature = null;
    this.threshold = null;
    this.left = null;
    this.right = null;
    this.prediction = null;
    this.depth = 0;
    this.samples = 0;
    this.gini = 0;
  }
}

class DecisionTree {
  constructor(maxDepth = 3) {
    this.maxDepth = maxDepth;
    this.root = null;
    this.nodeCount = 0;
    this.treeDepth = 0;
  }

  gini(labels) {
    if (!labels.length) return 0;
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    let imp = 1;
    for (const k in counts) imp -= (counts[k] / labels.length) ** 2;
    return imp;
  }

  bestSplit(X, y) {
    let bestGain = -Infinity, bestFeat = null, bestThresh = null;
    const n = X.length;
    const feats = X[0].length;
    const parentGini = this.gini(y);

    for (let f = 0; f < feats; f++) {
      const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        const leftIdx  = X.map((r, i) => r[f] <= thresh ? i : -1).filter(i => i >= 0);
        const rightIdx = X.map((r, i) => r[f] >  thresh ? i : -1).filter(i => i >= 0);
        if (!leftIdx.length || !rightIdx.length) continue;
        const lY = leftIdx.map(i => y[i]);
        const rY = rightIdx.map(i => y[i]);
        const gain = parentGini
          - (lY.length / n) * this.gini(lY)
          - (rY.length / n) * this.gini(rY);
        if (gain > bestGain) {
          bestGain = gain; bestFeat = f; bestThresh = thresh;
        }
      }
    }
    return { feature: bestFeat, threshold: bestThresh, gain: bestGain };
  }

  majority(labels) {
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    return +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }

  buildNode(X, y, depth) {
    this.nodeCount++;
    this.treeDepth = Math.max(this.treeDepth, depth);
    const node = new DecisionTreeNode();
    node.depth = depth;
    node.samples = X.length;
    node.gini = this.gini(y);

    const unique = [...new Set(y)];
    if (unique.length === 1 || depth >= this.maxDepth || X.length < 2) {
      node.prediction = this.majority(y);
      return node;
    }

    const { feature, threshold, gain } = this.bestSplit(X, y);
    if (feature === null || gain <= 0) {
      node.prediction = this.majority(y);
      return node;
    }

    node.feature = feature;
    node.threshold = threshold;

    const leftIdx  = X.map((r, i) => r[feature] <= threshold ? i : -1).filter(i => i >= 0);
    const rightIdx = X.map((r, i) => r[feature] >  threshold ? i : -1).filter(i => i >= 0);

    node.left  = this.buildNode(leftIdx.map(i => X[i]),  leftIdx.map(i => y[i]),  depth + 1);
    node.right = this.buildNode(rightIdx.map(i => X[i]), rightIdx.map(i => y[i]), depth + 1);

    return node;
  }

  fit(X, y) {
    this.nodeCount = 0;
    this.treeDepth = 0;
    this.root = this.buildNode(X, y, 0);
  }

  predictOne(x, node = this.root) {
    if (node.prediction !== null) return node.prediction;
    if (x[node.feature] <= node.threshold) return this.predictOne(x, node.left);
    return this.predictOne(x, node.right);
  }

  predict(X) { return X.map(x => this.predictOne(x)); }

  accuracy(X, y) {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === y[i]).length / y.length;
  }
}

/* ═══════════════════════════════════════════════════════════
   PERCEPTRÓN (Single Neuron with Sigmoid activation)
   ═══════════════════════════════════════════════════════════ */
class Perceptron {
  constructor(lr = 0.1, epochs = 100) {
    this.lr = lr;
    this.epochs = epochs;
    this.weights = null;
    this.bias = 0;
    this.lossHistory = [];
  }

  sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  fit(X, y) {
    const n = X[0].length;
    this.weights = new Array(n).fill(0).map(() => (Math.random() - .5) * .1);
    this.bias = 0;
    this.lossHistory = [];

    for (let e = 0; e < this.epochs; e++) {
      let totalLoss = 0;
      for (let i = 0; i < X.length; i++) {
        const z = X[i].reduce((s, xi, j) => s + xi * this.weights[j], this.bias);
        const pred = this.sigmoid(z);
        const err = pred - y[i];
        totalLoss += -y[i] * Math.log(pred + 1e-9) - (1 - y[i]) * Math.log(1 - pred + 1e-9);
        for (let j = 0; j < n; j++) this.weights[j] -= this.lr * err * X[i][j];
        this.bias -= this.lr * err;
      }
      if (e % Math.max(1, Math.floor(this.epochs / 50)) === 0) {
        this.lossHistory.push(totalLoss / X.length);
      }
    }
  }

  predictProb(x) {
    const z = x.reduce((s, xi, j) => s + xi * this.weights[j], this.bias);
    return this.sigmoid(z);
  }

  predictOne(x) { return this.predictProb(x) >= .5 ? 1 : 0; }

  predict(X) { return X.map(x => this.predictOne(x)); }

  accuracy(X, y) {
    const preds = this.predict(X);
    return preds.filter((p, i) => p === y[i]).length / y.length;
  }
}
