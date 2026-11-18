import { CalibrationBin, FEATURES, FeatureImportanceItem } from './types';

const NUM_FEATURES = FEATURES.length;
const NUM_BINS = 64;

interface TreeNode {
  isLeaf: boolean;
  leafValue: number;
  featureIdx: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

interface IsotonicBlock {
  xMin: number;
  xMax: number;
  sumY: number;
  count: number;
  value: number;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function logit(p: number): number {
  const clamped = Math.min(1 - 1e-7, Math.max(1e-7, p));
  return Math.log(clamped / (1 - clamped));
}

function binaryLogLoss(y: Uint8Array, start: number, end: number, preds: Float64Array): number {
  let total = 0;
  const n = end - start;
  for (let i = 0; i < n; i++) {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, preds[i]));
    const target = y[start + i];
    total -= target === 1 ? Math.log(p) : Math.log(1 - p);
  }
  return total / Math.max(1, n);
}

function computeQuantileThresholds(
  X: Float64Array,
  nTrain: number
): Float64Array[] {
  const sampleSize = Math.min(nTrain, 12000);
  const step = Math.max(1, Math.floor(nTrain / sampleSize));
  const thresholds: Float64Array[] = new Array(NUM_FEATURES);

  for (let f = 0; f < NUM_FEATURES; f++) {
    const vals: number[] = [];
    for (let i = 0; i < nTrain; i += step) {
      vals.push(X[i * NUM_FEATURES + f]);
    }
    vals.sort((a, b) => a - b);

    const cuts: number[] = [];
    for (let b = 1; b < NUM_BINS; b++) {
      const idx = Math.floor((b * vals.length) / NUM_BINS);
      const v = vals[idx];
      if (cuts.length === 0 || v > cuts[cuts.length - 1]) {
        cuts.push(v);
      }
    }
    thresholds[f] = Float64Array.from(cuts);
  }
  return thresholds;
}

function binFeatureMatrix(
  X: Float64Array,
  nRows: number,
  thresholds: Float64Array[]
): Uint8Array {
  const binned = new Uint8Array(nRows * NUM_FEATURES);
  for (let i = 0; i < nRows; i++) {
    const base = i * NUM_FEATURES;
    for (let f = 0; f < NUM_FEATURES; f++) {
      const cuts = thresholds[f];
      const v = X[base + f];
      let lo = 0;
      let hi = cuts.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (v <= cuts[mid]) hi = mid;
        else lo = mid + 1;
      }
      binned[base + f] = lo;
    }
  }
  return binned;
}

function fitIsotonicPAVA(probs: Float64Array, y: Uint8Array, yOffset: number): IsotonicBlock[] {
  const n = probs.length;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => probs[a] - probs[b]);

  const blocks: IsotonicBlock[] = [];
  for (let i = 0; i < n; i++) {
    const idx = order[i];
    const x = probs[idx];
    const target = y[yOffset + idx];
    blocks.push({
      xMin: x,
      xMax: x,
      sumY: target,
      count: 1,
      value: target,
    });
    while (
      blocks.length >= 2 &&
      blocks[blocks.length - 2].value > blocks[blocks.length - 1].value
    ) {
      const last = blocks.pop()!;
      const prev = blocks[blocks.length - 1];
      prev.xMax = last.xMax;
      prev.sumY += last.sumY;
      prev.count += last.count;
      prev.value = prev.sumY / prev.count;
    }
  }
  return blocks;
}

function evalIsotonic(blocks: IsotonicBlock[], p: number): number {
  if (blocks.length === 0) return p;
  if (p <= blocks[0].xMax) return blocks[0].value;
  const last = blocks[blocks.length - 1];
  if (p >= last.xMin) return last.value;

  let lo = 0;
  let hi = blocks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (p <= blocks[mid].xMax) hi = mid;
    else lo = mid + 1;
  }
  return blocks[lo].value;
}

// Seeded deterministic PRNG (Mulberry32) so training is reproducible across runs
function createRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export class CalibratedGBDT {
  private baseScore: number;
  private learningRate: number;
  private trees: TreeNode[] = [];
  private isotonicBlocks: IsotonicBlock[] = [];
  private featureGain = new Float64Array(NUM_FEATURES);

  constructor(baseScore: number, learningRate: number) {
    this.baseScore = baseScore;
    this.learningRate = learningRate;
  }

  predictRawProb(row: Float64Array | number[]): number {
    let score = this.baseScore;
    for (let t = 0; t < this.trees.length; t++) {
      let node: TreeNode | null = this.trees[t];
      while (node && !node.isLeaf) {
        node = row[node.featureIdx] <= node.threshold ? node.left : node.right;
      }
      if (node) score += this.learningRate * node.leafValue;
    }
    return sigmoid(score);
  }

  predictCalibratedProb(row: Float64Array | number[]): number {
    const raw = this.predictRawProb(row);
    return Math.min(0.995, Math.max(0.005, evalIsotonic(this.isotonicBlocks, raw)));
  }

  predictCalibratedBatch(X: Float64Array, startRow: number, endRow: number): Float64Array {
    const n = endRow - startRow;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const base = (startRow + i) * NUM_FEATURES;
      let score = this.baseScore;
      for (let t = 0; t < this.trees.length; t++) {
        let node: TreeNode | null = this.trees[t];
        while (node && !node.isLeaf) {
          node = X[base + node.featureIdx] <= node.threshold ? node.left : node.right;
        }
        if (node) score += this.learningRate * node.leafValue;
      }
      const raw = sigmoid(score);
      out[i] = Math.min(0.995, Math.max(0.005, evalIsotonic(this.isotonicBlocks, raw)));
    }
    return out;
  }

  getFeatureImportance(): FeatureImportanceItem[] {
    let total = 0;
    for (let f = 0; f < NUM_FEATURES; f++) total += this.featureGain[f];
    const items: FeatureImportanceItem[] = FEATURES.map((feature, idx) => ({
      feature,
      importance: total > 0 ? this.featureGain[idx] / total : 1 / NUM_FEATURES,
    }));
    items.sort((a, b) => b.importance - a.importance);
    return items;
  }

  getTreeCount(): number {
    return this.trees.length;
  }

  static train(
    X: Float64Array,
    y: Uint8Array,
    nMatches: number
  ): {
    model: CalibratedGBDT;
    i1: number;
    i2: number;
    accuracy: number;
    logLoss: number;
    brierScore: number;
    calibrationBins: CalibrationBin[];
    featureImportance: FeatureImportanceItem[];
  } {
    const i1 = Math.floor(nMatches * 0.7) * 2;
    const i2 = Math.floor(nMatches * 0.85) * 2;
    const totalRows = nMatches * 2;

    const maxDepth = 4;
    const learningRate = 0.05;
    const nEstimators = 180;
    const earlyStoppingRounds = 20;
    const regLambda = 1.5;
    const minChildWeight = 6.0;
    const colsample = 0.8;
    const subsample = 0.8;

    const rng = createRng(42);
    const thresholds = computeQuantileThresholds(X, i1);
    const binnedX = binFeatureMatrix(X, i1, thresholds);

    // Balanced mirrored dataset has p0 = 0.5 -> baseScore = 0
    const baseScore = logit(0.5);
    const model = new CalibratedGBDT(baseScore, learningRate);

    const trainScores = new Float64Array(i1);
    trainScores.fill(baseScore);
    const valLen = i2 - i1;
    const valScores = new Float64Array(valLen);
    valScores.fill(baseScore);
    const valProbs = new Float64Array(valLen);

    const grad = new Float64Array(i1);
    const hess = new Float64Array(i1);

    // Reusable buffers for histogram building
    const histG = new Float64Array(NUM_BINS);
    const histH = new Float64Array(NUM_BINS);
    const rowIndices = new Int32Array(i1);

    let bestValLoss = Infinity;
    let bestIter = 0;
    let noImprove = 0;

    for (let iter = 0; iter < nEstimators; iter++) {
      for (let i = 0; i < i1; i++) {
        const p = sigmoid(trainScores[i]);
        grad[i] = p - y[i];
        hess[i] = Math.max(1e-6, p * (1 - p));
      }

      // Row subsampling
      let activeCount = 0;
      for (let i = 0; i < i1; i++) {
        if (rng() < subsample) {
          rowIndices[activeCount++] = i;
        }
      }

      // Column subsampling
      const activeFeatures: number[] = [];
      for (let f = 0; f < NUM_FEATURES; f++) {
        if (rng() < colsample) activeFeatures.push(f);
      }
      if (activeFeatures.length === 0) activeFeatures.push(0);

      const buildNode = (start: number, end: number, depth: number): TreeNode => {
        let sumG = 0;
        let sumH = 0;
        for (let idx = start; idx < end; idx++) {
          const r = rowIndices[idx];
          sumG += grad[r];
          sumH += hess[r];
        }

        const leafVal = -sumG / (sumH + regLambda);
        if (depth >= maxDepth || end - start < 20 || sumH < minChildWeight * 2) {
          return {
            isLeaf: true,
            leafValue: leafVal,
            featureIdx: -1,
            threshold: 0,
            left: null,
            right: null,
          };
        }

        const parentScore = (sumG * sumG) / (sumH + regLambda);
        let bestGain = 1e-6;
        let bestFeature = -1;
        let bestBin = -1;

        for (let fi = 0; fi < activeFeatures.length; fi++) {
          const f = activeFeatures[fi];
          const nBins = thresholds[f].length + 1;
          histG.fill(0, 0, nBins);
          histH.fill(0, 0, nBins);

          for (let idx = start; idx < end; idx++) {
            const r = rowIndices[idx];
            const b = binnedX[r * NUM_FEATURES + f];
            histG[b] += grad[r];
            histH[b] += hess[r];
          }

          let leftG = 0;
          let leftH = 0;
          for (let b = 0; b < nBins - 1; b++) {
            leftG += histG[b];
            leftH += histH[b];
            const rightG = sumG - leftG;
            const rightH = sumH - leftH;
            if (leftH < minChildWeight || rightH < minChildWeight) continue;

            const gain =
              0.5 *
              ((leftG * leftG) / (leftH + regLambda) +
                (rightG * rightG) / (rightH + regLambda) -
                parentScore);
            if (gain > bestGain) {
              bestGain = gain;
              bestFeature = f;
              bestBin = b;
            }
          }
        }

        if (bestFeature === -1) {
          return {
            isLeaf: true,
            leafValue: leafVal,
            featureIdx: -1,
            threshold: 0,
            left: null,
            right: null,
          };
        }

        model.featureGain[bestFeature] += bestGain;
        const threshold = thresholds[bestFeature][bestBin];

        // In-place Hoare partition of rowIndices[start..end)
        let lo = start;
        let hi = end - 1;
        while (lo <= hi) {
          while (lo <= hi && binnedX[rowIndices[lo] * NUM_FEATURES + bestFeature] <= bestBin) {
            lo++;
          }
          while (lo <= hi && binnedX[rowIndices[hi] * NUM_FEATURES + bestFeature] > bestBin) {
            hi--;
          }
          if (lo < hi) {
            const tmp = rowIndices[lo];
            rowIndices[lo] = rowIndices[hi];
            rowIndices[hi] = tmp;
            lo++;
            hi--;
          }
        }
        const mid = lo;

        if (mid === start || mid === end) {
          return {
            isLeaf: true,
            leafValue: leafVal,
            featureIdx: -1,
            threshold: 0,
            left: null,
            right: null,
          };
        }

        return {
          isLeaf: false,
          leafValue: 0,
          featureIdx: bestFeature,
          threshold,
          left: buildNode(start, mid, depth + 1),
          right: buildNode(mid, end, depth + 1),
        };
      };

      const tree = buildNode(0, activeCount, 0);
      model.trees.push(tree);

      // Update trainScores across ALL training rows
      for (let i = 0; i < i1; i++) {
        let node: TreeNode | null = tree;
        const base = i * NUM_FEATURES;
        while (node && !node.isLeaf) {
          node = X[base + node.featureIdx] <= node.threshold ? node.left : node.right;
        }
        if (node) trainScores[i] += learningRate * node.leafValue;
      }

      // Update validation scores and check early stopping
      for (let i = 0; i < valLen; i++) {
        let node: TreeNode | null = tree;
        const base = (i1 + i) * NUM_FEATURES;
        while (node && !node.isLeaf) {
          node = X[base + node.featureIdx] <= node.threshold ? node.left : node.right;
        }
        if (node) valScores[i] += learningRate * node.leafValue;
        valProbs[i] = sigmoid(valScores[i]);
      }

      const valLoss = binaryLogLoss(y, i1, i2, valProbs);
      if (valLoss < bestValLoss - 1e-5) {
        bestValLoss = valLoss;
        bestIter = iter + 1;
        noImprove = 0;
      } else {
        noImprove++;
        if (noImprove >= earlyStoppingRounds) break;
      }
    }

    if (bestIter > 0 && bestIter < model.trees.length) {
      model.trees.length = bestIter;
    }

    // Recompute validation raw probabilities using bestIter trees for isotonic calibration
    const bestValRaw = new Float64Array(valLen);
    for (let i = 0; i < valLen; i++) {
      const base = (i1 + i) * NUM_FEATURES;
      let score = baseScore;
      for (let t = 0; t < model.trees.length; t++) {
        let node: TreeNode | null = model.trees[t];
        while (node && !node.isLeaf) {
          node = X[base + node.featureIdx] <= node.threshold ? node.left : node.right;
        }
        if (node) score += learningRate * node.leafValue;
      }
      bestValRaw[i] = sigmoid(score);
    }

    model.isotonicBlocks = fitIsotonicPAVA(bestValRaw, y, i1);

    // Evaluate on chronological test split [i2..totalRows)
    const testProbs = model.predictCalibratedBatch(X, i2, totalRows);
    const testLen = totalRows - i2;
    let correct = 0;
    let brierSum = 0;
    const binPredSum = new Float64Array(10);
    const binPosSum = new Float64Array(10);
    const binCounts = new Int32Array(10);

    for (let i = 0; i < testLen; i++) {
      const p = testProbs[i];
      const target = y[i2 + i];
      if ((p >= 0.5 ? 1 : 0) === target) correct++;
      const diff = p - target;
      brierSum += diff * diff;

      const bin = Math.min(9, Math.max(0, Math.floor(p * 10)));
      binPredSum[bin] += p;
      binPosSum[bin] += target;
      binCounts[bin]++;
    }

    const calibrationBins: CalibrationBin[] = [];
    for (let b = 0; b < 10; b++) {
      if (binCounts[b] > 0) {
        calibrationBins.push({
          binIndex: b,
          meanPredicted: binPredSum[b] / binCounts[b],
          fractionPositive: binPosSum[b] / binCounts[b],
          count: binCounts[b],
        });
      }
    }

    return {
      model,
      i1,
      i2,
      accuracy: correct / Math.max(1, testLen),
      logLoss: binaryLogLoss(y, i2, totalRows, testProbs),
      brierScore: brierSum / Math.max(1, testLen),
      calibrationBins,
      featureImportance: model.getFeatureImportance(),
    };
  }
}
