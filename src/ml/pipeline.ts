import { MatchedTestRow, attachOdds, kellyFraction, removeVig, runBacktest } from './backtest';
import { loadAllMatches, loadOddsForYears } from './dataLoader';
import { CalibratedGBDT } from './gbdt';
import {
  PreparedDataset,
  buildFeaturesObject,
  prepareMlDataset,
} from './stateTracker';
import {
  BacktestConfig,
  BacktestReport,
  EvaluationReport,
  FEATURES,
  OddsComparison,
  PipelineStatus,
  PlayerSummary,
  PredictionRequest,
  PredictionResponse,
} from './types';

interface PipelineArtifacts {
  dataset: PreparedDataset;
  model: CalibratedGBDT;
  i1: number;
  i2: number;
  evaluation: EvaluationReport;
  matchedOddsRows: MatchedTestRow[];
  defaultBacktest: BacktestReport;
  players: PlayerSummary[];
  playersByName: Map<string, PlayerSummary>;
}

const DEFAULT_BACKTEST_CONFIG: BacktestConfig = {
  kellyFrac: 0.25,
  minEdge: 0.03,
  startBankroll: 1000,
  maxBetPct: 0.05,
};

class TennisPipelineService {
  private status: PipelineStatus = {
    status: 'initializing',
    stage: 'Starting ML pipeline...',
  };
  private artifacts: PipelineArtifacts | null = null;
  private initPromise: Promise<void> | null = null;

  start(rootDir: string): void {
    if (!this.initPromise) {
      this.initPromise = this.runInit(rootDir);
    }
  }

  private async runInit(rootDir: string): Promise<void> {
    const t0 = Date.now();
    try {
      this.status = {
        status: 'initializing',
        stage: 'Loading ATP match CSVs (1968–2026) and merging 2026 feed...',
      };
      // Yield to event loop so Express stays responsive
      await new Promise((r) => setTimeout(r, 10));
      const matches = loadAllMatches(rootDir);

      this.status = {
        status: 'initializing',
        stage: `Building leak-free Elo, surface Elo, serve EWMA & H2H features for ${matches.length.toLocaleString()} matches...`,
      };
      await new Promise((r) => setTimeout(r, 10));
      const dataset = prepareMlDataset(matches);

      this.status = {
        status: 'initializing',
        stage: 'Training histogram GBDT classifier and isotonic calibration...',
      };
      await new Promise((r) => setTimeout(r, 10));
      const trainRes = CalibratedGBDT.train(dataset.X, dataset.y, dataset.nMatches);

      const m1 = Math.floor(trainRes.i1 / 2);
      const m2 = Math.floor(trainRes.i2 / 2);
      const fmtDate = (idx: number) =>
        new Date(dataset.matchDatesMs[Math.max(0, Math.min(dataset.nMatches - 1, idx))])
          .toISOString()
          .slice(0, 10);

      const evaluation: EvaluationReport = {
        totalMatches: dataset.nMatches,
        totalRows: dataset.nRows,
        trainMatches: m1,
        valMatches: m2 - m1,
        testMatches: dataset.nMatches - m2,
        trainDateRange: [fmtDate(0), fmtDate(m1 - 1)],
        valDateRange: [fmtDate(m1), fmtDate(m2 - 1)],
        testDateRange: [fmtDate(m2), fmtDate(dataset.nMatches - 1)],
        treesUsed: trainRes.model.getTreeCount(),
        accuracy: trainRes.accuracy,
        logLoss: trainRes.logLoss,
        brierScore: trainRes.brierScore,
        calibrationBins: trainRes.calibrationBins,
        featureImportance: trainRes.featureImportance,
        pipelineDurationMs: Date.now() - t0,
      };

      const testMinYear = dataset.matchYears[m2] ?? 2016;
      const testMaxYear = dataset.matchYears[dataset.nMatches - 1] ?? 2026;

      this.status = {
        status: 'initializing',
        stage: `Loading historical betting odds (${testMinYear}–${testMaxYear}) and running Kelly backtest...`,
      };
      await new Promise((r) => setTimeout(r, 10));
      const oddsRecords = loadOddsForYears(rootDir, testMinYear, testMaxYear);
      const matchedOddsRows = attachOdds(dataset, trainRes.i2, oddsRecords);
      const defaultBacktest = runBacktest(
        dataset,
        trainRes.i2,
        matchedOddsRows,
        trainRes.model,
        DEFAULT_BACKTEST_CONFIG
      );

      const players = dataset.state
        .getAllPlayerSummaries()
        .sort((a, b) => b.overallElo - a.overallElo);
      const playersByName = new Map(players.map((p) => [p.name.toLowerCase(), p]));

      evaluation.pipelineDurationMs = Date.now() - t0;

      this.artifacts = {
        dataset,
        model: trainRes.model,
        i1: trainRes.i1,
        i2: trainRes.i2,
        evaluation,
        matchedOddsRows,
        defaultBacktest,
        players,
        playersByName,
      };
      this.status = { status: 'ready', stage: 'Pipeline ready' };
      console.log(
        `[TennisML] Pipeline ready in ${evaluation.pipelineDurationMs}ms | ` +
          `${evaluation.totalMatches} matches | Acc ${(evaluation.accuracy * 100).toFixed(2)}% | ` +
          `LogLoss ${evaluation.logLoss.toFixed(4)} | Backtest bets ${defaultBacktest.nBets} (ROI ${(defaultBacktest.roi * 100).toFixed(2)}%)`
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[TennisML] Initialization failed:', err);
      this.status = { status: 'error', stage: 'Pipeline failed', error: msg };
    }
  }

  getStatus(): PipelineStatus {
    return this.status;
  }

  async ensureReady(): Promise<PipelineArtifacts> {
    if (this.initPromise) await this.initPromise;
    if (!this.artifacts) {
      throw new Error(this.status.error ?? 'Pipeline not initialized');
    }
    return this.artifacts;
  }

  async getSummary() {
    const art = await this.ensureReady();
    return {
      evaluation: art.evaluation,
      backtest: art.defaultBacktest,
      defaultBacktestConfig: DEFAULT_BACKTEST_CONFIG,
      topPlayers: art.players.slice(0, 25),
      totalPlayers: art.players.length,
    };
  }

  async searchPlayers(query: string, limit = 20): Promise<PlayerSummary[]> {
    const art = await this.ensureReady();
    const q = query.trim().toLowerCase();
    if (!q) {
      return art.players.slice(0, limit);
    }
    const matches: PlayerSummary[] = [];
    for (const p of art.players) {
      if (p.name.toLowerCase().includes(q)) {
        matches.push(p);
        if (matches.length >= limit) break;
      }
    }
    return matches;
  }

  async runCustomBacktest(config: BacktestConfig): Promise<BacktestReport> {
    const art = await this.ensureReady();
    return runBacktest(
      art.dataset,
      art.i2,
      art.matchedOddsRows,
      art.model,
      config
    );
  }

  async predictMatch(req: PredictionRequest): Promise<PredictionResponse> {
    const art = await this.ensureReady();
    const st = art.dataset.state;
    const todayMs = Date.now();

    const p1Meta = art.playersByName.get(req.player1.trim().toLowerCase());
    const p2Meta = art.playersByName.get(req.player2.trim().toLowerCase());
    const p1Name = p1Meta?.name ?? req.player1.trim();
    const p2Name = p2Meta?.name ?? req.player2.trim();

    const rank1 = req.rank1 && req.rank1 > 0 ? req.rank1 : p1Meta?.latestRank ?? 500;
    const rank2 = req.rank2 && req.rank2 > 0 ? req.rank2 : p2Meta?.latestRank ?? 500;
    const age1 = req.age1 && req.age1 > 0 ? req.age1 : p1Meta?.latestAge ?? 26;
    const age2 = req.age2 && req.age2 > 0 ? req.age2 : p2Meta?.latestAge ?? 26;

    const a = st.getState(p1Name, req.surface, todayMs, rank1, age1);
    const b = st.getState(p2Name, req.surface, todayMs, rank2, age2);
    const [aWins, bWins, total] = st.h2hRecord(p1Name, p2Name);
    a.h2hWins = aWins;
    a.h2hTotal = total;
    b.h2hWins = bWins;
    b.h2hTotal = total;

    const featureMap = buildFeaturesObject(a, b, req.bestOf);
    const rowVec = FEATURES.map((f) => featureMap[f]);

    const rawProbP1 = art.model.predictRawProb(rowVec);
    const probP1 = art.model.predictCalibratedProb(rowVec);
    const probP2 = 1 - probP1;

    let overround: number | null = null;
    let oddsComparison: OddsComparison[] | null = null;

    if (
      req.oddsP1 &&
      req.oddsP2 &&
      req.oddsP1 > 1.01 &&
      req.oddsP2 > 1.01
    ) {
      const vig = removeVig(req.oddsP1, req.oddsP2);
      overround = vig.overround;

      const buildSide = (
        player: string,
        modelProb: number,
        fairMarketProb: number,
        odds: number
      ): OddsComparison => {
        const edge = modelProb - fairMarketProb;
        const fullK = kellyFraction(modelProb, odds);
        return {
          player,
          modelProb,
          fairMarketProb,
          odds,
          edge,
          isValue: edge > 0.03,
          kellyFull: fullK,
          kellyQuarter: fullK * 0.25,
        };
      };

      oddsComparison = [
        buildSide(p1Name, probP1, vig.fairP1, req.oddsP1),
        buildSide(p2Name, probP2, vig.fairP2, req.oddsP2),
      ];
    }

    return {
      player1: p1Name,
      player2: p2Name,
      surface: req.surface,
      bestOf: req.bestOf,
      probPlayer1: probP1,
      probPlayer2: probP2,
      rawProbPlayer1: rawProbP1,
      h2h: {
        player1Wins: aWins,
        player2Wins: bWins,
        total,
      },
      player1State: {
        overallElo: a.elo,
        surfaceElo: a.selo,
        rank: rank1,
        age: age1,
        fatigue14d: a.fatigue,
        matchesPlayed: st.matchCount.get(p1Name) ?? 0,
        serve: a.srv,
      },
      player2State: {
        overallElo: b.elo,
        surfaceElo: b.selo,
        rank: rank2,
        age: age2,
        fatigue14d: b.fatigue,
        matchesPlayed: st.matchCount.get(p2Name) ?? 0,
        serve: b.srv,
      },
      features: featureMap,
      overround,
      oddsComparison,
    };
  }
}

export const tennisPipeline = new TennisPipelineService();
