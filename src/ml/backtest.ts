import { CalibratedGBDT } from './gbdt';
import { PreparedDataset } from './stateTracker';
import {
  BacktestBetSample,
  BacktestConfig,
  BacktestReport,
  EquityPoint,
  FEATURES,
  OddsRecord,
} from './types';

const NUM_FEATURES = FEATURES.length;

export function removeVig(
  oddsP1: number,
  oddsP2: number
): { fairP1: number; fairP2: number; overround: number } {
  const rawP1 = 1 / oddsP1;
  const rawP2 = 1 / oddsP2;
  const overround = rawP1 + rawP2;
  return {
    fairP1: rawP1 / overround,
    fairP2: rawP2 / overround,
    overround,
  };
}

export function closingLineValue(oddsTaken: number, closingOdds: number): number {
  const fairTaken = 1 / oddsTaken;
  const fairClose = 1 / closingOdds;
  return (fairClose - fairTaken) / fairTaken;
}

export function kellyFraction(prob: number, odds: number): number {
  if (odds <= 1) return 0;
  const b = odds - 1;
  const f = (prob * b - (1 - prob)) / b;
  return Math.max(0, f);
}

export function normalizePlayerKey(name: string): string {
  const parts = String(name).trim().split(/\s+/);
  if (parts.length < 2) return String(name).trim().toLowerCase();
  const last = parts.slice(1).join(' ');
  const firstInitial = parts[0][0];
  return `${last} ${firstInitial}.`.toLowerCase();
}

export interface MatchedTestRow {
  rowIdx: number; // positive-target row index in dataset.X
  dateStr: string;
  winnerName: string;
  loserName: string;
  oddsP1: number;
  oddsP2: number;
  closingOddsP1: number;
  closingOddsP2: number;
}

export function attachOdds(
  dataset: PreparedDataset,
  i2: number,
  oddsRecords: OddsRecord[]
): MatchedTestRow[] {
  const oddsMap = new Map<string, OddsRecord>();
  for (const o of oddsRecords) {
    oddsMap.set(`${o.wk}::${o.lk}::${o.year}`, o);
  }

  const matched: MatchedTestRow[] = [];
  const startMatchIdx = Math.floor(i2 / 2);

  for (let m = startMatchIdx; m < dataset.nMatches; m++) {
    const wk = normalizePlayerKey(dataset.matchWinners[m]);
    const lk = normalizePlayerKey(dataset.matchLosers[m]);
    const year = dataset.matchYears[m];
    const hit = oddsMap.get(`${wk}::${lk}::${year}`);
    if (!hit) continue;

    matched.push({
      rowIdx: m * 2, // positive orientation where player 1 won (target = 1)
      dateStr: new Date(dataset.matchDatesMs[m]).toISOString().slice(0, 10),
      winnerName: dataset.matchWinners[m],
      loserName: dataset.matchLosers[m],
      oddsP1: hit.b365w,
      oddsP2: hit.b365l,
      closingOddsP1: hit.psw,
      closingOddsP2: hit.psl,
    });
  }

  return matched;
}

export function runBacktest(
  dataset: PreparedDataset,
  i2: number,
  matchedRows: MatchedTestRow[],
  model: CalibratedGBDT,
  config: BacktestConfig
): BacktestReport {
  const totalTestMatches = dataset.nMatches - Math.floor(i2 / 2);
  let bankroll = config.startBankroll;
  let peak = bankroll;
  let maxDrawdown = 0;
  let nBets = 0;
  let nWins = 0;
  let totalStaked = 0;
  let totalPnl = 0;
  let clvSum = 0;

  const allBets: BacktestBetSample[] = [];
  const rowBuf = new Float64Array(NUM_FEATURES);

  for (let i = 0; i < matchedRows.length; i++) {
    const m = matchedRows[i];
    const base = m.rowIdx * NUM_FEATURES;
    for (let f = 0; f < NUM_FEATURES; f++) {
      rowBuf[f] = dataset.X[base + f];
    }

    const modelP1 = model.predictCalibratedProb(rowBuf);
    const modelP2 = 1 - modelP1;
    const { fairP1, fairP2 } = removeVig(m.oddsP1, m.oddsP2);

    const sides: Array<{
      playerBacked: string;
      opponent: string;
      sideP: number;
      fairP: number;
      odds: number;
      closeOdds: number;
      won: boolean;
    }> = [
      {
        playerBacked: m.winnerName,
        opponent: m.loserName,
        sideP: modelP1,
        fairP: fairP1,
        odds: m.oddsP1,
        closeOdds: m.closingOddsP1,
        won: true,
      },
      {
        playerBacked: m.loserName,
        opponent: m.winnerName,
        sideP: modelP2,
        fairP: fairP2,
        odds: m.oddsP2,
        closeOdds: m.closingOddsP2,
        won: false,
      },
    ];

    for (const s of sides) {
      const edge = s.sideP - s.fairP;
      if (edge < config.minEdge) continue;

      const f = kellyFraction(s.sideP, s.odds) * config.kellyFrac;
      const stake = Math.min(bankroll * f, bankroll * config.maxBetPct);
      if (stake <= 0) continue;

      const pnl = s.won ? stake * (s.odds - 1) : -stake;
      bankroll += pnl;
      totalStaked += stake;
      totalPnl += pnl;
      nBets++;
      if (s.won) nWins++;

      if (bankroll > peak) peak = bankroll;
      const dd = (bankroll - peak) / peak;
      if (dd < maxDrawdown) maxDrawdown = dd;

      const clv = closingLineValue(s.odds, s.closeOdds);
      clvSum += clv;

      allBets.push({
        date: m.dateStr,
        playerBacked: s.playerBacked,
        opponent: s.opponent,
        modelProb: s.sideP,
        fairProb: s.fairP,
        edge,
        odds: s.odds,
        closingOdds: s.closeOdds,
        clv,
        stake,
        won: s.won,
        pnl,
        bankrollAfter: bankroll,
      });
    }
  }

  // Downsample equity curve to max 220 points for fast charting
  const equityCurve: EquityPoint[] = [
    { betIndex: 0, bankroll: config.startBankroll, date: matchedRows[0]?.dateStr ?? '' },
  ];
  if (allBets.length > 0) {
    const stride = Math.max(1, Math.floor(allBets.length / 220));
    for (let i = 0; i < allBets.length; i += stride) {
      equityCurve.push({
        betIndex: i + 1,
        bankroll: allBets[i].bankrollAfter,
        date: allBets[i].date,
      });
    }
    const last = allBets[allBets.length - 1];
    if (equityCurve[equityCurve.length - 1].betIndex !== allBets.length) {
      equityCurve.push({
        betIndex: allBets.length,
        bankroll: last.bankrollAfter,
        date: last.date,
      });
    }
  }

  return {
    matchedMatches: matchedRows.length,
    totalTestMatches,
    matchRate: totalTestMatches > 0 ? matchedRows.length / totalTestMatches : 0,
    nBets,
    nWins,
    winRate: nBets > 0 ? nWins / nBets : 0,
    roi: totalStaked > 0 ? totalPnl / totalStaked : 0,
    totalStaked,
    totalPnl,
    startBankroll: config.startBankroll,
    finalBankroll: bankroll,
    maxDrawdown,
    avgClv: nBets > 0 ? clvSum / nBets : 0,
    equityCurve,
    recentBets: allBets.slice(-40).reverse(),
  };
}
