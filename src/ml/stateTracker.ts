import {
  FEATURES,
  MatchRecord,
  PlayerSummary,
  SURFACES,
  ServeStats,
  Surface,
} from './types';

export const DEFAULT_ELO = 1500;
export const EWMA_ALPHA = 0.2;
export const DEFAULT_SRV: ServeStats = {
  ace_rate: 0.07,
  df_rate: 0.04,
  first_in: 0.62,
  first_won: 0.72,
  second_won: 0.5,
};

const MS_PER_DAY = 86_400_000;
const NUM_FEATURES = FEATURES.length;

export function expectedScore(ratingA: number, ratingB: number): number {
  return 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
}

export function updateElo(
  ratingA: number,
  ratingB: number,
  scoreA: number,
  k: number
): [number, number] {
  const expA = expectedScore(ratingA, ratingB);
  return [
    ratingA + k * (scoreA - expA),
    ratingB + k * (1 - scoreA - (1 - expA)),
  ];
}

export function dynamicK(nMatches: number): number {
  return 250 / (nMatches + 5) ** 0.4;
}

export interface PlayerSnapshot {
  elo: number;
  selo: number;
  fatigue: number;
  rank: number;
  age: number;
  srv: ServeStats;
  h2hWins: number;
  h2hTotal: number;
}

export function buildFeaturesObject(
  a: PlayerSnapshot,
  b: PlayerSnapshot,
  bestOf: number
): Record<(typeof FEATURES)[number], number> {
  return {
    elo_diff: a.elo - b.elo,
    surface_elo_diff: a.selo - b.selo,
    rank_diff: Math.log(b.rank) - Math.log(a.rank),
    fatigue_diff: a.fatigue - b.fatigue,
    age_diff: a.age - b.age,
    ace_rate_diff: a.srv.ace_rate - b.srv.ace_rate,
    first_won_diff: a.srv.first_won - b.srv.first_won,
    second_won_diff: a.srv.second_won - b.srv.second_won,
    first_in_diff: a.srv.first_in - b.srv.first_in,
    serve_dominance: a.srv.first_won - a.srv.second_won,
    h2h_winrate: (a.h2hWins + 2) / (a.h2hTotal + 4),
    h2h_confidence: a.h2hTotal,
    best_of: bestOf,
  };
}

function writeFeaturesRow(
  dest: Float64Array,
  rowOffset: number,
  a: PlayerSnapshot,
  b: PlayerSnapshot,
  bestOf: number
): void {
  dest[rowOffset + 0] = a.elo - b.elo;
  dest[rowOffset + 1] = a.selo - b.selo;
  dest[rowOffset + 2] = Math.log(b.rank) - Math.log(a.rank);
  dest[rowOffset + 3] = a.fatigue - b.fatigue;
  dest[rowOffset + 4] = a.age - b.age;
  dest[rowOffset + 5] = a.srv.ace_rate - b.srv.ace_rate;
  dest[rowOffset + 6] = a.srv.first_won - b.srv.first_won;
  dest[rowOffset + 7] = a.srv.second_won - b.srv.second_won;
  dest[rowOffset + 8] = a.srv.first_in - b.srv.first_in;
  dest[rowOffset + 9] = a.srv.first_won - a.srv.second_won;
  dest[rowOffset + 10] = (a.h2hWins + 2) / (a.h2hTotal + 4);
  dest[rowOffset + 11] = a.h2hTotal;
  dest[rowOffset + 12] = bestOf;
}

export class StateTracker {
  overallElo = new Map<string, number>();
  surfaceElo: Record<Surface, Map<string, number>> = {
    Hard: new Map(),
    Clay: new Map(),
    Grass: new Map(),
    Carpet: new Map(),
  };
  matchCount = new Map<string, number>();
  matchDates = new Map<string, number[]>();
  serve = new Map<string, ServeStats>();
  h2h = new Map<string, Map<string, number>>();
  latestRank = new Map<string, number>();
  latestAge = new Map<string, number>();
  latestDateMs = new Map<string, number>();

  getState(
    player: string,
    surface: Surface,
    dateMs: number,
    rank: number,
    age: number
  ): PlayerSnapshot {
    const elo = this.overallElo.get(player) ?? DEFAULT_ELO;
    const selo = this.surfaceElo[surface].get(player) ?? elo;
    const dates = this.matchDates.get(player);
    let fatigue = 0;
    if (dates) {
      for (let i = 0; i < dates.length; i++) {
        const diffDays = (dateMs - dates[i]) / MS_PER_DAY;
        if (diffDays >= 0 && diffDays <= 14) fatigue++;
      }
    }
    const srv = this.serve.get(player) ?? DEFAULT_SRV;
    return {
      elo,
      selo,
      fatigue,
      rank: rank > 0 ? rank : 500,
      age,
      srv: { ...srv },
      h2hWins: 0,
      h2hTotal: 0,
    };
  }

  h2hRecord(a: string, b: string): [number, number, number] {
    const key = a < b ? `${a}|||${b}` : `${b}|||${a}`;
    const rec = this.h2h.get(key);
    if (!rec) return [0, 0, 0];
    const aWins = rec.get(a) ?? 0;
    const bWins = rec.get(b) ?? 0;
    return [aWins, bWins, aWins + bWins];
  }

  updateAfterMatch(r: MatchRecord): void {
    const { winnerName: w, loserName: l, surface, dateMs } = r;

    const wElo = this.overallElo.get(w) ?? DEFAULT_ELO;
    const lElo = this.overallElo.get(l) ?? DEFAULT_ELO;
    const wSelo = this.surfaceElo[surface].get(w) ?? wElo;
    const lSelo = this.surfaceElo[surface].get(l) ?? lElo;

    const k =
      (dynamicK(this.matchCount.get(w) ?? 0) +
        dynamicK(this.matchCount.get(l) ?? 0)) /
      2;

    const [nwElo, nlElo] = updateElo(wElo, lElo, 1, k);
    const [nwSelo, nlSelo] = updateElo(wSelo, lSelo, 1, k);

    this.overallElo.set(w, nwElo);
    this.overallElo.set(l, nlElo);
    this.surfaceElo[surface].set(w, nwSelo);
    this.surfaceElo[surface].set(l, nlSelo);

    this.matchCount.set(w, (this.matchCount.get(w) ?? 0) + 1);
    this.matchCount.set(l, (this.matchCount.get(l) ?? 0) + 1);

    this.latestRank.set(w, r.winnerRank);
    this.latestRank.set(l, r.loserRank);
    this.latestAge.set(w, r.winnerAge);
    this.latestAge.set(l, r.loserAge);
    this.latestDateMs.set(w, dateMs);
    this.latestDateMs.set(l, dateMs);

    const sides: Array<{
      player: string;
      svpt: number | null;
      ace: number;
      df: number;
      firstIn: number;
      firstWon: number;
      secondWon: number;
    }> = [
      {
        player: w,
        svpt: r.wSvpt,
        ace: r.wAce,
        df: r.wDf,
        firstIn: r.w1stIn,
        firstWon: r.w1stWon,
        secondWon: r.w2ndWon,
      },
      {
        player: l,
        svpt: r.lSvpt,
        ace: r.lAce,
        df: r.lDf,
        firstIn: r.l1stIn,
        firstWon: r.l1stWon,
        secondWon: r.l2ndWon,
      },
    ];

    for (const s of sides) {
      if (s.svpt !== null && s.svpt > 0) {
        const nextSrv: ServeStats = {
          ace_rate: s.ace / s.svpt,
          df_rate: s.df / s.svpt,
          first_in: s.firstIn / s.svpt,
          first_won: s.firstWon / Math.max(s.firstIn, 1),
          second_won: s.secondWon / Math.max(s.svpt - s.firstIn, 1),
        };
        const oldSrv = this.serve.get(s.player);
        if (!oldSrv) {
          this.serve.set(s.player, nextSrv);
        } else {
          this.serve.set(s.player, {
            ace_rate:
              EWMA_ALPHA * nextSrv.ace_rate + (1 - EWMA_ALPHA) * oldSrv.ace_rate,
            df_rate:
              EWMA_ALPHA * nextSrv.df_rate + (1 - EWMA_ALPHA) * oldSrv.df_rate,
            first_in:
              EWMA_ALPHA * nextSrv.first_in + (1 - EWMA_ALPHA) * oldSrv.first_in,
            first_won:
              EWMA_ALPHA * nextSrv.first_won + (1 - EWMA_ALPHA) * oldSrv.first_won,
            second_won:
              EWMA_ALPHA * nextSrv.second_won + (1 - EWMA_ALPHA) * oldSrv.second_won,
          });
        }
      }
    }

    for (const p of [w, l]) {
      const arr = this.matchDates.get(p) ?? [];
      arr.push(dateMs);
      const pruned: number[] = [];
      for (let i = 0; i < arr.length; i++) {
        if ((dateMs - arr[i]) / MS_PER_DAY <= 14) pruned.push(arr[i]);
      }
      this.matchDates.set(p, pruned);
    }

    const h2hKey = w < l ? `${w}|||${l}` : `${l}|||${w}`;
    let rec = this.h2h.get(h2hKey);
    if (!rec) {
      rec = new Map();
      this.h2h.set(h2hKey, rec);
    }
    rec.set(w, (rec.get(w) ?? 0) + 1);
    if (!rec.has(l)) rec.set(l, 0);
  }

  getAllPlayerSummaries(): PlayerSummary[] {
    const list: PlayerSummary[] = [];
    for (const [name, overallElo] of this.overallElo.entries()) {
      const surfaceElo = {} as Record<Surface, number>;
      for (const surf of SURFACES) {
        surfaceElo[surf] = this.surfaceElo[surf].get(name) ?? overallElo;
      }
      const dateMs = this.latestDateMs.get(name) ?? 0;
      list.push({
        name,
        overallElo,
        surfaceElo,
        matchCount: this.matchCount.get(name) ?? 0,
        latestRank: this.latestRank.get(name) ?? 500,
        latestAge: this.latestAge.get(name) ?? 26,
        lastMatchDate: dateMs ? new Date(dateMs).toISOString().slice(0, 10) : '',
        serve: { ...(this.serve.get(name) ?? DEFAULT_SRV) },
      });
    }
    return list;
  }
}

export interface PreparedDataset {
  nMatches: number;
  nRows: number;
  X: Float64Array;
  y: Uint8Array;
  matchDatesMs: Float64Array;
  matchYears: Int16Array;
  matchWinners: string[];
  matchLosers: string[];
  state: StateTracker;
}

export function prepareMlDataset(matches: MatchRecord[]): PreparedDataset {
  const nMatches = matches.length;
  const nRows = nMatches * 2;
  const X = new Float64Array(nRows * NUM_FEATURES);
  const y = new Uint8Array(nRows);
  const matchDatesMs = new Float64Array(nMatches);
  const matchYears = new Int16Array(nMatches);
  const matchWinners = new Array<string>(nMatches);
  const matchLosers = new Array<string>(nMatches);

  const state = new StateTracker();

  for (let i = 0; i < nMatches; i++) {
    const r = matches[i];
    const wState = state.getState(
      r.winnerName,
      r.surface,
      r.dateMs,
      r.winnerRank,
      r.winnerAge
    );
    const lState = state.getState(
      r.loserName,
      r.surface,
      r.dateMs,
      r.loserRank,
      r.loserAge
    );
    const [wH2h, lH2h, totalH2h] = state.h2hRecord(r.winnerName, r.loserName);
    wState.h2hWins = wH2h;
    wState.h2hTotal = totalH2h;
    lState.h2hWins = lH2h;
    lState.h2hTotal = totalH2h;

    const rowIdxW = i * 2;
    const rowIdxL = i * 2 + 1;
    writeFeaturesRow(X, rowIdxW * NUM_FEATURES, wState, lState, r.bestOf);
    y[rowIdxW] = 1;
    writeFeaturesRow(X, rowIdxL * NUM_FEATURES, lState, wState, r.bestOf);
    y[rowIdxL] = 0;

    matchDatesMs[i] = r.dateMs;
    matchYears[i] = r.year;
    matchWinners[i] = r.winnerName;
    matchLosers[i] = r.loserName;

    state.updateAfterMatch(r);
  }

  return {
    nMatches,
    nRows,
    X,
    y,
    matchDatesMs,
    matchYears,
    matchWinners,
    matchLosers,
    state,
  };
}
