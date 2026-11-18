export const SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet'] as const;
export type Surface = (typeof SURFACES)[number];

export const FEATURES = [
  'elo_diff',
  'surface_elo_diff',
  'rank_diff',
  'fatigue_diff',
  'age_diff',
  'ace_rate_diff',
  'first_won_diff',
  'second_won_diff',
  'first_in_diff',
  'serve_dominance',
  'h2h_winrate',
  'h2h_confidence',
  'best_of',
] as const;

export type FeatureName = (typeof FEATURES)[number];

export interface ServeStats {
  ace_rate: number;
  df_rate: number;
  first_in: number;
  first_won: number;
  second_won: number;
}

export interface MatchRecord {
  tourneyId: string;
  tourneyName: string;
  surface: Surface;
  dateMs: number;
  year: number;
  matchNum: number;
  winnerName: string;
  loserName: string;
  winnerRank: number;
  loserRank: number;
  winnerAge: number;
  loserAge: number;
  bestOf: number;
  wSvpt: number | null;
  wAce: number;
  wDf: number;
  w1stIn: number;
  w1stWon: number;
  w2ndWon: number;
  lSvpt: number | null;
  lAce: number;
  lDf: number;
  l1stIn: number;
  l1stWon: number;
  l2ndWon: number;
}

export interface OddsRecord {
  wk: string;
  lk: string;
  year: number;
  b365w: number;
  b365l: number;
  psw: number;
  psl: number;
  tournament: string;
  dateStr: string;
}

export interface CalibrationBin {
  binIndex: number;
  meanPredicted: number;
  fractionPositive: number;
  count: number;
}

export interface FeatureImportanceItem {
  feature: FeatureName;
  importance: number;
}

export interface EvaluationReport {
  totalMatches: number;
  totalRows: number;
  trainMatches: number;
  valMatches: number;
  testMatches: number;
  trainDateRange: [string, string];
  valDateRange: [string, string];
  testDateRange: [string, string];
  treesUsed: number;
  accuracy: number;
  logLoss: number;
  brierScore: number;
  calibrationBins: CalibrationBin[];
  featureImportance: FeatureImportanceItem[];
  pipelineDurationMs: number;
}

export interface BacktestConfig {
  kellyFrac: number;
  minEdge: number;
  startBankroll: number;
  maxBetPct: number;
}

export interface EquityPoint {
  betIndex: number;
  bankroll: number;
  date: string;
}

export interface BacktestBetSample {
  date: string;
  playerBacked: string;
  opponent: string;
  modelProb: number;
  fairProb: number;
  edge: number;
  odds: number;
  closingOdds: number;
  clv: number;
  stake: number;
  won: boolean;
  pnl: number;
  bankrollAfter: number;
}

export interface BacktestReport {
  matchedMatches: number;
  totalTestMatches: number;
  matchRate: number;
  nBets: number;
  nWins: number;
  winRate: number;
  roi: number;
  totalStaked: number;
  totalPnl: number;
  startBankroll: number;
  finalBankroll: number;
  maxDrawdown: number;
  avgClv: number;
  equityCurve: EquityPoint[];
  recentBets: BacktestBetSample[];
}

export interface PlayerSummary {
  name: string;
  overallElo: number;
  surfaceElo: Record<Surface, number>;
  matchCount: number;
  latestRank: number;
  latestAge: number;
  lastMatchDate: string;
  serve: ServeStats;
}

export interface PredictionRequest {
  player1: string;
  player2: string;
  surface: Surface;
  bestOf: number;
  rank1?: number;
  rank2?: number;
  age1?: number;
  age2?: number;
  oddsP1?: number | null;
  oddsP2?: number | null;
}

export interface OddsComparison {
  player: string;
  modelProb: number;
  fairMarketProb: number;
  odds: number;
  edge: number;
  isValue: boolean;
  kellyFull: number;
  kellyQuarter: number;
}

export interface PredictionResponse {
  player1: string;
  player2: string;
  surface: Surface;
  bestOf: number;
  probPlayer1: number;
  probPlayer2: number;
  rawProbPlayer1: number;
  h2h: {
    player1Wins: number;
    player2Wins: number;
    total: number;
  };
  player1State: {
    overallElo: number;
    surfaceElo: number;
    rank: number;
    age: number;
    fatigue14d: number;
    matchesPlayed: number;
    serve: ServeStats;
  };
  player2State: {
    overallElo: number;
    surfaceElo: number;
    rank: number;
    age: number;
    fatigue14d: number;
    matchesPlayed: number;
    serve: ServeStats;
  };
  features: Record<FeatureName, number>;
  overround: number | null;
  oddsComparison: OddsComparison[] | null;
}

export interface PipelineStatus {
  status: 'initializing' | 'ready' | 'error';
  stage: string;
  error?: string;
}
