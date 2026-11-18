import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRightLeft,
  BarChart3,
  Calculator,
  CheckCircle2,
  Loader2,
  Search,
  TrendingUp,
  Trophy,
} from 'lucide-react';
import type {
  BacktestConfig,
  BacktestReport,
  EvaluationReport,
  PipelineStatus,
  PlayerSummary,
  PredictionResponse,
  Surface,
} from './ml/types';

interface SummaryPayload {
  evaluation: EvaluationReport;
  backtest: BacktestReport;
  defaultBacktestConfig: BacktestConfig;
  topPlayers: PlayerSummary[];
  totalPlayers: number;
}

type ActiveTab = 'predict' | 'evaluate' | 'backtest' | 'players';

const SURFACE_OPTIONS: Surface[] = ['Hard', 'Clay', 'Grass', 'Carpet'];

const FEATURE_LABELS: Record<string, string> = {
  elo_diff: 'Overall Elo Diff',
  surface_elo_diff: 'Surface Elo Diff',
  rank_diff: 'Log Rank Diff',
  fatigue_diff: '14d Fatigue Diff',
  age_diff: 'Age Diff',
  ace_rate_diff: 'Ace Rate EWMA Diff',
  first_won_diff: '1st Serve Won Diff',
  second_won_diff: '2nd Serve Won Diff',
  first_in_diff: '1st Serve In% Diff',
  serve_dominance: 'P1 Serve Dominance',
  h2h_winrate: 'Smoothed H2H Winrate',
  h2h_confidence: 'H2H Match Count',
  best_of: 'Best-of Format',
};

function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

function signedPct(v: number, digits = 2): string {
  const s = (v * 100).toFixed(digits);
  return v >= 0 ? `+${s}%` : `${s}%`;
}

function PlayerSearchInput({
  label,
  value,
  onChange,
  onSelectPlayer,
}: {
  label: string;
  value: string;
  onChange: (val: string) => void;
  onSelectPlayer: (player: PlayerSummary) => void;
}) {
  const [results, setResults] = useState<PlayerSummary[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(
          `/api/players?q=${encodeURIComponent(value)}&limit=8`
        );
        if (res.ok) {
          const data = await res.json();
          setResults(data.players ?? []);
        }
      } catch {
        // ignore transient search errors
      }
    }, 120);
    return () => clearTimeout(timer);
  }, [value, open]);

  return (
    <div className="relative">
      <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">
        {label}
      </label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          value={value}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 160)}
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
          }}
          placeholder="Search ATP player..."
          className="w-full rounded-lg border border-slate-700 bg-slate-900 py-2.5 pl-9 pr-3 text-sm text-slate-100 placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
        />
      </div>
      {open && results.length > 0 && (
        <div className="absolute z-20 mt-1 max-h-60 w-full overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 shadow-xl">
          {results.map((p) => (
            <button
              key={p.name}
              type="button"
              onMouseDown={() => {
                onSelectPlayer(p);
                setOpen(false);
              }}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-800"
            >
              <div>
                <div className="font-medium text-slate-100">{p.name}</div>
                <div className="text-xs text-slate-400">
                  Rank #{p.latestRank} · {p.matchCount} matches · Last {p.lastMatchDate}
                </div>
              </div>
              <span className="font-mono text-xs text-emerald-400">
                Elo {Math.round(p.overallElo)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function CalibrationChart({ bins }: { bins: EvaluationReport['calibrationBins'] }) {
  const w = 440;
  const h = 260;
  const pad = 38;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  const toX = (p: number) => pad + p * plotW;
  const toY = (p: number) => pad + (1 - p) * plotH;

  const points = bins
    .map((b) => `${toX(b.meanPredicted).toFixed(1)},${toY(b.fractionPositive).toFixed(1)}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-lg">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line
            x1={toX(0)}
            y1={toY(t)}
            x2={toX(1)}
            y2={toY(t)}
            stroke="#1e293b"
            strokeWidth={1}
          />
          <line
            x1={toX(t)}
            y1={toY(0)}
            x2={toX(t)}
            y2={toY(1)}
            stroke="#1e293b"
            strokeWidth={1}
          />
          <text
            x={pad - 8}
            y={toY(t) + 4}
            textAnchor="end"
            className="fill-slate-400 text-[10px]"
          >
            {Math.round(t * 100)}%
          </text>
          <text
            x={toX(t)}
            y={h - pad + 16}
            textAnchor="middle"
            className="fill-slate-400 text-[10px]"
          >
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      <line
        x1={toX(0)}
        y1={toY(0)}
        x2={toX(1)}
        y2={toY(1)}
        stroke="#475569"
        strokeDasharray="4 4"
        strokeWidth={1.5}
      />
      {bins.length > 1 && (
        <polyline
          fill="none"
          stroke="#10b981"
          strokeWidth={2.5}
          points={points}
        />
      )}
      {bins.map((b) => (
        <g key={b.binIndex}>
          <circle
            cx={toX(b.meanPredicted)}
            cy={toY(b.fractionPositive)}
            r={4.5}
            fill="#10b981"
            stroke="#022c22"
            strokeWidth={1.5}
          />
          <title>
            Predicted {(b.meanPredicted * 100).toFixed(1)}% vs Observed{' '}
            {(b.fractionPositive * 100).toFixed(1)}% ({b.count} rows)
          </title>
        </g>
      ))}
      <text
        x={w / 2}
        y={h - 4}
        textAnchor="middle"
        className="fill-slate-400 text-[10px]"
      >
        Predicted Win Probability
      </text>
    </svg>
  );
}

function EquityCurveChart({
  curve,
  startBankroll,
}: {
  curve: BacktestReport['equityCurve'];
  startBankroll: number;
}) {
  if (curve.length < 2) {
    return (
      <div className="flex h-52 items-center justify-center text-sm text-slate-400">
        No bets placed under current thresholds.
      </div>
    );
  }

  const w = 640;
  const h = 240;
  const padL = 52;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const vals = curve.map((c) => c.bankroll);
  const minVal = Math.min(...vals, startBankroll) * 0.96;
  const maxVal = Math.max(...vals, startBankroll) * 1.04;
  const span = Math.max(1, maxVal - minVal);
  const maxIdx = Math.max(1, curve[curve.length - 1].betIndex);

  const toX = (idx: number) => padL + (idx / maxIdx) * plotW;
  const toY = (v: number) => padT + (1 - (v - minVal) / span) * plotH;

  const linePoints = curve
    .map((pt) => `${toX(pt.betIndex).toFixed(1)},${toY(pt.bankroll).toFixed(1)}`)
    .join(' ');

  const areaPoints = `${toX(curve[0].betIndex).toFixed(1)},${(padT + plotH).toFixed(
    1
  )} ${linePoints} ${toX(curve[curve.length - 1].betIndex).toFixed(1)},${(
    padT + plotH
  ).toFixed(1)}`;

  const finalUp = curve[curve.length - 1].bankroll >= startBankroll;
  const strokeColor = finalUp ? '#10b981' : '#f43f5e';
  const fillColor = finalUp ? 'rgba(16, 185, 129, 0.14)' : 'rgba(244, 63, 94, 0.14)';

  const yTicks = [minVal, (minVal + maxVal) / 2, maxVal];

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      {yTicks.map((tv, idx) => (
        <g key={idx}>
          <line
            x1={padL}
            y1={toY(tv)}
            x2={w - padR}
            y2={toY(tv)}
            stroke="#1e293b"
            strokeWidth={1}
          />
          <text
            x={padL - 8}
            y={toY(tv) + 4}
            textAnchor="end"
            className="fill-slate-400 text-[10px]"
          >
            {Math.round(tv)}
          </text>
        </g>
      ))}
      <line
        x1={padL}
        y1={toY(startBankroll)}
        x2={w - padR}
        y2={toY(startBankroll)}
        stroke="#475569"
        strokeDasharray="4 4"
        strokeWidth={1.2}
      />
      <polygon points={areaPoints} fill={fillColor} />
      <polyline fill="none" stroke={strokeColor} strokeWidth={2} points={linePoints} />
      <text x={padL} y={h - 6} className="fill-slate-400 text-[10px]">
        Bet #0 ({curve[0].date})
      </text>
      <text
        x={w - padR}
        y={h - 6}
        textAnchor="end"
        className="fill-slate-400 text-[10px]"
      >
        Bet #{maxIdx} ({curve[curve.length - 1].date})
      </text>
    </svg>
  );
}

export default function App() {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>({
    status: 'initializing',
    stage: 'Connecting to ML pipeline...',
  });
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('predict');

  // Prediction form state (prefilled with tennis.py live-prediction example)
  const [player1, setPlayer1] = useState('Zizou Bergs');
  const [player2, setPlayer2] = useState('Toby Samuel');
  const [surface, setSurface] = useState<Surface>('Grass');
  const [bestOf, setBestOf] = useState<3 | 5>(3);
  const [rank1, setRank1] = useState('48');
  const [rank2, setRank2] = useState('142');
  const [age1, setAge1] = useState('');
  const [age2, setAge2] = useState('');
  const [oddsP1, setOddsP1] = useState('1.37');
  const [oddsP2, setOddsP2] = useState('3.10');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Backtest custom controls
  const [btConfig, setBtConfig] = useState<BacktestConfig>({
    kellyFrac: 0.25,
    minEdge: 0.03,
    startBankroll: 1000,
    maxBetPct: 0.05,
  });
  const [btReport, setBtReport] = useState<BacktestReport | null>(null);
  const [btRunning, setBtRunning] = useState(false);

  // Poll pipeline readiness until ready
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const st: PipelineStatus = await res.json();
        if (cancelled) return;
        setPipelineStatus(st);
        if (st.status === 'ready') {
          const sumRes = await fetch('/api/summary');
          if (sumRes.ok && !cancelled) {
            const payload: SummaryPayload = await sumRes.json();
            setSummary(payload);
            setBtReport(payload.backtest);
            setBtConfig(payload.defaultBacktestConfig);
          }
          return;
        }
        if (st.status === 'error') return;
      } catch {
        // retry on transient startup error
      }
      if (!cancelled) setTimeout(poll, 700);
    }
    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  // Run initial live prediction once summary is loaded
  useEffect(() => {
    if (summary && !prediction) {
      handlePredict();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary]);

  async function handlePredict(e?: React.FormEvent) {
    if (e) e.preventDefault();
    setPredicting(true);
    setPredictError(null);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player1,
          player2,
          surface,
          bestOf,
          rank1: rank1 ? Number(rank1) : undefined,
          rank2: rank2 ? Number(rank2) : undefined,
          age1: age1 ? Number(age1) : undefined,
          age2: age2 ? Number(age2) : undefined,
          oddsP1: oddsP1 ? Number(oddsP1) : null,
          oddsP2: oddsP2 ? Number(oddsP2) : null,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setPredictError(data.error ?? 'Prediction request failed');
      } else {
        setPrediction(data);
      }
    } catch (err) {
      setPredictError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setPredicting(false);
    }
  }

  async function handleBacktest(e: React.FormEvent) {
    e.preventDefault();
    setBtRunning(true);
    try {
      const res = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(btConfig),
      });
      if (res.ok) {
        const data: BacktestReport = await res.json();
        setBtReport(data);
      }
    } finally {
      setBtRunning(false);
    }
  }

  function swapPlayers() {
    setPlayer1(player2);
    setPlayer2(player1);
    setRank1(rank2);
    setRank2(rank1);
    setAge1(age2);
    setAge2(age1);
    setOddsP1(oddsP2);
    setOddsP2(oddsP1);
  }

  const maxImportance = useMemo(() => {
    if (!summary) return 1;
    return Math.max(
      0.01,
      ...summary.evaluation.featureImportance.map((f) => f.importance)
    );
  }, [summary]);

  if (pipelineStatus.status === 'initializing' || (!summary && pipelineStatus.status !== 'error')) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-slate-950 px-4 text-center">
        <div className="max-w-md rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-2xl">
          <Loader2 className="mx-auto mb-4 h-10 w-10 animate-spin text-emerald-400" />
          <h1 className="text-lg font-semibold text-slate-100">
            Initializing ATP Tennis ML Pipeline
          </h1>
          <p className="mt-2 text-sm text-slate-400">{pipelineStatus.stage}</p>
          <p className="mt-4 text-xs text-slate-500">
            Parsing 1968–2026 ATP match CSVs, computing chronological Elo &amp; serve EWMA
            features, training GBDT + isotonic calibration, and matching historical odds.
          </p>
        </div>
      </div>
    );
  }

  if (pipelineStatus.status === 'error') {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
        <div className="max-w-lg rounded-2xl border border-rose-800/60 bg-rose-950/30 p-6 text-rose-200">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-5 w-5 text-rose-400" />
            Pipeline Initialization Error
          </div>
          <p className="mt-2 text-sm text-rose-300">{pipelineStatus.error}</p>
        </div>
      </div>
    );
  }

  const evalData = summary!.evaluation;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30">
                <Activity className="h-4 w-4" />
              </div>
              <h1 className="text-lg font-semibold tracking-tight text-slate-100">
                ATP Tennis Match ML Predictions
              </h1>
            </div>
            <p className="mt-1 text-xs text-slate-400">
              {evalData.totalMatches.toLocaleString()} ATP matches ({evalData.trainDateRange[0].slice(0, 4)}–
              {evalData.testDateRange[1].slice(0, 4)}) · {evalData.treesUsed} boosted trees +
              isotonic calibration · {summary!.totalPlayers.toLocaleString()} tracked players
            </p>
          </div>

          <nav className="flex flex-wrap gap-1.5 rounded-xl border border-slate-800 bg-slate-950/70 p-1">
            {(
              [
                ['predict', 'Match Predictor', Calculator],
                ['evaluate', 'Model & Calibration', BarChart3],
                ['backtest', 'Odds Backtest', TrendingUp],
                ['players', 'Elo Leaderboard', Trophy],
              ] as const
            ).map(([id, label, Icon]) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  activeTab === id
                    ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                    : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {activeTab === 'predict' && (
          <div className="grid gap-6 lg:grid-cols-12">
            <form
              onSubmit={handlePredict}
              className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900/50 p-5 lg:col-span-5"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
                  Matchup Parameters
                </h2>
                <button
                  type="button"
                  onClick={swapPlayers}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/70 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
                >
                  <ArrowRightLeft className="h-3 w-3" />
                  Swap
                </button>
              </div>

              <div className="grid gap-4">
                <PlayerSearchInput
                  label="Player 1"
                  value={player1}
                  onChange={setPlayer1}
                  onSelectPlayer={(p) => {
                    setPlayer1(p.name);
                    setRank1(String(p.latestRank));
                    setAge1(p.latestAge.toFixed(1));
                  }}
                />
                <PlayerSearchInput
                  label="Player 2"
                  value={player2}
                  onChange={setPlayer2}
                  onSelectPlayer={(p) => {
                    setPlayer2(p.name);
                    setRank2(String(p.latestRank));
                    setAge2(p.latestAge.toFixed(1));
                  }}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">
                    Surface
                  </label>
                  <select
                    value={surface}
                    onChange={(e) => setSurface(e.target.value as Surface)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  >
                    {SURFACE_OPTIONS.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">
                    Format
                  </label>
                  <select
                    value={bestOf}
                    onChange={(e) => setBestOf(Number(e.target.value) === 5 ? 5 : 3)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  >
                    <option value={3}>Best of 3</option>
                    <option value={5}>Best of 5</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-xs text-slate-400">P1 ATP Rank</label>
                  <input
                    type="number"
                    min={1}
                    max={2500}
                    value={rank1}
                    onChange={(e) => setRank1(e.target.value)}
                    placeholder="Auto"
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-slate-400">P2 ATP Rank</label>
                  <input
                    type="number"
                    min={1}
                    max={2500}
                    value={rank2}
                    onChange={(e) => setRank2(e.target.value)}
                    placeholder="Auto"
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-slate-400">P1 Age (opt.)</label>
                  <input
                    type="number"
                    step="0.1"
                    min={15}
                    max={50}
                    value={age1}
                    onChange={(e) => setAge1(e.target.value)}
                    placeholder="Latest"
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-slate-400">P2 Age (opt.)</label>
                  <input
                    type="number"
                    step="0.1"
                    min={15}
                    max={50}
                    value={age2}
                    onChange={(e) => setAge2(e.target.value)}
                    placeholder="Latest"
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
              </div>

              <div className="border-t border-slate-800 pt-4">
                <div className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">
                  Market Decimal Odds (Optional Value Check)
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="mb-1 block text-xs text-slate-400">P1 Odds</label>
                    <input
                      type="number"
                      step="0.01"
                      min={1.02}
                      value={oddsP1}
                      onChange={(e) => setOddsP1(e.target.value)}
                      placeholder="e.g. 1.37"
                      className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs text-slate-400">P2 Odds</label>
                    <input
                      type="number"
                      step="0.01"
                      min={1.02}
                      value={oddsP2}
                      onChange={(e) => setOddsP2(e.target.value)}
                      placeholder="e.g. 3.10"
                      className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              {predictError && (
                <div className="rounded-lg border border-rose-800/60 bg-rose-950/30 px-3 py-2 text-xs text-rose-300">
                  {predictError}
                </div>
              )}

              <button
                type="submit"
                disabled={predicting}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:opacity-60"
              >
                {predicting && <Loader2 className="h-4 w-4 animate-spin" />}
                Predict Match Outcome
              </button>
            </form>

            <div className="space-y-5 lg:col-span-7">
              {prediction && (
                <>
                  <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
                      <span>
                        {prediction.surface} · Best of {prediction.bestOf}
                      </span>
                      <span>
                        Head-to-Head: {prediction.player1} {prediction.h2h.player1Wins}–
                        {prediction.h2h.player2Wins} {prediction.player2}
                      </span>
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm font-medium text-slate-300">
                          {prediction.player1}
                        </div>
                        <div className="mt-1 font-mono text-3xl font-bold text-emerald-400">
                          {pct(prediction.probPlayer1, 1)}
                        </div>
                        <div className="mt-1 text-xs text-slate-400">
                          Elo {Math.round(prediction.player1State.overallElo)} ·{' '}
                          {prediction.surface} Elo{' '}
                          {Math.round(prediction.player1State.surfaceElo)} · Rank #
                          {prediction.player1State.rank}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-slate-300">
                          {prediction.player2}
                        </div>
                        <div className="mt-1 font-mono text-3xl font-bold text-sky-400">
                          {pct(prediction.probPlayer2, 1)}
                        </div>
                        <div className="mt-1 text-xs text-slate-400">
                          Elo {Math.round(prediction.player2State.overallElo)} ·{' '}
                          {prediction.surface} Elo{' '}
                          {Math.round(prediction.player2State.surfaceElo)} · Rank #
                          {prediction.player2State.rank}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="bg-emerald-500 transition-all duration-300"
                        style={{ width: `${prediction.probPlayer1 * 100}%` }}
                      />
                      <div
                        className="bg-sky-500 transition-all duration-300"
                        style={{ width: `${prediction.probPlayer2 * 100}%` }}
                      />
                    </div>
                  </div>

                  {prediction.oddsComparison && (
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
                      <div className="mb-3 flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-slate-200">
                          Market Vig-Removed Edge Comparison
                        </h3>
                        {prediction.overround !== null && (
                          <span className="font-mono text-xs text-slate-400">
                            Overround {((prediction.overround - 1) * 100).toFixed(2)}%
                          </span>
                        )}
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        {prediction.oddsComparison.map((side) => (
                          <div
                            key={side.player}
                            className={`rounded-xl border p-3.5 ${
                              side.isValue
                                ? 'border-emerald-500/40 bg-emerald-500/10'
                                : 'border-slate-800 bg-slate-950/60'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-slate-100">
                                {side.player}
                              </span>
                              <span className="font-mono text-xs text-slate-300">
                                @ {side.odds.toFixed(2)}
                              </span>
                            </div>
                            <div className="mt-2 space-y-1 text-xs text-slate-400">
                              <div className="flex justify-between">
                                <span>Model Prob</span>
                                <span className="font-mono text-slate-200">
                                  {pct(side.modelProb, 1)}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>No-Vig Market</span>
                                <span className="font-mono text-slate-200">
                                  {pct(side.fairMarketProb, 1)}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Edge</span>
                                <span
                                  className={`font-mono font-semibold ${
                                    side.isValue ? 'text-emerald-400' : 'text-slate-300'
                                  }`}
                                >
                                  {signedPct(side.edge, 1)}
                                </span>
                              </div>
                            </div>
                            <div className="mt-2.5 flex items-center justify-between border-t border-slate-800/80 pt-2 text-xs">
                              {side.isValue ? (
                                <span className="inline-flex items-center gap-1 font-medium text-emerald-400">
                                  <CheckCircle2 className="h-3.5 w-3.5" />
                                  VALUE (&gt;3% edge)
                                </span>
                              ) : (
                                <span className="text-slate-500">No Value (&le;3%)</span>
                              )}
                              <span className="font-mono text-slate-300">
                                0.25x Kelly: {pct(side.kellyQuarter, 2)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
                    <h3 className="mb-3 text-sm font-semibold text-slate-200">
                      Engineered Feature Vector (P1 − P2)
                    </h3>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
                      {Object.entries(prediction.features).map(([k, v]) => (
                        <div
                          key={k}
                          className="flex items-center justify-between border-b border-slate-800/70 py-1.5"
                        >
                          <span className="text-slate-400">
                            {FEATURE_LABELS[k] ?? k}
                          </span>
                          <span className="font-mono text-slate-200">
                            {Number.isInteger(v) ? v : v.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {activeTab === 'evaluate' && (
          <div className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Test Accuracy (50% threshold)</div>
                <div className="mt-1 font-mono text-2xl font-bold text-emerald-400">
                  {pct(evalData.accuracy, 2)}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {evalData.testMatches.toLocaleString()} out-of-sample matches
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Log Loss (Calibrated)</div>
                <div className="mt-1 font-mono text-2xl font-bold text-slate-100">
                  {evalData.logLoss.toFixed(4)}
                </div>
                <div className="mt-1 text-xs text-slate-500">Lower is better</div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Brier Score</div>
                <div className="mt-1 font-mono text-2xl font-bold text-slate-100">
                  {evalData.brierScore.toFixed(4)}
                </div>
                <div className="mt-1 text-xs text-slate-500">Mean squared prob error</div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Chronological Split (70/15/15)</div>
                <div className="mt-1 font-mono text-lg font-semibold text-slate-100">
                  {evalData.testDateRange[0].slice(0, 4)}–
                  {evalData.testDateRange[1].slice(0, 4)}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  Test window ({evalData.treesUsed} trees)
                </div>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
                <h2 className="text-sm font-semibold text-slate-200">
                  Isotonic Calibration Curve (Test Set)
                </h2>
                <p className="mt-1 text-xs text-slate-400">
                  Compares mean predicted probabilities per decile bin against observed match
                  win rates.
                </p>
                <div className="mt-4 flex justify-center">
                  <CalibrationChart bins={evalData.calibrationBins} />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
                <h2 className="text-sm font-semibold text-slate-200">
                  Feature Gain Importance
                </h2>
                <p className="mt-1 text-xs text-slate-400">
                  Relative split gain across all {evalData.treesUsed} gradient-boosted trees.
                </p>
                <div className="mt-4 space-y-2.5">
                  {evalData.featureImportance.map((item) => (
                    <div key={item.feature} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300">
                          {FEATURE_LABELS[item.feature] ?? item.feature}
                        </span>
                        <span className="font-mono text-slate-400">
                          {pct(item.importance, 1)}
                        </span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{
                            width: `${Math.max(2, (item.importance / maxImportance) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'backtest' && btReport && (
          <div className="space-y-6">
            <form
              onSubmit={handleBacktest}
              className="grid gap-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-5 sm:grid-cols-5 sm:items-end"
            >
              <div>
                <label className="mb-1 block text-xs text-slate-400">Kelly Fraction</label>
                <input
                  type="number"
                  step="0.05"
                  min={0.05}
                  max={1}
                  value={btConfig.kellyFrac}
                  onChange={(e) =>
                    setBtConfig({ ...btConfig, kellyFrac: Number(e.target.value) })
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-slate-400">Min Edge</label>
                <input
                  type="number"
                  step="0.005"
                  min={0}
                  max={0.25}
                  value={btConfig.minEdge}
                  onChange={(e) =>
                    setBtConfig({ ...btConfig, minEdge: Number(e.target.value) })
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-slate-400">Start Bankroll</label>
                <input
                  type="number"
                  step="100"
                  min={100}
                  value={btConfig.startBankroll}
                  onChange={(e) =>
                    setBtConfig({ ...btConfig, startBankroll: Number(e.target.value) })
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-slate-400">Max Stake %</label>
                <input
                  type="number"
                  step="0.01"
                  min={0.01}
                  max={0.25}
                  value={btConfig.maxBetPct}
                  onChange={(e) =>
                    setBtConfig({ ...btConfig, maxBetPct: Number(e.target.value) })
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                />
              </div>
              <button
                type="submit"
                disabled={btRunning}
                className="flex items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-60"
              >
                {btRunning && <Loader2 className="h-4 w-4 animate-spin" />}
                Run Backtest
              </button>
            </form>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Odds Matched</div>
                <div className="mt-1 font-mono text-xl font-bold text-slate-100">
                  {btReport.matchedMatches.toLocaleString()}
                </div>
                <div className="text-xs text-slate-500">
                  {pct(btReport.matchRate, 1)} of {btReport.totalTestMatches.toLocaleString()}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Bets / Win Rate</div>
                <div className="mt-1 font-mono text-xl font-bold text-slate-100">
                  {btReport.nBets.toLocaleString()}
                </div>
                <div className="text-xs text-slate-500">
                  {pct(btReport.winRate, 1)} won
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">ROI / Yield</div>
                <div
                  className={`mt-1 font-mono text-xl font-bold ${
                    btReport.roi >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {signedPct(btReport.roi, 2)}
                </div>
                <div className="text-xs text-slate-500">On total staked</div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Final Bankroll</div>
                <div className="mt-1 font-mono text-xl font-bold text-slate-100">
                  {btReport.finalBankroll.toFixed(0)}
                </div>
                <div className="text-xs text-slate-500">
                  Start {btReport.startBankroll.toFixed(0)}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Max Drawdown</div>
                <div className="mt-1 font-mono text-xl font-bold text-rose-400">
                  {pct(btReport.maxDrawdown, 1)}
                </div>
                <div className="text-xs text-slate-500">Peak-to-trough</div>
              </div>
              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
                <div className="text-xs text-slate-400">Average CLV</div>
                <div
                  className={`mt-1 font-mono text-xl font-bold ${
                    btReport.avgClv >= 0 ? 'text-emerald-400' : 'text-amber-400'
                  }`}
                >
                  {signedPct(btReport.avgClv, 2)}
                </div>
                <div className="text-xs text-slate-500">vs Pinnacle close</div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
              <h3 className="mb-3 text-sm font-semibold text-slate-200">
                Simulated Bankroll Equity Curve (B365 Offer vs Pinnacle Close)
              </h3>
              <EquityCurveChart
                curve={btReport.equityCurve}
                startBankroll={btReport.startBankroll}
              />
            </div>

            {btReport.recentBets.length > 0 && (
              <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50">
                <div className="border-b border-slate-800 px-5 py-3 text-sm font-semibold text-slate-200">
                  Most Recent Backtest Bets ({btReport.recentBets.length})
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/60 text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5">Date</th>
                        <th className="px-4 py-2.5">Backed Player</th>
                        <th className="px-4 py-2.5">Opponent</th>
                        <th className="px-4 py-2.5 text-right">Model</th>
                        <th className="px-4 py-2.5 text-right">Edge</th>
                        <th className="px-4 py-2.5 text-right">Odds</th>
                        <th className="px-4 py-2.5 text-right">CLV</th>
                        <th className="px-4 py-2.5 text-right">PnL</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70">
                      {btReport.recentBets.map((b, i) => (
                        <tr key={`${b.date}-${b.playerBacked}-${i}`} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2 font-mono text-slate-400">{b.date}</td>
                          <td className="px-4 py-2 font-medium text-slate-100">
                            {b.playerBacked}
                          </td>
                          <td className="px-4 py-2 text-slate-300">{b.opponent}</td>
                          <td className="px-4 py-2 text-right font-mono">
                            {pct(b.modelProb, 1)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-emerald-400">
                            {signedPct(b.edge, 1)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono">
                            {b.odds.toFixed(2)}
                          </td>
                          <td
                            className={`px-4 py-2 text-right font-mono ${
                              b.clv >= 0 ? 'text-emerald-400' : 'text-amber-400'
                            }`}
                          >
                            {signedPct(b.clv, 1)}
                          </td>
                          <td
                            className={`px-4 py-2 text-right font-mono font-semibold ${
                              b.won ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {b.pnl >= 0 ? `+${b.pnl.toFixed(1)}` : b.pnl.toFixed(1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'players' && (
          <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50">
            <div className="border-b border-slate-800 px-5 py-3.5">
              <h2 className="text-sm font-semibold text-slate-200">
                Top 25 ATP Players by Chronological Overall Elo
              </h2>
              <p className="text-xs text-slate-400">
                Click P1 or P2 to load a player directly into the Match Predictor.
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/60 text-slate-400">
                  <tr>
                    <th className="px-4 py-2.5">#</th>
                    <th className="px-4 py-2.5">Player</th>
                    <th className="px-4 py-2.5 text-right">Overall Elo</th>
                    <th className="px-4 py-2.5 text-right">Hard</th>
                    <th className="px-4 py-2.5 text-right">Clay</th>
                    <th className="px-4 py-2.5 text-right">Grass</th>
                    <th className="px-4 py-2.5 text-right">Matches</th>
                    <th className="px-4 py-2.5 text-right">1st Won</th>
                    <th className="px-4 py-2.5 text-right">Select</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {summary!.topPlayers.map((p, idx) => (
                    <tr key={p.name} className="hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-mono text-slate-500">{idx + 1}</td>
                      <td className="px-4 py-2.5 font-medium text-slate-100">{p.name}</td>
                      <td className="px-4 py-2.5 text-right font-mono font-semibold text-emerald-400">
                        {Math.round(p.overallElo)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-300">
                        {Math.round(p.surfaceElo.Hard)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-300">
                        {Math.round(p.surfaceElo.Clay)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-300">
                        {Math.round(p.surfaceElo.Grass)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-400">
                        {p.matchCount}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-300">
                        {pct(p.serve.first_won, 1)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <div className="inline-flex gap-1">
                          <button
                            type="button"
                            onClick={() => {
                              setPlayer1(p.name);
                              setRank1(String(p.latestRank));
                              setAge1(p.latestAge.toFixed(1));
                              setActiveTab('predict');
                            }}
                            className="rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:border-emerald-500 hover:text-emerald-300"
                          >
                            P1
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setPlayer2(p.name);
                              setRank2(String(p.latestRank));
                              setAge2(p.latestAge.toFixed(1));
                              setActiveTab('predict');
                            }}
                            className="rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:border-sky-500 hover:text-sky-300"
                          >
                            P2
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
