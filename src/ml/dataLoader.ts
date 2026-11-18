import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { MatchRecord, OddsRecord, SURFACES, Surface } from './types';

const require = createRequire(import.meta.url);
const XLSX = require('xlsx');

const VALID_SURFACES = new Set<string>(SURFACES);

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function parseDateMs(raw: string): { dateMs: number; year: number } | null {
  const cleaned = raw.trim().replace(/-/g, '');
  if (!/^\d{8}$/.test(cleaned)) return null;
  const year = Number(cleaned.slice(0, 4));
  const month = Number(cleaned.slice(4, 6)) - 1;
  const day = Number(cleaned.slice(6, 8));
  const ms = Date.UTC(year, month, day);
  if (Number.isNaN(ms)) return null;
  return { dateMs: ms, year };
}

function numOrNaN(raw: string | undefined): number {
  if (!raw) return Number.NaN;
  const trimmed = raw.trim();
  if (!trimmed) return Number.NaN;
  const v = Number(trimmed);
  return Number.isFinite(v) ? v : Number.NaN;
}

function numOrNull(raw: string | undefined): number | null {
  const v = numOrNaN(raw);
  return Number.isNaN(v) ? null : v;
}

function parseCsvFile(filePath: string, dedupMap: Map<string, MatchRecord>): void {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split(/\r?\n/);
  if (lines.length < 2) return;

  const headers = parseCsvLine(lines[0]).map((h) => h.trim());
  const idx = (name: string) => headers.indexOf(name);

  const iTourneyId = idx('tourney_id');
  const iTourneyName = idx('tourney_name');
  const iSurface = idx('surface');
  const iDate = idx('tourney_date');
  const iMatchNum = idx('match_num');
  const iWinner = idx('winner_name');
  const iLoser = idx('loser_name');
  const iWRank = idx('winner_rank');
  const iLRank = idx('loser_rank');
  const iWAge = idx('winner_age');
  const iLAge = idx('loser_age');
  const iBestOf = idx('best_of');
  const iWSvpt = idx('w_svpt');
  const iWAce = idx('w_ace');
  const iWDf = idx('w_df');
  const iW1stIn = idx('w_1stIn');
  const iW1stWon = idx('w_1stWon');
  const iW2ndWon = idx('w_2ndWon');
  const iLSvpt = idx('l_svpt');
  const iLAce = idx('l_ace');
  const iLDf = idx('l_df');
  const iL1stIn = idx('l_1stIn');
  const iL1stWon = idx('l_1stWon');
  const iL2ndWon = idx('l_2ndWon');

  for (let r = 1; r < lines.length; r++) {
    const line = lines[r];
    if (!line || line.trim().length === 0) continue;
    const cols = parseCsvLine(line);

    const winnerName = (cols[iWinner] ?? '').trim();
    const loserName = (cols[iLoser] ?? '').trim();
    const surface = (cols[iSurface] ?? '').trim();
    if (!winnerName || !loserName || !VALID_SURFACES.has(surface)) continue;

    const dateParsed = parseDateMs(cols[iDate] ?? '');
    if (!dateParsed) continue;

    const tourneyId = (cols[iTourneyId] ?? '').trim();
    const matchNumRaw = numOrNaN(cols[iMatchNum]);
    const matchNum = Number.isNaN(matchNumRaw) ? r : matchNumRaw;

    const wRank = numOrNaN(cols[iWRank]);
    const lRank = numOrNaN(cols[iLRank]);
    const wAge = numOrNaN(cols[iWAge]);
    const lAge = numOrNaN(cols[iLAge]);
    const bestOfRaw = numOrNaN(cols[iBestOf]);

    const record: MatchRecord = {
      tourneyId,
      tourneyName: (cols[iTourneyName] ?? '').trim(),
      surface: surface as Surface,
      dateMs: dateParsed.dateMs,
      year: dateParsed.year,
      matchNum,
      winnerName,
      loserName,
      winnerRank: Number.isNaN(wRank) || wRank <= 0 ? 500 : wRank,
      loserRank: Number.isNaN(lRank) || lRank <= 0 ? 500 : lRank,
      winnerAge: Number.isNaN(wAge) ? 26 : wAge,
      loserAge: Number.isNaN(lAge) ? 26 : lAge,
      bestOf: Number.isNaN(bestOfRaw) ? 3 : bestOfRaw,
      wSvpt: numOrNull(cols[iWSvpt]),
      wAce: numOrNull(cols[iWAce]) ?? 0,
      wDf: numOrNull(cols[iWDf]) ?? 0,
      w1stIn: numOrNull(cols[iW1stIn]) ?? 0,
      w1stWon: numOrNull(cols[iW1stWon]) ?? 0,
      w2ndWon: numOrNull(cols[iW2ndWon]) ?? 0,
      lSvpt: numOrNull(cols[iLSvpt]),
      lAce: numOrNull(cols[iLAce]) ?? 0,
      lDf: numOrNull(cols[iLDf]) ?? 0,
      l1stIn: numOrNull(cols[iL1stIn]) ?? 0,
      l1stWon: numOrNull(cols[iL1stWon]) ?? 0,
      l2ndWon: numOrNull(cols[iL2ndWon]) ?? 0,
    };

    const dedupKey = tourneyId ? `${tourneyId}::${matchNum}` : `${filePath}::${r}`;
    dedupMap.set(dedupKey, record);
  }
}

export function loadAllMatches(rootDir: string): MatchRecord[] {
  const atpDir = path.join(rootDir, 'atp_matches');
  const dedupMap = new Map<string, MatchRecord>();

  if (fs.existsSync(atpDir)) {
    const files = fs
      .readdirSync(atpDir)
      .filter((f) => /^atp_matches_[12]\d{3}\.csv$/.test(f))
      .sort();
    for (const file of files) {
      parseCsvFile(path.join(atpDir, file), dedupMap);
    }
  }

  const seasonal2026 = path.join(rootDir, '2026.csv');
  if (fs.existsSync(seasonal2026)) {
    parseCsvFile(seasonal2026, dedupMap);
  }

  const matches = Array.from(dedupMap.values());
  matches.sort((a, b) => a.dateMs - b.dateMs || a.matchNum - b.matchNum);
  return matches;
}

function parseExcelDateYear(val: unknown): { year: number; dateStr: string } | null {
  if (val instanceof Date && !Number.isNaN(val.getTime())) {
    return {
      year: val.getUTCFullYear(),
      dateStr: val.toISOString().slice(0, 10),
    };
  }
  if (typeof val === 'number' && Number.isFinite(val)) {
    // Excel serial date (days since 1899-12-30)
    const utcMs = Math.round((val - 25569) * 86400 * 1000);
    const d = new Date(utcMs);
    if (!Number.isNaN(d.getTime())) {
      return { year: d.getUTCFullYear(), dateStr: d.toISOString().slice(0, 10) };
    }
  }
  if (typeof val === 'string') {
    const d = new Date(val);
    if (!Number.isNaN(d.getTime())) {
      return { year: d.getUTCFullYear(), dateStr: d.toISOString().slice(0, 10) };
    }
  }
  return null;
}

export function loadOddsForYears(rootDir: string, minYear: number, maxYear: number): OddsRecord[] {
  const oddsDir = path.join(rootDir, 'odds');
  if (!fs.existsSync(oddsDir)) return [];

  const files = fs
    .readdirSync(oddsDir)
    .filter((f) => /^tennis_odds_\d{4}\.xlsx$/.test(f))
    .sort();

  const dedup = new Map<string, OddsRecord>();

  for (const file of files) {
    const match = file.match(/tennis_odds_(\d{4})\.xlsx/);
    if (!match) continue;
    const fileYear = Number(match[1]);
    if (fileYear < minYear - 1 || fileYear > maxYear + 1) continue;

    const wb = XLSX.readFile(path.join(oddsDir, file), { cellDates: true });
    const firstSheetName = wb.SheetNames[0];
    if (!firstSheetName) continue;
    const rows = XLSX.utils.sheet_to_json(wb.Sheets[firstSheetName], {
      defval: null,
    }) as Record<string, unknown>[];

    for (const row of rows) {
      const winner = String(row.Winner ?? '').trim().toLowerCase();
      const loser = String(row.Loser ?? '').trim().toLowerCase();
      if (!winner || !loser) continue;

      const parsedDate = parseExcelDateYear(row.Date);
      if (!parsedDate) continue;

      const b365w = Number(row.B365W);
      const b365l = Number(row.B365L);
      const psw = Number(row.PSW);
      const psl = Number(row.PSL);
      if (
        !Number.isFinite(b365w) ||
        !Number.isFinite(b365l) ||
        !Number.isFinite(psw) ||
        !Number.isFinite(psl) ||
        b365w <= 1.01 ||
        b365l <= 1.01 ||
        psw <= 1.01 ||
        psl <= 1.01
      ) {
        continue;
      }

      const key = `${winner}::${loser}::${parsedDate.year}`;
      if (!dedup.has(key)) {
        dedup.set(key, {
          wk: winner,
          lk: loser,
          year: parsedDate.year,
          b365w,
          b365l,
          psw,
          psl,
          tournament: String(row.Tournament ?? ''),
          dateStr: parsedDate.dateStr,
        });
      }
    }
  }

  return Array.from(dedup.values());
}
