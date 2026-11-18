import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import { tennisPipeline } from './src/ml/pipeline';
import { SURFACES, Surface } from './src/ml/types';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const isProd = process.env.NODE_ENV === 'production';

async function bootstrap() {
  const app = express();
  app.use(express.json());

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, pipeline: tennisPipeline.getStatus() });
  });

  app.get('/api/status', (_req, res) => {
    res.json(tennisPipeline.getStatus());
  });

  app.get('/api/summary', async (_req, res) => {
    try {
      const summary = await tennisPipeline.getSummary();
      res.json(summary);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to load pipeline summary',
      });
    }
  });

  app.get('/api/players', async (req, res) => {
    try {
      const q = String(req.query.q ?? '');
      const limit = Math.min(100, Math.max(1, Number(req.query.limit ?? 20)));
      const players = await tennisPipeline.searchPlayers(q, limit);
      res.json({ players });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to search players',
      });
    }
  });

  app.post('/api/predict', async (req, res) => {
    try {
      const body = req.body ?? {};
      const player1 = String(body.player1 ?? '').trim();
      const player2 = String(body.player2 ?? '').trim();
      const surface = String(body.surface ?? 'Hard') as Surface;
      const bestOf = Number(body.bestOf) === 5 ? 5 : 3;

      if (!player1 || !player2) {
        return res.status(400).json({ error: 'Both player1 and player2 are required.' });
      }
      if (!SURFACES.includes(surface)) {
        return res.status(400).json({ error: `Surface must be one of: ${SURFACES.join(', ')}` });
      }

      const result = await tennisPipeline.predictMatch({
        player1,
        player2,
        surface,
        bestOf,
        rank1: body.rank1 ? Number(body.rank1) : undefined,
        rank2: body.rank2 ? Number(body.rank2) : undefined,
        age1: body.age1 ? Number(body.age1) : undefined,
        age2: body.age2 ? Number(body.age2) : undefined,
        oddsP1: body.oddsP1 ? Number(body.oddsP1) : null,
        oddsP2: body.oddsP2 ? Number(body.oddsP2) : null,
      });

      res.json(result);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Prediction failed',
      });
    }
  });

  app.post('/api/backtest', async (req, res) => {
    try {
      const body = req.body ?? {};
      const kellyFrac = Math.min(1, Math.max(0.05, Number(body.kellyFrac ?? 0.25)));
      const minEdge = Math.min(0.25, Math.max(0, Number(body.minEdge ?? 0.03)));
      const startBankroll = Math.max(100, Number(body.startBankroll ?? 1000));
      const maxBetPct = Math.min(0.25, Math.max(0.005, Number(body.maxBetPct ?? 0.05)));

      const report = await tennisPipeline.runCustomBacktest({
        kellyFrac,
        minEdge,
        startBankroll,
        maxBetPct,
      });
      res.json(report);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Backtest failed',
      });
    }
  });

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] Listening on http://0.0.0.0:${PORT}`);
    tennisPipeline.start(__dirname);
  });
}

bootstrap();
