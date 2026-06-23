# World Cup 2026 · Prediction Net

A single-page, self-contained web app that predicts who will win the 2026 World Cup. It blends three simple models — Elo rating, recent form, and head-to-head history — and runs a Monte Carlo simulation (10,000 tournaments per refresh) to turn them into probabilities. Hosted free on GitHub Pages; predictions refresh automatically every ~2 hours via a scheduled GitHub Action as real results come in.

Four tabs: **Overview** (predicted champion, confidence, the path to the final, plain-language reasoning), **Bracket** (projected knockout tree with per-tie win probabilities), **Data** (ingestion status, every group standing, and a result feed showing how each match moved the ratings — your verification view), and **Methodology** (how it works and where it can be wrong).

---

## ⚠️ Before you create the repo — rotate the token

A GitHub personal access token was pasted in plaintext earlier in this project. **Rotate / revoke it before this repo is created or made public.** This is a hard prerequisite, not a suggestion — a leaked token in history can be scraped within minutes of a repo going public.

1. GitHub → Settings → Developer settings → Personal access tokens → revoke the exposed token.
2. Issue a new one only if you actually need it (this project does **not** require a PAT to run).
3. Make sure the old token appears nowhere in the files or git history you push.

The app itself needs only an **API-Football key**, stored as an encrypted repository secret (below) — never committed to the code.

---

## What's in here

| File | Purpose |
|------|---------|
| `index.html` | The entire app — HTML, CSS, JS, and an embedded `prediction-data` JSON block. Self-contained, no build step. |
| `predict.py` | Fetches results, builds the model, runs the simulation, and injects fresh data into `index.html`. |
| `requirements.txt` | Python dependency (`requests`). |
| `.github/workflows/predict.yml` | Scheduled + manual GitHub Action that rebuilds and deploys the page. |

The committed `index.html` ships with clearly-labelled **seed data** so the page looks right before the first live run.

---

## How predictions work (locked design)

- **Inputs:** Elo-style ratings, recent form (last ~10 matches), head-to-head history.
- **Blend:** a weighted mix — Elo 45%, form 35%, head-to-head 20% (tunable in `predict.py` → `WEIGHTS`). No single signal is allowed to dominate.
- **Simulation:** each tie's blended probability is played out across 10,000 simulated tournaments; a team's title odds = how often it wins across all runs.
- **Data:** free match-result API (API-Football). No paid odds, no betting-market data.
- **Refresh:** every ~2 hours via the Action, plus a manual trigger.

Probabilities, never asserted scorelines for unplayed games.

---

## Deploy in ~10 minutes

1. **Rotate the leaked token** (see above). Don't skip this.
2. **Create a new GitHub repo** and push these files (keep the structure, including `.github/workflows/`).
3. **Get a free API-Football key** at <https://www.api-football.com/> (or the API-Sports dashboard).
4. **Add the key as a secret:** repo → Settings → Secrets and variables → Actions → *New repository secret* → name `API_FOOTBALL_KEY`, paste the key.
   - Optional repo *variables* if your provider's IDs differ: `WC_LEAGUE_ID` (default `1`), `WC_SEASON` (default `2026`).
5. **Enable Pages:** repo → Settings → Pages → *Source: GitHub Actions*.
6. **Run it once:** repo → Actions → *Refresh World Cup 2026 prediction* → *Run workflow*. When it finishes, your site is live at `https://<you>.github.io/<repo>/`.

After that it updates itself on the 2-hourly schedule. The page's **Refresh** button reloads to pull the latest deployed build; to force a brand-new computation, hit *Run workflow* in the Actions tab.

---

## Run / test locally

No API key needed — synthesised results drive a full offline build:

```bash
pip install -r requirements.txt
python predict.py --sample            # rebuild index.html from synthesised results
python predict.py --sample --dump data.json   # also write the computed DATA to data.json
open index.html                       # or just double-click it
```

A live local run (uses your key, hits the API):

```bash
export API_FOOTBALL_KEY=your_key_here
python predict.py
```

`--sample` runs leave the page's "illustrative seed data" banner on; a live run turns it off.

---

## Tuning

- **Model weights:** `WEIGHTS` in `predict.py` (must sum to 1.0).
- **Simulation count:** `SIMS` (default 10,000).
- **Form window / K-factor / home nudge:** `FORM_WINDOW`, `K_FACTOR`, `HOME_TEAMS`.
- **Starting ratings & groups:** `BASE_ELO` and `GROUPS`.

---

## Honest limitations

The model can't see injuries, suspensions, red cards, weather, or rotated line-ups; ratings lag reality until new results land; head-to-head samples are often tiny; and knockout football is high-variance, so even a 52% favourite loses often. An ~18% champion is the *most likely* winner and still fails most of the time. The Methodology tab spells this out for anyone reading the page.

**Simplification to be aware of:** the knockout bracket is seeded from current group standings by adjusted Elo (avoiding same-group first-round meetings) rather than the exact official 2026 slotting template. It's a reasonable projection; the probabilities are the point, not the precise slot map.
