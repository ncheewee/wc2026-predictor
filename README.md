# World Cup 2026 · Prediction Net

A single-page, self-contained web app that predicts who will win the 2026 World Cup. It blends three simple models — Elo rating, recent form, and head-to-head history — and runs a Monte Carlo simulation (10,000 tournaments per refresh) to turn them into probabilities. Hosted free on GitHub Pages; predictions refresh automatically every ~2 hours via a scheduled GitHub Action as real results come in.

Five tabs: **Overview** (predicted champion, confidence, the path to the final, plain-language reasoning), **Bracket** (projected knockout tree with per-tie win probabilities), **Data** (ingestion status, every group standing, and a result feed showing how each match moved the ratings — your verification view), **Betting** (Singapore Pools 1X2 odds for upcoming matches compared against the model, with a measured value/edge read), and **Methodology** (how it works and where it can be wrong).

### Betting tab — read this

The Betting tab is a **model-vs-market comparison for information only — not betting advice.** It pulls Singapore Pools 1X2 odds (mirrored by [sgodds.com](https://sgodds.com), since Singapore Pools' own odds page is JavaScript-only and can't be scraped in CI), converts them to implied probabilities, and shows where the model's probability differs. It deliberately suppresses longshot "value" blow-ups (a simple Elo model is *not* as sharp as the market) and caps/filters edges, so a flagged "value" lean is modest by design. **Outright winner** odds can't be auto-scraped, so the market column there is a manual reference in `predict.py` (`OUTRIGHT_ODDS`) — edit it to keep it current. Betting in Singapore is 21+ and legal only via Singapore Pools; if gambling may be affecting you, call the National Problem Gambling Helpline 1800-6-668-668.

---

## ⚠️ Before you create the repo — rotate the token

A GitHub personal access token was pasted in plaintext earlier in this project. **Rotate / revoke it before this repo is created or made public.** This is a hard prerequisite, not a suggestion — a leaked token in history can be scraped within minutes of a repo going public.

1. GitHub → Settings → Developer settings → Personal access tokens → revoke the exposed token.
2. Issue a new one only if you actually need it (this project does **not** require a PAT to run).
3. Make sure the old token appears nowhere in the files or git history you push.

The app itself needs **no keys or secrets at all** — match data comes from the public-domain openfootball dataset over plain HTTPS.

---

## What's in here

| File | Purpose |
|------|---------|
| `index.html` | The entire app — HTML, CSS, JS, and an embedded `prediction-data` JSON block. Self-contained, no build step. |
| `predict.py` | Fetches results (openfootball), builds the model, runs the simulation, and injects fresh data into `index.html`. Standard library only. |
| `requirements.txt` | No third-party deps (kept so the workflow's `pip install` step is a no-op). |
| `.github/workflows/predict.yml` | Scheduled + manual GitHub Action that rebuilds and deploys the page. |

The committed `index.html` ships with clearly-labelled **seed data** so the page looks right before the first live run.

---

## How predictions work (locked design)

- **Inputs:** Elo-style ratings, recent form (last ~10 matches), head-to-head history.
- **Blend:** a weighted mix — Elo 45%, form 35%, head-to-head 20% (tunable in `predict.py` → `WEIGHTS`). No single signal is allowed to dominate.
- **Simulation:** each tie's blended probability is played out across 10,000 simulated tournaments; a team's title odds = how often it wins across all runs.
- **Data:** [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) — free, public-domain (CC0) match results and groups, **no API key required**. No paid odds, no betting-market data.
- **Refresh:** every ~2 hours via the Action, plus a manual trigger.

Probabilities, never asserted scorelines for unplayed games.

---

## Deploy in ~10 minutes

1. **Rotate the leaked token** (see above). Don't skip this.
2. **Create a new GitHub repo** and push these files (keep the structure, including `.github/workflows/`).
3. **Enable Pages:** repo → Settings → Pages → *Source: GitHub Actions*.
4. **Run it once:** repo → Actions → *Refresh World Cup 2026 prediction* → *Run workflow*. When it finishes, your site is live at `https://<you>.github.io/<repo>/`.

That's it — **no API key, no account, no secret to configure.** Data comes from the public-domain openfootball dataset. After the first run it updates itself on the 2-hourly schedule. The page's **Refresh** button reloads to pull the latest deployed build; to force a brand-new computation, hit *Run workflow* in the Actions tab.

---

## Run / test locally

No dependencies, no key — `predict.py` uses only the Python standard library.

```bash
python predict.py                     # live: fetches openfootball, rebuilds index.html
python predict.py --sample            # offline: synthesised results (no network)
python predict.py --dump data.json    # also write the computed DATA to data.json
open index.html                       # or just double-click it
```

`--sample` runs leave the page's "illustrative seed data" banner on; a live run turns it off. If openfootball is ever unreachable, the live run automatically falls back to synthesised seed data so the page still deploys.

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
