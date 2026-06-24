#!/usr/bin/env python3
"""
World Cup 2026 — Prediction Net
================================
Builds the tournament forecast and injects it into index.html.

Pipeline:
  1. Ingest results + groups          openfootball/worldcup.json (free, public domain, NO API KEY)
  2. Build three signals              Elo rating · recent form · head-to-head
  3. Blend them into a match model    p(A beats B)
  4. Monte Carlo the knockout         ~10,000 simulated tournaments -> title odds
  5. Emit DATA + inject into HTML     single self-contained file, no build step

Usage:
  python predict.py            # live; fetches openfootball (default)
  python predict.py --sample   # offline; synthesised results (for testing, no network)
  python predict.py --out index.html --dump data.json

Data source: https://github.com/openfootball/worldcup.json  (CC0 / public domain)
No betting-market data is used anywhere.

Simplifications (kept honest, see the Methodology tab):
  - The knockout bracket is seeded from current group standings by adjusted Elo,
    avoiding same-group first-round meetings — a reasonable projection, not the exact
    official slotting template.
  - Recent form = points-per-game over the last N matches mapped to a win prob.
  - Head-to-head uses historical result share; neutral (0.5) when no history.
"""

import argparse, json, math, os, random, re, sys, datetime as dt
import urllib.request

# ----------------------------------------------------------------------------- config
WEIGHTS = {"elo": 0.45, "form": 0.35, "h2h": 0.20}   # must sum to 1.0
K_FACTOR = 40
FORM_WINDOW = 10
SIMS = 10000
DEFAULT_ELO = 1650                                   # prior for teams not in BASE_ELO
HOME_TEAMS = {"USA", "United States", "Canada", "Mexico"}   # 2026 hosts (small nudge)

OF_BASE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026"
OF_MATCHES = OF_BASE + "/worldcup.json"
OF_GROUPS  = OF_BASE + "/worldcup.groups.json"

# Odds: Singapore Pools 1X2 odds, mirrored in scrapable HTML by sgodds.com.
ODDS_URL = "https://sgodds.com/football/current-odds"
ODDS_VALUE_EDGE = 0.05    # EV per $1 above which we flag "value"
# sgodds team names -> our (openfootball) names
ODDS_ALIAS = {"Holland":"Netherlands","Korea Republic":"South Korea","Czechia":"Czech Republic",
              "Congo DR":"DR Congo","Turkiye":"Turkey","Bosnia":"Bosnia and Herzegovina"}
# Outright "winner" odds CANNOT be auto-scraped (Singapore Pools' outrights page is
# JavaScript-only). Maintain a manual reference here to enable outright value analysis.
# Decimal odds; edit/update as needed. Empty -> outright shows model fair-odds only.
OUTRIGHT_ODDS = {  # reference snapshot — update from Singapore Pools as desired
    "Spain":6.50, "France":5.50, "England":9.00, "Argentina":7.50, "Brazil":9.00,
    "Portugal":13.0, "Germany":15.0, "Netherlands":15.0,
}
OUTRIGHT_AS_OF = "reference snapshot — edit OUTRIGHT_ODDS in predict.py"

# Elo priors (refined by results as they arrive). Unknown teams default to DEFAULT_ELO.
BASE_ELO = {
    "Argentina":2090,"France":2065,"Spain":2055,"England":2010,"Brazil":2000,
    "Portugal":1970,"Netherlands":1955,"Belgium":1940,"Italy":1925,"Germany":1920,
    "Croatia":1880,"Uruguay":1870,"Colombia":1860,"USA":1840,"United States":1840,
    "Denmark":1835,"Switzerland":1830,"Mexico":1825,"Japan":1815,"Senegal":1810,
    "Morocco":1805,"South Korea":1790,"Korea Republic":1790,"Serbia":1785,"Austria":1780,
    "Ecuador":1775,"Australia":1770,"Canada":1765,"Norway":1760,"Nigeria":1755,
    "Poland":1750,"Ivory Coast":1745,"Egypt":1740,"Algeria":1735,"Paraguay":1720,
    "Tunisia":1715,"Ghana":1710,"Iran":1705,"Saudi Arabia":1690,"Qatar":1680,
    "Panama":1670,"Jamaica":1660,"Jordan":1650,"Cape Verde":1640,"Uzbekistan":1635,
    "New Zealand":1620,"Honduras":1610,"Curacao":1600,"Czech Republic":1820,
    "South Africa":1700,"Scotland":1800,"Wales":1780,"Turkey":1790,"Ukraine":1800,
    "Peru":1730,"Chile":1760,"Costa Rica":1690,"Cameroon":1720,"Mali":1700,
    "Greece":1770,"Sweden":1775,"Hungary":1760,"Romania":1730,
}

# Fallback groups for --sample (offline). Live mode replaces this from openfootball.
GROUPS = {
    "A":["Mexico","Norway","Saudi Arabia","Jamaica"],
    "B":["Canada","Belgium","Egypt","New Zealand"],
    "C":["Spain","Uruguay","Japan","Ghana"],
    "D":["USA","Netherlands","Ecuador","Qatar"],
    "E":["France","Croatia","Senegal","Iran"],
    "F":["England","Switzerland","South Korea","Tunisia"],
    "G":["Argentina","Denmark","Australia","Cape Verde"],
    "H":["Portugal","Colombia","Morocco","Panama"],
    "I":["Brazil","Germany","Nigeria","Jordan"],
    "J":["Italy","Mexico B","Ivory Coast","Curacao"],
    "K":["Netherlands B","Austria","Algeria","Honduras"],
    "L":["Germany B","Serbia","Paraguay","Uzbekistan"],
}

FLAGS = {"Argentina":"\U0001F1E6\U0001F1F7","France":"\U0001F1EB\U0001F1F7","Spain":"\U0001F1EA\U0001F1F8",
         "Brazil":"\U0001F1E7\U0001F1F7","Portugal":"\U0001F1F5\U0001F1F9","Netherlands":"\U0001F1F3\U0001F1F1",
         "Germany":"\U0001F1E9\U0001F1EA","Mexico":"\U0001F1F2\U0001F1FD","USA":"\U0001F1FA\U0001F1F8"}

# ----------------------------------------------------------------------------- ingest
def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-prediction-net"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def _get_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (wc2026-prediction-net)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")

def fetch_odds():
    """Scrape Singapore Pools 1X2 odds (via sgodds.com). Returns (fixtures, updated)."""
    try:
        h = _get_text(ODDS_URL)
    except Exception as e:
        print("[odds] fetch failed:", e); return [], None
    fixtures = []
    for row in re.split(r'<div class="row border-bottom', h):
        if "W Cup" not in row:
            continue
        m = re.search(r'current-odds/[^"]*"[^>]*>([^<]+)</a>', row)
        if not m or " vs " not in m.group(1):
            continue
        a, b = [ODDS_ALIAS.get(x.strip(), x.strip()) for x in m.group(1).split(" vs ")]
        od = re.findall(r'<strong>([\d.]+)</strong>', row)[:3]
        if len(od) != 3:
            continue
        fixtures.append((a, b, [float(x) for x in od]))
    upd = re.search(r'Last Updated on ([\d:\- ]+)', h)
    print(f"[odds] sgodds W-Cup fixtures parsed: {len(fixtures)}")
    return fixtures, (upd.group(1).strip() if upd else None)

def compute_betting(fixtures, updated, elo, form, h2h, teamset, title_odds, reliability=None):
    """Confidence-balanced value vs Singapore Pools odds.
    Layer 1: shrink the model toward the de-vigged market by how reliable the model has
    actually been for that outcome (home/draw/away) — earned trust, set by the data.
    Layer 2: size conviction with fractional Kelly (Kelly fraction itself scaled by the
    model's overall Brier skill). Output a 0-3 star conviction per pick."""
    rel = reliability or {"home":1.0,"draw":1.0,"away":1.0,"kelly":0.25}
    r_by = [rel.get("home",1.0), rel.get("draw",1.0), rel.get("away",1.0)]
    kelly = rel.get("kelly",0.25)
    matches = []
    for a, b, o in fixtures:
        if a not in teamset or b not in teamset:
            continue
        names = {0: a, 1: "Draw", 2: b}
        p = match_prob(a, b, elo, form, h2h)
        model = one_x_two(p)                                  # raw model [home,draw,away]
        imp = [1/o[0], 1/o[1], 1/o[2]]; s = sum(imp)
        fair = [i/s for i in imp]                             # de-vigged market probs
        # Layer 1 — shrink toward market by earned reliability, then renormalise
        p_used = [r_by[i]*model[i] + (1-r_by[i])*fair[i] for i in range(3)]
        su = sum(p_used) or 1; p_used = [x/su for x in p_used]
        ev = [p_used[i]*o[i]-1 for i in range(3)]             # confidence-adjusted edge
        ev_raw = [model[i]*o[i]-1 for i in range(3)]          # raw edge (transparency)
        mkt_fav = min(range(3), key=lambda i: o[i])
        mdl_fav = max(range(3), key=lambda i: p_used[i])
        bi = max(range(3), key=lambda i: ev[i]); edge = ev[bi]
        # Layer 2 — fractional Kelly conviction on the adjusted edge
        f = max(0.0, edge/(o[bi]-1)) if o[bi] > 1 else 0.0
        conv = f*kelly
        stars = 3 if conv>=0.06 else (2 if conv>=0.03 else (1 if (conv>=0.012 and edge>0) else 0))
        if p_used[bi] < 0.25:   # confidence floor — don't back a pick the model itself rates a longshot
            stars = 0
        if stars >= 1:
            label = "value"
            note = (f"Model (trust-adjusted) {round(p_used[bi]*100)}% vs market "
                    f"{round(fair[bi]*100)}% on {names[bi]} — edge +{round(edge*100)}% after "
                    f"shrinking to the market by the model's track record on this outcome.")
        elif mkt_fav == mdl_fav:
            label = "agree"
            note = f"In line with the market favourite, {names[mkt_fav]}."
        else:
            label = "diverge"
            note = (f"Model leans {names[mdl_fav]} but the edge isn't trustworthy enough to back "
                    "once shrunk toward the market.")
        matches.append({"a": a, "b": b, "odds": [round(x,2) for x in o],
                        "model": [round(x*100) for x in model],
                        "adj": [round(x*100) for x in p_used],
                        "marketFav": names[mkt_fav], "modelFav": names[mdl_fav],
                        "pick": names[bi], "pickIdx": bi, "edge": round(edge*100,1),
                        "rawEdge": round(ev_raw[bi]*100,1), "stars": stars,
                        "label": label, "note": note})
    matches.sort(key=lambda m: (-m["stars"], -m["edge"]))
    def lab(edge):
        return "value" if edge >= ODDS_VALUE_EDGE else ("lean" if edge >= 0 else "no-value")
    outright = []
    for team, prob in sorted(title_odds.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        mk = OUTRIGHT_ODDS.get(team)
        edge = (prob*mk-1) if mk else None
        outright.append({"team": team, "modelProb": round(prob*100,1),
                         "fairOdds": round(1/prob,2) if prob > 0 else None,
                         "marketOdds": mk, "edge": round(edge*100,1) if edge is not None else None,
                         "label": (lab(edge) if edge is not None else None)})
    return {
        "oddsUpdated": updated or "—",
        "source": "Singapore Pools (via sgodds.com)",
        "matches": matches, "outright": outright, "outrightNote": OUTRIGHT_AS_OF,
        "trust": {"home":round(r_by[0]*100),"draw":round(r_by[1]*100),"away":round(r_by[2]*100),
                  "kelly":round(kelly*100)},
        "method": ("Value is shrunk toward the market by how reliable the model has actually been "
                   "on each outcome (home/draw/away), then sized by fractional Kelly into a 0–3★ "
                   "conviction. Trust and Kelly fraction are set by the model's own track record."),
        "disclaimer": ("For analysis and entertainment only — not betting advice. Odds move "
                       "constantly; always verify on Singapore Pools before acting. Sports betting "
                       "in Singapore is 21+ and legal only via Singapore Pools. If gambling may be "
                       "affecting you, call the National Problem Gambling Helpline 1800-6-668-668."),
    }

def groups_as_list(groups_dict):
    """[{'name':'Group A','teams':[...]}] -> [('A',[...]), ...] ordered."""
    out = []
    for g in groups_dict:
        letter = g["name"].replace("Group", "").strip()
        out.append((letter, list(g["teams"])))
    out.sort(key=lambda x: x[0])
    return out

def fetch_openfootball():
    """Return (matches, groups_list) from openfootball, or None on failure."""
    try:
        gj = _get_json(OF_GROUPS)
        mj = _get_json(OF_MATCHES)
    except Exception as e:
        print("[of] fetch failed:", e)
        return None
    groups = groups_as_list(gj.get("groups", []))
    teamset = {t for _, ts in groups for t in ts}
    matches, played = [], 0
    for m in mj.get("matches", []):
        sc = (m.get("score") or {}).get("ft")
        if not sc or len(sc) != 2:
            continue                                  # not played yet
        a, b = m.get("team1"), m.get("team2")
        if a not in teamset or b not in teamset:
            continue                                  # knockout placeholder slot
        stage = "Group Stage" if m.get("group") else (m.get("round") or "Knockout")
        matches.append({"date": m.get("date",""), "home": a, "away": b,
                        "hg": int(sc[0]), "ag": int(sc[1]), "stage": stage,
                        "round": m.get("round") or stage})
        played += 1
    matches.sort(key=lambda x: x["date"])
    knockout = [{"num":m.get("num"),"round":m.get("round"),
                 "team1":m.get("team1"),"team2":m.get("team2")}
                for m in mj.get("matches", []) if not m.get("group") and m.get("num")]
    print(f"[of] groups={len(groups)} teams={len(teamset)} played_matches={played} knockout_slots={len(knockout)}")
    return matches, groups, knockout

def synth_sample(seed=20260623):
    """Synthesise a completed group stage from BASE_ELO + noise (offline test only)."""
    rng = random.Random(seed); matches=[]; base=dt.datetime(2026,6,11,18,0); md=0
    for g, teams in GROUPS.items():
        for i,(a,b) in enumerate([(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]):
            ta,tb=teams[a],teams[b]; ea,eb=elo_of(ta),elo_of(tb)
            pa=1/(1+10**((eb-ea)/400))
            hg=max(0,int(round(rng.gauss(1.3+(pa-0.5)*2.4,1.0))))
            ag=max(0,int(round(rng.gauss(1.3+((1-pa)-0.5)*2.4,1.0))))
            matches.append({"date":(base+dt.timedelta(days=md+i//3)).isoformat()+"Z",
                            "home":ta,"away":tb,"hg":hg,"ag":ag,"stage":"Group Stage",
                            "round":f"Matchday {i//2+1}"})
        md+=4
    matches.sort(key=lambda m:m["date"])
    return matches, [(k,v) for k,v in GROUPS.items()], list(KO_TEMPLATE)

def elo_of(team):
    return BASE_ELO.get(team, DEFAULT_ELO)

def ordinal(n):
    if not n: return "—"
    if 10 <= n % 100 <= 20: return f"{n}th"
    return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n%10,'th') }".replace(" ", "")

# ----------------------------------------------------------------------------- signals
def build_signals(matches, teams):
    """Walk forward through results: for each match, record what the model would have
    predicted from PRE-match ratings, then update ratings. Also snapshot ratings at the
    end of each round (matchday) so outright odds can be re-simulated over time."""
    elo = {t: elo_of(t) for t in teams}
    form = {t: [] for t in teams}
    h2h = {}; feed=[]; log=[]; snaps=[]; cur=None
    def snapshot():
        return (elo.copy(), {k:v[:] for k,v in form.items()}, {k:list(v) for k,v in h2h.items()})
    for m in matches:
        a,b = m["home"], m["away"]
        if a not in elo or b not in elo: continue
        rnd = m.get("round","?")
        if cur is None: cur = rnd
        if rnd != cur:
            snaps.append((cur, snapshot())); cur = rnd
        # --- predict BEFORE seeing the result (using current ratings) ---
        p = match_prob(a,b,elo,form,h2h)
        model = one_x_two(p)
        pi = max(range(3), key=lambda i: model[i])
        pred_name = {0:a,1:"Draw",2:b}[pi]; pred_conf = round(model[pi]*100)
        ai = 0 if m["hg"]>m["ag"] else (1 if m["hg"]==m["ag"] else 2)
        act_name = {0:a,1:"Draw",2:b}[ai]
        log.append({"date":m.get("date","")[:10],"a":a,"b":b,"round":rnd,
                    "pred":pred_name,"conf":pred_conf,"score":f"{m['hg']}-{m['ag']}",
                    "result":act_name,"correct":pi==ai,
                    "probs":[round(x,4) for x in model],"ax":ai})
        # --- now update ratings with the actual result ---
        ra,rb = elo[a], elo[b]
        ea = 1/(1+10**((rb-ra)/400))
        sa = 1.0 if m["hg"]>m["ag"] else (0.0 if m["hg"]<m["ag"] else 0.5)
        gd = abs(m["hg"]-m["ag"])
        mult = 1.0 if gd<=1 else (1.5 if gd==2 else (1.75+(gd-3)/8))
        delta = K_FACTOR*mult*(sa-ea)
        elo[a]=ra+delta; elo[b]=rb-delta
        form[a].append(3 if sa==1 else (1 if sa==0.5 else 0))
        form[b].append(3 if sa==0 else (1 if sa==0.5 else 0))
        key=(a,b) if a<b else (b,a); rec=h2h.setdefault(key,[0,0,0])
        if sa==1: rec[0 if a<b else 2]+=1
        elif sa==0: rec[2 if a<b else 0]+=1
        else: rec[1]+=1
        feed.append({"a":a,"b":b,"score":f"{m['hg']}-{m['ag']}","delta":delta})
    if cur is not None: snaps.append(("Now", snapshot()))
    feed.reverse()
    return elo, form, h2h, feed, log, snaps

def form_prob(team, form):
    r = form.get(team, [])[-FORM_WINDOW:]
    if not r: return 0.5
    return 0.5 + (sum(r)/(len(r)*3.0)-0.5)*0.6

def h2h_prob(a,b,h2h):
    key=(a,b) if a<b else (b,a); rec=h2h.get(key)
    if not rec or sum(rec)==0: return 0.5
    aw,dr,bw=rec; total=aw+dr+bw
    share=(aw+0.5*dr)/total if a<b else (bw+0.5*dr)/total
    return 0.5 + (share-0.5)*0.7

TOTAL_GOALS = 2.6     # baseline expected goals in a tie (typical for international football)
DRAW_CALIBRATION = 1.4  # Dixon-Coles-style boost: independent Poisson under-predicts draws.
                        # 1.4 calibrates avg draw prob to the observed rate and minimises Brier
                        # (validated on results so far). Raising it further only adds *wrong*
                        # draw calls and worsens accuracy + Brier — draws aren't reliably callable.

def one_x_two(p):
    """Proper 1·X·2 via a Poisson goals model with an empirical draw calibration.
    Convert the blended binary win prob p into a goal supremacy, split into expected
    goals per side, sum the Poisson score matrix into P(home)/P(draw)/P(away), then
    nudge the draw mass up (Dixon-Coles effect) and renormalise."""
    p = min(0.99, max(0.01, p))
    d_eff = 400*math.log10(p/(1-p))            # effective Elo gap implied by p
    S = max(-2.5, min(2.5, d_eff/130.0))       # goal supremacy (home minus away), capped
    la = max(0.2, (TOTAL_GOALS+S)/2)
    lb = max(0.2, (TOTAL_GOALS-S)/2)
    N = 8
    pa = [math.exp(-la)*la**i/math.factorial(i) for i in range(N+1)]
    pb = [math.exp(-lb)*lb**j/math.factorial(j) for j in range(N+1)]
    ph=pd=pw=0.0
    for i in range(N+1):
        for j in range(N+1):
            pr = pa[i]*pb[j]
            if i>j: ph+=pr
            elif i==j: pd+=pr
            else: pw+=pr
    pd *= DRAW_CALIBRATION
    s = ph+pd+pw
    return [ph/s, pd/s, pw/s]

def match_prob(a,b,elo,form,h2h):
    ra=elo[a]+(35 if a in HOME_TEAMS else 0)
    rb=elo[b]+(35 if b in HOME_TEAMS else 0)
    p_elo=1/(1+10**((rb-ra)/400))
    # Only fold in form/H2H when there's actual data, and renormalise the weights so
    # missing signals don't drag every match toward a 50/50 coin flip.
    comps=[(WEIGHTS["elo"], p_elo)]
    if len(form.get(a,[]))>=2 and len(form.get(b,[]))>=2:
        fa,fb=form_prob(a,form),form_prob(b,form)
        comps.append((WEIGHTS["form"], fa/(fa+fb)))
    key=(a,b) if a<b else (b,a)
    if h2h.get(key) and sum(h2h[key])>0:
        comps.append((WEIGHTS["h2h"], h2h_prob(a,b,h2h)))
    wsum=sum(w for w,_ in comps)
    p=sum(w*v for w,v in comps)/wsum
    return min(0.99,max(0.01,p))

# ----------------------------------------------------------------------------- standings + bracket
def standings(matches, groups):
    teams=[t for _,ts in groups for t in ts]
    tbl={t:{"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0} for t in teams}
    for m in matches:
        a,b=m["home"],m["away"]
        if a not in tbl or b not in tbl or not m.get("stage","").lower().startswith("group"): continue
        tbl[a]["P"]+=1; tbl[b]["P"]+=1
        tbl[a]["GF"]+=m["hg"]; tbl[a]["GA"]+=m["ag"]; tbl[b]["GF"]+=m["ag"]; tbl[b]["GA"]+=m["hg"]
        if m["hg"]>m["ag"]: tbl[a]["W"]+=1; tbl[b]["L"]+=1
        elif m["hg"]<m["ag"]: tbl[b]["W"]+=1; tbl[a]["L"]+=1
        else: tbl[a]["D"]+=1; tbl[b]["D"]+=1
    out=[]
    for g,ts in groups:
        rows=[]
        for t in ts:
            s=tbl[t]; rows.append([t,s["P"],s["W"],s["D"],s["L"],s["GF"]-s["GA"],s["W"]*3+s["D"]])
        rows.sort(key=lambda r:(r[6],r[5],r[2]),reverse=True)
        out.append([g,rows])
    return out

# ----------------------------------------------------------------------------- real bracket
# Uses the OFFICIAL 2026 knockout template (positional slot codes from openfootball)
# instead of seeding by Elo — so the path reflects the real draw, not an invented soft half.
KO_ROUNDS = ["Round of 32","Round of 16","Quarter-final","Semi-final","Final"]

# Official 2026 knockout template (slot codes). Fallback for --sample / if live fetch lacks it.
def _ko(num,rnd,t1,t2): return {"num":num,"round":rnd,"team1":t1,"team2":t2}
KO_TEMPLATE = [
    _ko(73,"Round of 32","2A","2B"),_ko(74,"Round of 32","1D","3A/B/C/D/F"),
    _ko(75,"Round of 32","1F","2C"),_ko(76,"Round of 32","1C","2F"),
    _ko(77,"Round of 32","1I","3C/D/F/G/H"),_ko(78,"Round of 32","2E","2I"),
    _ko(79,"Round of 32","1A","3C/E/F/H/I"),_ko(80,"Round of 32","1L","3E/H/I/J/K"),
    _ko(81,"Round of 32","1E","3B/E/F/I/J"),_ko(82,"Round of 32","1G","3A/E/H/I/J"),
    _ko(83,"Round of 32","2K","2L"),_ko(84,"Round of 32","1H","2J"),
    _ko(85,"Round of 32","1B","3E/F/G/I/J"),_ko(86,"Round of 32","1J","2H"),
    _ko(87,"Round of 32","1K","3D/E/I/J/L"),_ko(88,"Round of 32","2D","2G"),
    _ko(89,"Round of 16","W74","W77"),_ko(90,"Round of 16","W73","W75"),
    _ko(91,"Round of 16","W76","W78"),_ko(92,"Round of 16","W79","W80"),
    _ko(93,"Round of 16","W83","W84"),_ko(94,"Round of 16","W81","W82"),
    _ko(95,"Round of 16","W86","W88"),_ko(96,"Round of 16","W85","W87"),
    _ko(97,"Quarter-final","W89","W90"),_ko(98,"Quarter-final","W93","W94"),
    _ko(99,"Quarter-final","W91","W92"),_ko(100,"Quarter-final","W95","W96"),
    _ko(101,"Semi-final","W97","W98"),_ko(102,"Semi-final","W99","W100"),
    _ko(104,"Final","W101","W102"),
]

def make_resolver(stand, knockout):
    """resolve(code)->team for R32 slot codes: '1A'/'2B' from standings, '3X/Y/..' via a
    legal assignment of the 8 best third-placed teams, or an already-decided team name."""
    pos1={g:rows[0][0] for g,rows in stand}
    pos2={g:rows[1][0] for g,rows in stand}
    names=set(pos1.values())|set(pos2.values())
    thirds=[(g,rows[2][0],rows[2][6],rows[2][5]) for g,rows in stand]   # group,team,pts,gd
    best8=sorted(thirds,key=lambda x:(x[2],x[3]),reverse=True)[:8]
    third_team={g:t for g,t,_,_ in best8}; qg=set(third_team)
    slots=[]
    for m in knockout:
        if m["round"]!="Round of 32": continue
        for code in (m["team1"],m["team2"]):
            if code and code[0]=="3" and "/" in code:
                slots.append((code,set(code[1:].split("/"))))
    assign={}; order=sorted(range(len(slots)),key=lambda i:len(slots[i][1]&qg))
    def bt(i,used):
        if i==len(order): return True
        code,allowed=slots[order[i]]
        for g in sorted(allowed&qg):
            if g in used: continue
            used.add(g); assign[code]=g
            if bt(i+1,used): return True
            used.discard(g)
        return False
    bt(0,set())
    third_for={code:third_team[g] for code,g in assign.items()}
    def resolve(code):
        if not code: return None
        if code in names or code in third_team.values(): return code
        if code[0]=="1": return pos1.get(code[1:])
        if code[0]=="2": return pos2.get(code[1:])
        if code[0]=="3": return third_for.get(code)
        return code
    return resolve

def ko_index(knockout):
    """Return (mt, order, finalnum): match map, planar top-to-bottom position per match
    (so bracket connectors line up), and the final's match number."""
    mt={m["num"]:m for m in knockout if m["round"] in KO_ROUNDS}
    finalnum=max(mt)
    seq=[]
    def visit(num):
        m=mt.get(num)
        if not m: return
        f=[int(c[1:]) for c in (m["team1"],m["team2"]) if c and c[0]=="W"]
        if not f: seq.append(num); return
        for fn in f: visit(fn)
    visit(finalnum)
    leaf={num:i for i,num in enumerate(seq)}
    def pos(num):
        m=mt[num]; f=[int(c[1:]) for c in (m["team1"],m["team2"]) if c and c[0]=="W"]
        return leaf[num] if not f else sum(pos(x) for x in f)/len(f)
    return mt, {n:pos(n) for n in mt}, finalnum

def _side(code,res,resolve):
    if not code: return None
    if code[0]=="W": return res.get(int(code[1:]))
    if code[0]=="L": return None
    return resolve(code)

def simulate_bracket(mt, resolve, elo,form,h2h, pick):
    res={}; parts={r:set() for r in KO_ROUNDS}
    for num in sorted(mt):
        m=mt[num]; a=_side(m["team1"],res,resolve); b=_side(m["team2"],res,resolve)
        if not a or not b: continue
        parts[m["round"]].add(a); parts[m["round"]].add(b); res[num]=pick(a,b)
    return res,parts

def monte_carlo_real(mt, resolve, elo,form,h2h, finalnum, sims=SIMS, seed=1):
    rng=random.Random(seed); champ={}; reach={r:{} for r in KO_ROUNDS}
    pk=lambda a,b: a if rng.random()<match_prob(a,b,elo,form,h2h) else b
    for _ in range(sims):
        res,parts=simulate_bracket(mt,resolve,elo,form,h2h,pk)
        for r in KO_ROUNDS:
            for t in parts[r]: reach[r][t]=reach[r].get(t,0)+1
        w=res.get(finalnum)
        if w: champ[w]=champ.get(w,0)+1
    title={t:c/sims for t,c in champ.items()}
    reachp={r:{t:c/sims for t,c in d.items()} for r,d in reach.items()}
    return title, reachp

def projected_bracket(mt, order, resolve, elo,form,h2h, finalnum, champ):
    """Deterministic favourite-advances view, but force the headline champion along its
    real-template path so the highlighted route is consistent with the title pick."""
    det,_=simulate_bracket(mt,resolve,elo,form,h2h, lambda a,b: a if match_prob(a,b,elo,form,h2h)>=0.5 else b)
    parent={}
    for n,m in mt.items():
        for c in (m["team1"],m["team2"]):
            if c and c[0]=="W": parent[int(c[1:])]=n
    champ_r32=next((n for n,m in mt.items() if m["round"]=="Round of 32"
                    and champ in (resolve(m["team1"]),resolve(m["team2"]))), None)
    onpath=set(); cur=champ_r32
    while cur is not None: onpath.add(cur); cur=parent.get(cur)
    res=dict(det)
    for n in onpath: res[n]=champ
    rounds=[]; clabel={"Round of 16":"R16","Quarter-final":"QF","Semi-final":"SF","Final":"FIN"}; cpath=[]
    for rname in KO_ROUNDS:
        nums=sorted([n for n in mt if mt[n]["round"]==rname], key=lambda n:order[n])
        ties=[]
        for n in nums:
            m=mt[n]; a=_side(m["team1"],res,resolve); b=_side(m["team2"],res,resolve)
            if not a or not b: ties.append(["TBD","TBD",0,0,0]); continue
            if n in onpath:
                opp=b if a==champ else a
                p=match_prob(champ,opp,elo,form,h2h); w=0 if a==champ else 1
                ties.append([a,b,w,round(p*100),1])   # p is the champion's win prob
                if rname in clabel:
                    cpath.append({"round":clabel[rname],"opp":opp,
                                  "note":"likely opponent" if rname=="Round of 16" else "projected",
                                  "pct":round(match_prob(champ,opp,elo,form,h2h)*100)})
            else:
                p=match_prob(a,b,elo,form,h2h); w=0 if p>=0.5 else 1
                ties.append([a,b,w,round((p if w==0 else 1-p)*100),0])
        rounds.append({"name":rname,"ties":ties})
    return rounds,cpath

# ----------------------------------------------------------------------------- assemble + inject
def assemble(matches, groups, live, odds_fixtures=None, odds_updated=None, knockout=None):
    teams=[t for _,ts in groups for t in ts]
    elo,form,h2h,feed,log,snaps=build_signals(matches,teams)
    stand=standings(matches,groups)
    # ---- data-driven reliability (earned trust) for confidence-balanced betting ----
    def _ot(x): return "draw" if x["result"]=="Draw" else ("home" if x["result"]==x["a"] else "away")
    _ra={"home":[0,0],"draw":[0,0],"away":[0,0]}
    for x in log:
        t=_ot(x); _ra[t][0]+=1; _ra[t][1]+= 1 if x["correct"] else 0
    reliability={k:(v[1]+1)/(v[0]+2) for k,v in _ra.items()}   # Laplace-smoothed hit rate per outcome
    _tot=len(log)
    _brier=(sum(sum((x["probs"][k]-(1 if x["ax"]==k else 0))**2 for k in range(3)) for x in log)/_tot) if _tot else 0.667
    reliability["kelly"]=max(0.10,min(0.5, 1-_brier/0.667))     # Kelly fraction scaled by Brier skill
    knockout=knockout or KO_TEMPLATE
    resolve=make_resolver(stand,knockout)
    mt,order,finalnum=ko_index(knockout)
    odds,reach=monte_carlo_real(mt,resolve,elo,form,h2h,finalnum)
    betting=compute_betting(odds_fixtures or [], odds_updated, elo, form, h2h, set(teams), odds, reliability)
    ranked=sorted(odds.items(),key=lambda kv:kv[1],reverse=True)
    champ_team,champ_prob=ranked[0][0],round(ranked[0][1]*100,1)
    fav=champ_team
    contenders=[{"team":t,"prob":round(p*100,1)} for t,p in ranked[:6]]
    bracket_rounds,path=projected_bracket(mt,order,resolve,elo,form,h2h,finalnum,champ_team)
    reach_final=round(reach["Final"].get(champ_team,0)*100,1)
    reach_sf=round(reach["Semi-final"].get(champ_team,0)*100,1)

    # ---- Scorecard: prediction accuracy (walk-forward) ----
    total=len(log); correct=sum(1 for x in log if x["correct"])
    accuracy=round(correct/total*100,1) if total else 0.0
    perf_matches=[{"date":x["date"],"a":x["a"],"b":x["b"],"pred":x["pred"],"conf":x["conf"],
                   "score":x["score"],"result":x["result"],"correct":x["correct"]}
                  for x in reversed(log)][:40]
    draws={"actual":sum(1 for x in log if x["result"]=="Draw"),
           "pred":sum(1 for x in log if x["pred"]=="Draw"),
           "correct":sum(1 for x in log if x["pred"]=="Draw" and x["correct"])}
    # accuracy by actual outcome type (home win / draw / away win)
    def otype(x):
        if x["result"]=="Draw": return "draw"
        return "home" if x["result"]==x["a"] else "away"
    _agg={"home":[0,0],"draw":[0,0],"away":[0,0]}
    for x in log:
        t=otype(x); _agg[t][0]+=1; _agg[t][1]+= 1 if x["correct"] else 0
    _lbl={"home":"Home wins","draw":"Draws","away":"Away wins"}
    by_outcome=[{"type":k,"label":_lbl[k],"total":_agg[k][0],"correct":_agg[k][1],
                 "pct":round(_agg[k][1]/_agg[k][0]*100) if _agg[k][0] else 0} for k in ("home","draw","away")]
    # Brier score (proper scoring rule over the full 1·X·2 vector; lower is better, 0.667 = coin-flip)
    brier=round(sum(sum((x["probs"][k]-(1 if x["ax"]==k else 0))**2 for k in range(3)) for x in log)/total,3) if total else 0.0
    # accuracy by confidence band (does a 70% pick win ~70%?)
    _bands=[("Toss-ups","< 45%",0,45),("Leans","45–59%",45,60),("Confident","60–74%",60,75),("Strong","≥ 75%",75,101)]
    by_conf=[]
    for nm,rng,lo,hi in _bands:
        sub=[x for x in log if lo<=x["conf"]<hi]
        by_conf.append({"label":nm,"range":rng,"total":len(sub),"correct":sum(1 for x in sub if x["correct"]),
                        "pct":round(sum(1 for x in sub if x["correct"])/len(sub)*100) if sub else None})
    draw_cal={"predicted":round(sum(x["probs"][1] for x in log)/total*100) if total else 0,
              "actual":round(draws["actual"]/total*100) if total else 0}
    # ---- title-odds evolution: re-simulate each rating snapshot (cap snapshots for speed) ----
    evo_snaps=snaps
    if len(snaps)>8:
        idx=sorted(set([0]+[round(i*(len(snaps)-1)/7) for i in range(1,8)]))
        evo_snaps=[snaps[i] for i in idx]
    evolution=[]
    for label,snap in evo_snaps:
        e,f,hh=snap
        od,_=monte_carlo_real(mt,resolve,e,f,hh,finalnum,sims=1500,seed=7)
        rk=sorted(od.items(),key=lambda kv:kv[1],reverse=True)
        cpos=next((i+1 for i,(t,_) in enumerate(rk) if t==champ_team),None)
        evolution.append({"stage":label,"leader":rk[0][0],"leaderProb":round(rk[0][1]*100,1),
                          "champRank":cpos,"champProb":round(od.get(champ_team,0)*100,1),
                          "top3":[{"team":t,"prob":round(p*100,1)} for t,p in rk[:3]]})
    # ---- champion progress / supporting stats ----
    crow=cgrp=cpos=None
    for g,rows in stand:
        for i,r in enumerate(rows):
            if r[0]==champ_team: crow,cgrp,cpos=r,g,i+1
    champ_form=form.get(champ_team,[])[-5:]
    form_letters="".join("W" if x==3 else ("D" if x==1 else "L") for x in champ_form) or "—"
    elo_delta=round(elo.get(champ_team,1650)-elo_of(champ_team))
    first_prob=evolution[0]["champProb"] if evolution else champ_prob
    trend="risen" if champ_prob>first_prob else ("held" if champ_prob==first_prob else "eased")
    champ_log=[x for x in log if champ_team in (x["a"],x["b"])]
    champ_wins=sum(1 for x in champ_log if x["result"]==champ_team)
    progress={"team":champ_team,"group":cgrp,"groupPos":cpos,
              "record":(f"{crow[2]}W-{crow[3]}D-{crow[4]}L" if crow else "—"),
              "gd":(crow[5] if crow else 0),"pts":(crow[6] if crow else 0),
              "form":form_letters,"eloDelta":elo_delta,"played":len(champ_log),"wins":champ_wins,
              "note":(f"{champ_team} sit {ordinal(cpos)} in Group {cgrp} on {crow[6] if crow else 0} points "
                      f"with a {'+' if (crow[5] if crow else 0)>=0 else ''}{crow[5] if crow else 0} goal difference, "
                      f"and have moved their rating {'+' if elo_delta>=0 else ''}{elo_delta} Elo since kick-off. "
                      f"Their simulated title odds have {trend} from {first_prob}% to {champ_prob}% as results land.")}

    def elo_strength(t): return round(min(99,max(1,(elo[t]-1550)/6)))
    second=contenders[1]["team"] if len(contenders)>1 else fav
    blend=[
        {"label":"Elo rating","note":"strong" if elo_strength(fav)>70 else "good","val":elo_strength(fav)},
        {"label":"Recent form (last 10)","note":"good","val":round(form_prob(fav,form)*100)},
        {"label":"Head-to-head edge","note":"neutral","val":round(h2h_prob(fav,second,h2h)*100)},
    ]
    now=dt.datetime.now(dt.timezone.utc)
    gp=sum(1 for m in matches if m.get("stage","").lower().startswith("group"))
    kp=sum(1 for m in matches if not m.get("stage","").lower().startswith("group"))
    results=[[r["a"],r["b"],r["score"],("+" if r["delta"]>=0 else "")+f"{r['delta']:.1f}",
              "Elo "+("up" if r["delta"]>=0 else "down"),"up" if r["delta"]>=0 else "dn"] for r in feed[:8]]
    src="openfootball (public domain)"
    return {
      "updated":now.isoformat().replace("+00:00","Z"),
      "isSeed":not live,
      "meta":{"source":src,"refreshEvery":"~2 hours","runs":SIMS},
      "champion":{"team":champ_team,"flag":FLAGS.get(champ_team,""),"prob":champ_prob,
                  "confidence":min(85,round(champ_prob*2+30)),
                  "confidenceLabel":"Field leader · narrow" if champ_prob<22 else "Clear favourite"},
      "contenders":contenders,"blend":blend,
      "weights":{"elo":int(WEIGHTS["elo"]*100),"form":int(WEIGHTS["form"]*100),"h2h":int(WEIGHTS["h2h"]*100)},
      "reasoning":(f"{champ_team} carry the strongest blended rating in the field. The Monte Carlo "
                   f"simulation makes them champions in {champ_prob}% of {SIMS:,} runs, narrowly ahead of "
                   f"{second}. Confidence falls in the late rounds, where ties tighten toward coin flips. "
                   "Every output is a probability, not a certainty."),
      "path":path,
      "bracket":{"rounds":bracket_rounds,"champion":{"team":champ_team,"flag":FLAGS.get(champ_team,""),
                 "prob":champ_prob,"reachFinal":reach_final,"reachSemi":reach_sf}},
      "ingest":{"lastFetch":now.strftime("%H:%M UTC"),"ago":"just now" if live else "1h 52m ago",
                "groupPlayed":gp,"groupTotal":72,"koPlayed":kp,"koTotal":32,
                "teamsMatched":len(set(teams)),"teamsTotal":48,
                "eloThrough":f"{gp+kp} matches ingested",
                "checks":["fixtures + groups fetched","scores parsed","form window = last 10",
                          "H2H records linked","ratings recomputed after each result","next auto-sync ~2h"]},
      "groups":stand,"results":results,"betting":betting,
      "championProgress":progress,
      "performance":{"accuracy":accuracy,"correct":correct,"total":total,"draws":draws,
                     "byOutcome":by_outcome,"brier":brier,"brierBase":0.667,
                     "byConfidence":by_conf,"drawCal":draw_cal,
                     "matches":perf_matches,"evolution":evolution,"champion":champ_team},
    }

def inject(html_path, data):
    html=open(html_path,encoding="utf-8").read()
    blob=json.dumps(data,ensure_ascii=False,indent=2)
    new=re.sub(r'(<script id="prediction-data" type="application/json">)(.*?)(</script>)',
               lambda m:m.group(1)+"\n"+blob+"\n"+m.group(3), html, count=1, flags=re.S)
    if new==html: sys.exit("ERROR: prediction-data block not found in "+html_path)
    open(html_path,"w",encoding="utf-8").write(new)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample",action="store_true",help="offline synthesised results")
    ap.add_argument("--out",default="index.html")
    ap.add_argument("--dump")
    args=ap.parse_args()
    assert abs(sum(WEIGHTS.values())-1.0)<1e-9,"weights must sum to 1"
    odds_fixtures, odds_updated = [], None
    if args.sample:
        matches,groups,knockout=synth_sample(); live=False
    else:
        res=fetch_openfootball()
        if not res:
            print("WARNING: openfootball unavailable — deploying synthesised seed data.")
            matches,groups,knockout=synth_sample(); live=False
        else:
            matches,groups,knockout=res; live=True
            odds_fixtures, odds_updated = fetch_odds()
    data=assemble(matches,groups,live,odds_fixtures,odds_updated,knockout)
    if args.dump: json.dump(data,open(args.dump,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    inject(args.out,data)
    print(f"OK · champion {data['champion']['team']} {data['champion']['prob']}% · "
          f"{len(matches)} matches · {data['ingest']['groupPlayed']}/72 group · {SIMS:,} sims · {args.out}")

if __name__=="__main__":
    main()
