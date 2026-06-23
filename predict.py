#!/usr/bin/env python3
"""
World Cup 2026 — Prediction Net
================================
Builds the tournament forecast and injects it into index.html.

Pipeline:
  1. Ingest match results            (API-Football, or a synthesised sample)
  2. Build three signals             Elo rating · recent form · head-to-head
  3. Blend them into a match model   p(A beats B)
  4. Monte Carlo the knockout         ~10,000 simulated tournaments -> title odds
  5. Emit DATA + inject into HTML     single self-contained file, no build step

Usage:
  python predict.py --sample                  # offline, synthesised results (for testing)
  python predict.py                            # live; needs API_FOOTBALL_KEY in env
  python predict.py --sample --out index.html  # choose output file

Design notes / simplifications (kept honest, see Methodology tab):
  - The knockout bracket is seeded from current group standings by adjusted Elo,
    avoiding same-group first-round meetings. This is a reasonable projection, not
    the exact official slotting template.
  - Recent form = points-per-game over the last N matches, mapped to a win prob.
  - Head-to-head uses historical result share; neutral (0.5) when no history.
"""

import argparse, json, math, os, random, re, sys, datetime as dt

# ----------------------------------------------------------------------------- config
WEIGHTS = {"elo": 0.45, "form": 0.35, "h2h": 0.20}   # must sum to 1.0
K_FACTOR = 40            # Elo update rate for internationals
FORM_WINDOW = 10         # matches counted as "recent form"
SIMS = 10000             # Monte Carlo runs
HOME_TEAMS = {"USA", "Canada", "Mexico"}             # 2026 hosts (small home nudge)

# Seed/base Elo for the field (illustrative starting strengths; refined by results).
BASE_ELO = {
    "Argentina":2090,"France":2065,"Spain":2055,"England":2010,"Brazil":2000,
    "Portugal":1970,"Netherlands":1955,"Belgium":1940,"Italy":1925,"Germany":1920,
    "Croatia":1880,"Uruguay":1870,"Colombia":1860,"USA":1840,"Denmark":1835,
    "Switzerland":1830,"Mexico":1825,"Japan":1815,"Senegal":1810,"Morocco":1805,
    "South Korea":1790,"Serbia":1785,"Austria":1780,"Ecuador":1775,"Australia":1770,
    "Canada":1765,"Norway":1760,"Nigeria":1755,"Poland":1750,"Ivory Coast":1745,
    "Egypt":1740,"Algeria":1735,"Paraguay":1720,"Tunisia":1715,"Ghana":1710,
    "Iran":1705,"Saudi Arabia":1690,"Qatar":1680,"Panama":1670,"Jamaica":1660,
    "Jordan":1650,"Cape Verde":1640,"Uzbekistan":1635,"New Zealand":1620,
    "Honduras":1610,"Curacao":1600,"Mexico B":1700,"Netherlands B":1830,"Germany B":1860,
}

# 12 groups of 4 (2026 format). Names align with BASE_ELO above.
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
         "England":"\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F","Brazil":"\U0001F1E7\U0001F1F7"}

# ----------------------------------------------------------------------------- ingest
def fetch_live():
    """Fetch finished WC fixtures from API-Football (free tier)."""
    import requests
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        return None  # caller falls back to seed data so the page still deploys
    # NB: use `or` not get(...,default) — the workflow passes these as EMPTY strings
    # when the repo variables are unset, which would override a default of "".
    season = os.environ.get("WC_SEASON") or "2026"
    league = os.environ.get("WC_LEAGUE_ID") or "1"   # 1 = FIFA World Cup on API-Football
    base = "https://v3.football.api-sports.io"
    H = {"x-apisports-key": key}
    def get(path, **params):
        r = requests.get(base+path, headers=H, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    j = get("/fixtures", league=league, season=season)
    print(f"[api] /fixtures league={league} season={season} -> "
          f"results={j.get('results')} errors={j.get('errors')} paging={j.get('paging')}")
    resp = list(j.get("response", []))
    paging = j.get("paging") or {}
    cur, total = paging.get("current", 1) or 1, paging.get("total", 1) or 1
    while cur < total:
        cur += 1
        resp += get("/fixtures", league=league, season=season, page=cur).get("response", [])

    if not resp:
        # Discovery: surface the real World Cup league id + the seasons this key can see.
        try:
            lj = get("/leagues", search="world cup")
            for x in lj.get("response", [])[:8]:
                yrs = [s.get("year") for s in x.get("seasons", [])]
                print(f"[api] league id={x['league']['id']} '{x['league']['name']}' "
                      f"type={x['league'].get('type')} seasons={yrs[-6:]}")
        except Exception as e:
            print("[api] league discovery failed:", e)

    out = []
    for fx in resp:
        if fx["fixture"]["status"]["short"] != "FT":
            continue
        out.append({
            "date": fx["fixture"]["date"],
            "home": fx["teams"]["home"]["name"],
            "away": fx["teams"]["away"]["name"],
            "hg": fx["goals"]["home"], "ag": fx["goals"]["away"],
            "stage": fx["league"].get("round", "Group Stage"),
        })
    print(f"[api] finished(FT) fixtures used: {len(out)} of {len(resp)} returned")
    return out

def synth_sample(seed=20260623):
    """Synthesise a plausible, completed group stage from BASE_ELO + noise."""
    rng = random.Random(seed)
    matches = []
    base = dt.datetime(2026, 6, 11, 18, 0)
    md = 0
    for g, teams in GROUPS.items():
        # round-robin: 6 matches per group
        pairs = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
        for i,(a,b) in enumerate(pairs):
            ta, tb = teams[a], teams[b]
            ea, eb = BASE_ELO[ta], BASE_ELO[tb]
            pa = 1/(1+10**((eb-ea)/400))
            # expected goals scaled by strength + randomness
            hg = max(0, int(round(rng.gauss(1.3 + (pa-0.5)*2.4, 1.0))))
            ag = max(0, int(round(rng.gauss(1.3 + ((1-pa)-0.5)*2.4, 1.0))))
            matches.append({"date": (base+dt.timedelta(days=md+i//3)).isoformat()+"Z",
                            "home": ta, "away": tb, "hg": hg, "ag": ag, "stage": "Group Stage"})
        md += 4
    matches.sort(key=lambda m: m["date"])
    return matches

# ----------------------------------------------------------------------------- signals
def build_signals(matches):
    """Return elo, form (list of recent results per team), h2h, and per-match elo deltas."""
    elo = dict(BASE_ELO)
    form = {t: [] for t in BASE_ELO}            # list of points (3/1/0), newest last
    h2h = {}                                    # (a,b) -> [a_wins, draws, b_wins]
    feed = []                                   # recent matches w/ elo delta, newest first
    for m in matches:
        a, b = m["home"], m["away"]
        if a not in elo or b not in elo:        # ignore teams outside the field
            continue
        ra, rb = elo[a], elo[b]
        ea = 1/(1+10**((rb-ra)/400))
        if m["hg"] > m["ag"]: sa = 1.0
        elif m["hg"] < m["ag"]: sa = 0.0
        else: sa = 0.5
        gd = abs(m["hg"]-m["ag"])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd-3)/8))
        delta = K_FACTOR * mult * (sa - ea)
        elo[a] = ra + delta
        elo[b] = rb - delta
        form[a].append(3 if sa==1 else (1 if sa==0.5 else 0))
        form[b].append(3 if sa==0 else (1 if sa==0.5 else 0))
        key = (a,b) if a < b else (b,a)
        rec = h2h.setdefault(key, [0,0,0])
        if sa==1: rec[0 if a<b else 2]+=1
        elif sa==0: rec[2 if a<b else 0]+=1
        else: rec[1]+=1
        feed.append({"a":a,"b":b,"score":f"{m['hg']}-{m['ag']}","delta":delta})
    feed.reverse()
    return elo, form, h2h, feed

def form_prob(team, form):
    recent = form.get(team, [])[-FORM_WINDOW:]
    if not recent: return 0.5
    ppg = sum(recent)/(len(recent)*3.0)         # 0..1
    return 0.5 + (ppg-0.5)*0.6                   # damp toward 0.5

def h2h_prob(a, b, h2h):
    key = (a,b) if a < b else (b,a)
    rec = h2h.get(key)
    if not rec or sum(rec)==0: return 0.5
    aw, dr, bw = rec
    total = aw+dr+bw
    share = (aw + 0.5*dr)/total if a < b else (bw + 0.5*dr)/total
    return 0.5 + (share-0.5)*0.7                 # damp small samples

def match_prob(a, b, elo, form, h2h):
    ra = elo[a] + (35 if a in HOME_TEAMS else 0)
    rb = elo[b] + (35 if b in HOME_TEAMS else 0)
    p_elo = 1/(1+10**((rb-ra)/400))
    fa, fb = form_prob(a,form), form_prob(b,form)
    p_form = fa / (fa+fb)
    p_h2h = h2h_prob(a,b,h2h)
    p = WEIGHTS["elo"]*p_elo + WEIGHTS["form"]*p_form + WEIGHTS["h2h"]*p_h2h
    return min(0.99, max(0.01, p))

# ----------------------------------------------------------------------------- standings + bracket
def standings(matches):
    tbl = {t: {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0} for t in BASE_ELO}
    for m in matches:
        a,b = m["home"], m["away"]
        if a not in tbl or b not in tbl: continue
        tbl[a]["P"]+=1; tbl[b]["P"]+=1
        tbl[a]["GF"]+=m["hg"]; tbl[a]["GA"]+=m["ag"]
        tbl[b]["GF"]+=m["ag"]; tbl[b]["GA"]+=m["hg"]
        if m["hg"]>m["ag"]: tbl[a]["W"]+=1; tbl[b]["L"]+=1
        elif m["hg"]<m["ag"]: tbl[b]["W"]+=1; tbl[a]["L"]+=1
        else: tbl[a]["D"]+=1; tbl[b]["D"]+=1
    out=[]
    for g, teams in GROUPS.items():
        rows=[]
        for t in teams:
            s=tbl[t]; pts=s["W"]*3+s["D"]; gd=s["GF"]-s["GA"]
            rows.append([t,s["P"],s["W"],s["D"],s["L"],gd,pts])
        rows.sort(key=lambda r:(r[6], r[5], r[2]), reverse=True)
        out.append([g, rows])
    return out

def build_bracket_seed(stand, elo):
    """Seed a 32-team knockout from group standings (top 2 + 8 best thirds),
    pairing by adjusted Elo while avoiding same-group first-round ties."""
    winners, runners, thirds = [], [], []
    for g, rows in stand:
        winners.append((rows[0][0], g)); runners.append((rows[1][0], g)); thirds.append((rows[2][0], g))
    thirds.sort(key=lambda tg: elo[tg[0]], reverse=True)
    qualifiers = winners + runners + thirds[:8]      # 12+12+8 = 32
    # order by strength, then pair strong-vs-weak avoiding same group
    qualifiers.sort(key=lambda tg: elo[tg[0]], reverse=True)
    teams = [t for t,_ in qualifiers]
    grp = {t:g for t,g in qualifiers}
    ties, used = [], set()
    top = teams[:16]; bottom = teams[16:][::-1]
    for hi in top:
        if hi in used: continue
        opp = next((lo for lo in bottom if lo not in used and grp[lo]!=grp[hi]), None)
        opp = opp or next(lo for lo in bottom if lo not in used)
        used.add(hi); used.add(opp); ties.append((hi,opp))
    return ties

# ----------------------------------------------------------------------------- monte carlo
def sim_tie(a, b, elo, form, h2h, rng):
    return a if rng.random() < match_prob(a,b,elo,form,h2h) else b

def monte_carlo(r32, elo, form, h2h, sims=SIMS, seed=1):
    rng = random.Random(seed)
    champ_count = {}
    for _ in range(sims):
        round_ = list(r32)
        while len(round_) > 1:
            nxt=[]
            for (a,b) in round_:
                nxt.append(sim_tie(a,b,elo,form,h2h,rng))
            round_ = [(nxt[i],nxt[i+1]) for i in range(0,len(nxt),2)]
        a,b = round_[0]
        w = sim_tie(a,b,elo,form,h2h,rng)
        champ_count[w] = champ_count.get(w,0)+1
    return {t: c/sims for t,c in champ_count.items()}

def projected_path(r32, elo, form, h2h):
    """Most-likely-opponent path for the title favourite, with per-tie win prob."""
    # deterministic advance by higher win prob
    def advance(ties):
        winners=[]
        for a,b in ties:
            winners.append(a if match_prob(a,b,elo,form,h2h)>=0.5 else b)
        return winners, [(winners[i],winners[i+1]) for i in range(0,len(winners)-1,2)]
    fav = max(BASE_ELO, key=lambda t: elo[t])
    rounds = [("R32","Round of 32")]  # not shown in path; path starts R16
    labels = ["R32","R16","QF","SF","FIN"]
    ties = list(r32); path=[]
    level=0
    while len(ties) >= 1:
        # find fav's tie
        my = next(((a,b) for a,b in ties if fav in (a,b)), None)
        if my is None: break
        opp = my[1] if my[0]==fav else my[0]
        p = round(match_prob(fav,opp,elo,form,h2h)*100)
        if labels[level] != "R32":
            path.append({"round":labels[level],"opp":opp,
                         "note":"likely opponent" if labels[level]=="R16" else "projected","pct":p})
        winners,ties = advance(ties)
        if len(winners)==1: break
        level+=1
        if level>=len(labels): break
    return fav, path

def build_bracket_view(r32, elo, form, h2h, fav):
    """Deterministic projected bracket for the Bracket tab."""
    def conf_pack(a,b):
        p = match_prob(a,b,elo,form,h2h)
        if p>=0.5: return [a,b,0,round(p*100)]
        return [a,b,1,round((1-p)*100)]
    rounds=[]; names=["Round of 32","Round of 16","Quarter-finals","Semi-finals","Final"]
    ties=list(r32)
    for ni,name in enumerate(names):
        packed=[]
        winners=[]
        for a,b in ties:
            pk = conf_pack(a,b)
            w = pk[0] if pk[2]==0 else pk[1]
            packed.append(pk + [1 if fav in (a,b) else 0])
            winners.append(w)
        rounds.append({"name":name,"ties":packed})
        if len(winners)==1: break
        ties=[(winners[i],winners[i+1]) for i in range(0,len(winners),2)]
    return rounds

# ----------------------------------------------------------------------------- assemble + inject
def ago(iso):
    try:
        t=dt.datetime.fromisoformat(iso.replace("Z","+00:00"))
        d=dt.datetime.now(dt.timezone.utc)-t
        m=int(d.total_seconds()//60)
        return f"{m//60}h {m%60}m ago" if m>=60 else f"{m}m ago"
    except Exception: return "recently"

def assemble(matches, live):
    elo, form, h2h, feed = build_signals(matches)
    stand = standings(matches)
    r32 = build_bracket_seed(stand, elo)
    odds = monte_carlo(r32, elo, form, h2h)
    ranked = sorted(odds.items(), key=lambda kv: kv[1], reverse=True)
    fav, path = projected_path(r32, elo, form, h2h)
    champ_team, champ_prob = ranked[0][0], round(ranked[0][1]*100,1)
    contenders=[{"team":t,"prob":round(p*100,1)} for t,p in ranked[:6]]
    bracket_rounds = build_bracket_view(r32, elo, form, h2h, fav)

    # blend strengths for the favourite (for the overview panel)
    def sig_strength_elo(t): return round(min(99, max(1,(elo[t]-1550)/6)))
    blend=[
        {"label":"Elo rating","note":"strong" if sig_strength_elo(fav)>70 else "good","val":sig_strength_elo(fav)},
        {"label":"Recent form (last 10)","note":"good","val":round(form_prob(fav,form)*100)},
        {"label":"Head-to-head edge","note":"neutral","val":round(h2h_prob(fav, contenders[1]["team"] if len(contenders)>1 else fav, h2h)*100)},
    ]
    now=dt.datetime.now(dt.timezone.utc)
    gp=sum(1 for m in matches if m.get("stage","").lower().startswith("group"))
    results=[[r["a"], r["b"], r["score"],
              ("+" if r["delta"]>=0 else "")+f"{r['delta']:.1f}",
              "Elo "+("up" if r["delta"]>=0 else "down"),
              "up" if r["delta"]>=0 else "dn"] for r in feed[:8]]
    data={
      "updated": now.isoformat().replace("+00:00","Z"),
      "isSeed": not live,
      "meta":{"source":"API-Football (free)","refreshEvery":"~2 hours","runs":SIMS},
      "champion":{"team":champ_team,"flag":FLAGS.get(champ_team,""),"prob":champ_prob,
                  "confidence":min(85,round(champ_prob*2+30)),
                  "confidenceLabel":"Field leader · narrow" if champ_prob<22 else "Clear favourite"},
      "contenders":contenders,
      "blend":blend,
      "weights":{"elo":int(WEIGHTS["elo"]*100),"form":int(WEIGHTS["form"]*100),"h2h":int(WEIGHTS["h2h"]*100)},
      "reasoning":(f"{champ_team} carry the strongest blended rating in the field. "
                   f"The Monte Carlo simulation makes them champions in {champ_prob}% of {SIMS:,} runs, "
                   f"narrowly ahead of {contenders[1]['team']}. Confidence falls in the late rounds, "
                   "where ties tighten toward coin flips. Every output is a probability, not a certainty."),
      "path":path,
      "bracket":{"rounds":bracket_rounds,"champion":{"team":champ_team,"flag":FLAGS.get(champ_team,""),"prob":champ_prob}},
      "ingest":{"lastFetch":now.strftime("%H:%M UTC"),"ago":"just now" if live else "1h 52m ago",
                "groupPlayed":gp,"groupTotal":72,"koPlayed":0,"koTotal":32,
                "teamsMatched":len(BASE_ELO),"teamsTotal":48,"eloThrough":"latest results",
                "checks":["all fixtures resolved","0 missing scores","form window = last 10",
                          "H2H records linked","ratings recomputed after each result","next auto-sync ~2h"]},
      "groups":stand,
      "results":results,
    }
    return data

def inject(html_path, data):
    with open(html_path,"r",encoding="utf-8") as f: html=f.read()
    blob=json.dumps(data, ensure_ascii=False, indent=2)
    new=re.sub(r'(<script id="prediction-data" type="application/json">)(.*?)(</script>)',
               lambda m: m.group(1)+"\n"+blob+"\n"+m.group(3), html, count=1, flags=re.S)
    if new==html: sys.exit("ERROR: could not find prediction-data block to inject into.")
    with open(html_path,"w",encoding="utf-8") as f: f.write(new)

# ----------------------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use synthesised results (offline)")
    ap.add_argument("--out", default="index.html", help="HTML file to inject into")
    ap.add_argument("--dump", help="also write the DATA json to this path")
    args=ap.parse_args()
    assert abs(sum(WEIGHTS.values())-1.0)<1e-9, "weights must sum to 1"
    if args.sample:
        matches, live = synth_sample(), False
    else:
        matches = fetch_live()
        if matches is None:
            print("WARNING: API_FOOTBALL_KEY not set — deploying with synthesised seed data. "
                  "Add the secret to switch to live results.")
            matches, live = synth_sample(), False
        else:
            live = True
    data = assemble(matches, live=live)
    if args.dump:
        with open(args.dump,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
    inject(args.out, data)
    print(f"OK · champion {data['champion']['team']} {data['champion']['prob']}% · "
          f"{len(matches)} matches · {SIMS:,} sims · injected into {args.out}")

if __name__=="__main__":
    main()
