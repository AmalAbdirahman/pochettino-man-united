import re, csv, sys, statistics as st
src, out = sys.argv[1], sys.argv[2]
L=[l.strip() for l in open(src,encoding="utf-8")]
LABELS=["Ball possession","Expected goals (xG)","Total shots","Shots on target","Touches in opposition box","Big chances",
"Big chances missed","Accurate passes","Yellow cards","Red cards","Corners","Distance covered","Running","Sprinting","Number of sprints",
"xG open play","xG set play","Non-penalty xG","xG on target (xGOT)","Shots off target","Blocked shots","Hit woodwork",
"Shots inside box","Shots outside box","Duels won","Ground duels won","Aerial duels won","Successful dribbles","Fouls committed",
"Passes","Own half","Opposition half","Accurate long balls","Accurate crosses","Throws","Offsides","Tackles","Interceptions",
"Blocks","Clearances","Keeper saves"]
hdr=re.compile(r"^gw\s*(\d+)\s+(.+?)\s+(\d+)\s+(.+?)\s+(\d+)\s*$",re.I)
def num(s): return float(re.match(r"[\d.]+",s).group())
matches=[];cur=None;i=0
while i<len(L):
    m=hdr.match(L[i])
    if m:
        cur={"gw":int(m[1]),"home":m[2],"hg":int(m[3]),"away":m[4],"ag":int(m[5]),"s":{}};matches.append(cur);i+=1;continue
    if cur and L[i] in LABELS and L[i] not in cur["s"] and i+2<len(L):
        a,b=L[i+1],L[i+2]
        if a=="" or b=="" or (a in LABELS): i+=1;continue
        cur["s"][L[i]]=(num(a),num(b))
        if L[i]=="Passes": pass
        i+=3;continue
    i+=1
rows=[]
for m in matches:
    hu="manchester united" in m["home"].lower()
    opp=m["away"] if hu else m["home"]
    g=lambda k:(m["s"][k] if hu else m["s"][k][::-1])
    gf,ga=(m["hg"],m["ag"]) if hu else (m["ag"],m["hg"])
    r={"gameweek":m["gw"],"opponent":opp.title().replace("Aton","Aston"),"venue":"home" if hu else "away","goals_for":gf,"goals_against":ga,
       "points":3 if gf>ga else 1 if gf==ga else 0}
    for k,col in [("Ball possession","possession"),("Expected goals (xG)","xg"),("Non-penalty xG","npxg"),("xG open play","xg_open"),
        ("xG set play","xg_set"),("xG on target (xGOT)","xgot"),("Total shots","shots"),("Shots on target","sot"),("Big chances","big_chances"),
        ("Touches in opposition box","box_touches"),("Passes","passes"),("Accurate passes","acc_passes"),("Opposition half","acc_passes_opp_half"),
        ("Tackles","tackles"),("Interceptions","interceptions"),("Fouls committed","fouls"),("Corners","corners"),("Distance covered","distance_km"),
        ("Number of sprints","sprints"),("Accurate long balls","acc_long_balls"),("Accurate crosses","acc_crosses")]:
        f,a=g(k); r[col+"_for"]=f; r[col+"_against"]=a
    rows.append(r)
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(len(rows),"matches parsed")
for r in rows:
    tilt=r["acc_passes_opp_half_for"]/(r["acc_passes_opp_half_for"]+r["acc_passes_opp_half_against"])
    press=r["passes_against"]/(r["tackles_for"]+r["interceptions_for"]+r["fouls_for"])
    r["tilt"]=tilt;r["press"]=press
    print(f'GW{r["gameweek"]:>2} {r["venue"]:4s} {r["opponent"]:18s} {r["goals_for"]}-{r["goals_against"]}  xG {r["xg_for"]:.2f}-{r["xg_against"]:.2f}  npxG {r["npxg_for"]:.2f}-{r["npxg_against"]:.2f}  poss {r["possession_for"]:.0f}%  tilt {tilt:.0%}  passes/def.action {press:.1f}  setxG {r["xg_set_for"]:.2f}-{r["xg_set_against"]:.2f}')
n=len(rows); S=lambda k:sum(r[k] for r in rows)
print(f'\nPoints {S("points")} from {n} (PPG {S("points")/n:.2f}); goals {S("goals_for")}-{S("goals_against")} (GD {S("goals_for")-S("goals_against")})')
print(f'xG {S("xg_for"):.2f}-{S("xg_against"):.2f} (xGD/g {(S("xg_for")-S("xg_against"))/n:+.2f}); npxG {S("npxg_for"):.2f}-{S("npxg_against"):.2f} (npxGD/g {(S("npxg_for")-S("npxg_against"))/n:+.2f})')
print(f'goals - xG for {S("goals_for")-S("xg_for"):+.2f}; goals - xG against {S("goals_against")-S("xg_against"):+.2f}')
print(f'xGOT for {S("xgot_for"):.2f} vs xG {S("xg_for"):.2f}; xGOT against {S("xgot_against"):.2f} vs conceded {S("goals_against")}')
print(f'mean possession {st.mean(r["possession_for"] for r in rows):.1f}%  mean tilt {st.mean(r["tilt"] for r in rows):.1%}  mean passes per def action {st.mean(r["press"] for r in rows):.1f}')
print(f'shots {S("shots_for"):.0f}-{S("shots_against"):.0f}  box touches {S("box_touches_for"):.0f}-{S("box_touches_against"):.0f}  big chances {S("big_chances_for"):.0f}-{S("big_chances_against"):.0f}')
print(f'set-piece xG {S("xg_set_for"):.2f}-{S("xg_set_against"):.2f}  share of xG for from set pieces {S("xg_set_for")/S("xg_for"):.0%}, against {S("xg_set_against")/S("xg_against"):.0%}')
print(f'xG per shot for {S("xg_for")/S("shots_for"):.3f} against {S("xg_against")/S("shots_against"):.3f}')
print(f'pens: npxG diff for {S("xg_for")-S("npxg_for"):.2f}, against {S("xg_against")-S("npxg_against"):.2f}')
# expected points via Poisson from xG
import math
def pmf(l,k): return math.exp(-l)*l**k/math.factorial(k)
xp=0
for r in rows:
    pw=pd=0
    for i in range(11):
        for j in range(11):
            p=pmf(r["xg_for"],i)*pmf(r["xg_against"],j)
            if i>j: pw+=p
            elif i==j: pd+=p
    xp+=3*pw+pd
print(f'expected points from xG (Poisson) {xp:.1f} vs actual {S("points")}')
