# (date, opp, usa_goals, opp_goals, comp, venue, usa_change, usa_after, opp_after)
M=[("2024-10-12","Panama",2,0,"F","H",11,1737,1719),("2024-10-15","Mexico",0,2,"F","A",-9,1728,1801),
("2024-11-14","Jamaica",1,0,"NL","A",19,1747,1586),("2024-11-18","Jamaica",4,2,"NL","H",11,1758,1575),
("2025-01-18","Venezuela",3,1,"F","H",10,1768,1734),("2025-01-22","Costa Rica",3,0,"F","H",8,1776,1645),
("2025-03-20","Panama",0,1,"NL","H",-29,1747,1740),("2025-03-23","Canada",1,2,"NL","H",-25,1722,1778),
("2025-06-07","Turkey",1,2,"F","H",-10,1712,1847),("2025-06-10","Switzerland",0,4,"F","H",-18,1694,1844),
("2025-06-15","Trinidad & Tobago",5,0,"GC","H",9,1703,1387),("2025-06-19","Saudi Arabia",1,0,"GC","H",12,1715,1597),
("2025-06-22","Haiti",2,1,"GC","H",6,1721,1467),("2025-06-29","Costa Rica",2,2,"GC","H",-9,1712,1693),
("2025-07-02","Guatemala",2,1,"GC","H",8,1720,1525),("2025-07-06","Mexico",1,2,"GC","H",-24,1696,1861),
("2025-09-06","South Korea",0,2,"F","H",-17,1679,1771),("2025-09-09","Japan",2,0,"F","H",19,1698,1863),
("2025-10-10","Ecuador",1,1,"F","H",4,1702,1929),("2025-10-14","Australia",2,1,"F","H",10,1712,1793),
("2025-11-15","Paraguay",2,1,"F","H",11,1723,1821),("2025-11-18","Uruguay",5,1,"F","H",24,1747,1890),
("2026-03-28","Belgium",2,5,"F","H",-17,1730,1866),("2026-03-31","Portugal",0,2,"F","H",-9,1721,1984),
("2026-05-31","Senegal",3,2,"F","H",12,1733,1867),("2026-06-06","Germany",1,2,"F","H",-7,1726,1932),
("2026-06-12","Paraguay",4,1,"WC","H",54,1780,1779),("2026-06-19","Australia",2,0,"WC","H",40,1820,1798),
("2026-06-25","Turkey",2,3,"WC","H",-39,1781,1852),("2026-07-01","Bosnia & Herz.",2,0,"WC","H",17,1798,1605),
("2026-07-06","Belgium",1,4,"WC","H",-51,1747,1960),("2026-09-26","Peru",4,1,"F","H",10,1757,1689),
("2026-09-29","Chile",4,2,"F","H",9,1766,1701),("2026-10-03","Mexico",3,0,"F","H",20,1786,1890),
("2026-10-06","Canada",1,0,"F","H",6,1792,1739)]
rows=[];
for d,o,g,a,c,v,ch,ua,oa in M:
    ub, ob = ua-ch, oa+ch
    dr = ub-ob + (100 if v=="H" else -100)
    we = 1/(10**(-dr/400)+1)
    w = 1 if g>a else 0.5 if g==a else 0
    rows.append((d,o,c,g,a,ub,ob,we,w,w-we,ch))
    print(f"{d} {o:18s} {c:3s} {g}-{a}  USA {ub} v {ob}  exp {we:.2f}  got {w:.1f}  resid {w-we:+.2f}  elo {ch:+d}")
import statistics as s
for name,sel in [("all",lambda r:True),("competitive",lambda r:r[2]!="F"),("friendly",lambda r:r[2]=="F"),("world cup",lambda r:r[2]=="WC")]:
    rs=[r[9] for r in rows if sel(r)]
    print(f"{name:12s} n={len(rs):2d} mean resid {s.mean(rs):+.3f}  sd {s.stdev(rs):.3f}  se {s.stdev(rs)/len(rs)**.5:.3f}")
wts=[1 if r[2]!="F" else 0.5 for r in rows]
wm=sum(w*r[9] for w,r in zip(wts,rows))/sum(wts)
print(f"weighted (friendly 0.5): {wm:+.3f}; elo start {rows[0][5]} end {M[-1][7]} net {sum(r[10] for r in rows):+d}")
print(f"expected pts {sum(r[7] for r in rows):.1f} actual {sum(r[8] for r in rows):.1f}")
