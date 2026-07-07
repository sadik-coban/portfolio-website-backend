"""BI dashboard aggregation — served from the local DuckDB file, computed server-side.

This is a 1:1 Python port of the frontend's client-side aggregation
(app/_site/bi/compute.ts) + the columnar loader (scripts/build-bi-rows.mjs), moved
to the backend so the ~30K raw rows never reach the browser (privacy). The row set
is built once from cars.duckdb and cached in memory (numpy columns + dicts); every
request filters + aggregates in-memory (<50 ms over 30K rows).

Canonical row set = TR-plated (gb_plate_origin), latest snapshot per ad_id. Segment
is derived from the series (clean); damage is packed 11 parts x 2 bits.
"""
import math
import re
from datetime import datetime
from typing import Optional

import numpy as np

from app.core.db import get_db_connection, table_exists

# ── constants (mirror compute.ts / build-bi-rows.mjs) ────────────────────
SEG = ["B", "C", "D", "E", "F", "S"]
FUEL = ["Benzin", "Dizel", "LPG", "Hibrit"]
FUEL_COLORS = {"Benzin": "#059669", "Dizel": "#0d9aba", "LPG": "#7c5cff", "Hibrit": "#e08a1e"}
BRANDS = ["BMW", "Audi"]
PAL = ["#059669", "#7c5cff"]  # scatter palette by brand
PARTS = ["kaput", "tavan", "bagaj", "door_fl", "door_fr", "door_rl", "door_rr",
         "fender_fl", "fender_fr", "fender_rl", "fender_rr"]
CURRENT_YEAR = 2026
MATRIX_YEARS = [2021, 2022, 2023, 2024]
BUCKETS = [(0, 2014, "≤2014"), (2015, 2016, "15–16"), (2017, 2018, "17–18"),
           (2019, 2020, "19–20"), (2021, 2022, "21–22"), (2023, 9999, "23+")]
SEGMENT_MAP = {
    "1 Serisi": "C", "2 Serisi": "C", "3 Serisi": "D", "4 Serisi": "D", "5 Serisi": "E",
    "6 Serisi": "E", "7 Serisi": "F", "8 Serisi": "F",
    "X1": "C", "X2": "C", "X3": "D", "X4": "D", "X5": "E", "X6": "E", "X7": "F", "Z4": "S",
    "A1": "B", "A3": "C", "A4": "D", "A5": "D", "A6": "E", "A7": "E", "A8": "F",
    "Q2": "C", "Q3": "C", "Q5": "D", "Q7": "F", "Q8": "F", "TT": "S", "R8": "S",
}


def _fuel_idx(f) -> int:
    if f == "Benzin":
        return 0
    if f == "Dizel":
        return 1
    if f == "LPG & Benzin":
        return 2
    if f == "Hibrit":
        return 3
    return -1


def _seg_idx(series) -> int:
    return SEG.index(SEGMENT_MAP.get(series, "D"))


def _brand_idx(b) -> int:
    return 0 if b == "bmw" else 1


# ── small stats helpers (match compute.ts semantics) ─────────────────────
def _round(x) -> int:
    return int(math.floor(float(x) + 0.5))


def _fixed(x, d) -> float:
    m = 10 ** d
    return math.floor(float(x) * m + 0.5) / m


def _median(vals) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0
    m = n >> 1
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _quantile_sorted(s, q) -> float:
    n = len(s)
    if n == 0:
        return 0
    pos = (n - 1) * q
    base = int(math.floor(pos))
    rest = pos - base
    if base + 1 < n:
        return s[base] + rest * (s[base + 1] - s[base])
    return s[base]


def _mean(vals) -> float:
    n = len(vals)
    return (sum(vals) / n) if n else 0


def _dpart(dmg: int, i: int) -> int:
    return (dmg >> (2 * i)) & 3


# ── columnar row loader (cached) — mirrors build-bi-rows.mjs ──────────────
_ROWS: Optional[dict] = None


def _part_case(p: str, i: int) -> str:
    return (f"(CASE WHEN {p}_degisen>0 THEN 3 WHEN {p}_boyali>0 THEN 2 "
            f"WHEN {p}_lokal>0 THEN 1 ELSE 0 END) AS d{i}")


def load_rows(force: bool = False) -> dict:
    """Build (once) the columnar row set from cars.duckdb. Returns {cols, dict, meta}."""
    global _ROWS
    if _ROWS is not None and not force:
        return _ROWS

    dmg_select = ", ".join(_part_case(p, i) for i, p in enumerate(PARTS))
    view_sql = r"""
        CREATE OR REPLACE TEMP VIEW cars AS
        WITH ranked AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY ad_id ORDER BY search_date DESC, scraped_at DESC) rn
          FROM car_listings
          WHERE price > 0 AND gb_plate_origin = '(TR) Türkiye'
        )
        SELECT *,
          trim(regexp_extract(location, ',\s*([^,]+)$', 1)) AS city,
          trim(coalesce(nullif(regexp_extract(location, 'Mh\.?\s*([^,]+),', 1), ''),
                        regexp_extract(location, '^([^,]+),', 1))) AS district
        FROM ranked WHERE rn = 1;
    """
    main_sql = f"""
        SELECT brand, series, model, gb_year AS y, gb_mileage AS km, kb_fuel AS fuel,
               kb_body_type AS body, is_heavy_damaged AS hd, price AS p, city, district,
               listing_date::VARCHAR AS ld, {dmg_select}
        FROM cars
        ORDER BY listing_date DESC
    """
    meta_sql = """
        SELECT COUNT(*) n_unique, (SELECT COUNT(*) FROM car_listings) n_raw,
               ROUND(AVG(NULLIF(tramer_fee,0))) tramer_avg,
               SUM(CASE WHEN tramer_fee>0 THEN 1 ELSE 0 END) tramer_n
        FROM cars WHERE price > 0
    """
    snap_sql = "SELECT search_date::VARCHAR d, COUNT(*) n FROM car_listings GROUP BY 1 ORDER BY 1"

    with get_db_connection() as conn:
        conn.execute(view_sql)
        rows = conn.execute(main_sql).fetchall()
        meta_row = conn.execute(meta_sql).fetchone()
        snaps = conn.execute(snap_sql).fetchall()

    # base date = earliest listing_date across the set
    lds = sorted(r[11] for r in rows if r[11])
    base_date = lds[0] if lds else None
    base_dt = datetime.fromisoformat(base_date) if base_date else None

    def day_offset(iso):
        if not iso or base_dt is None:
            return -1
        return round((datetime.fromisoformat(iso) - base_dt).total_seconds() / 86400.0)

    # dictionaries (interned in row order; indices are internally consistent)
    series_dict, series_map = [], {}
    city_dict, city_map = [], {}
    dist_dict, dist_map = [], {}
    model_dict, model_map = [], {}
    body_dict, body_map = [], {}

    def intern(dic, mp, key, make):
        if key in mp:
            return mp[key]
        i = len(dic)
        dic.append(make())
        mp[key] = i
        return i

    p_, y_, km_, f_, s_, b_, se_, ci_, di_, md_, bt_, ld_, hd_, dmg_ = ([] for _ in range(14))
    for r in rows:
        brand, series, model, y, km, fuel, body, hd, price, city, district, ld = r[:12]
        b = _brand_idx(brand)
        se = intern(series_dict, series_map, f"{b} {series or ''}",
                    (lambda b=b, series=series: {"b": b, "name": series or "—"}))
        ci = intern(city_dict, city_map, city, (lambda city=city: city)) if city else -1
        di = intern(dist_dict, dist_map, f"{ci} {district}", (lambda district=district: district)) if district else -1
        md = intern(model_dict, model_map, model or series or "—", (lambda model=model, series=series: model or series or "—"))
        bt = intern(body_dict, body_map, body, (lambda body=body: body)) if body else -1
        dmg = 0
        for i in range(len(PARTS)):
            dmg |= (int(r[12 + i]) & 3) << (2 * i)
        p_.append(int(price))
        y_.append(int(y) if y is not None else -1)
        km_.append(int(km) if km is not None else -1)
        f_.append(_fuel_idx(fuel))
        s_.append(_seg_idx(series))
        b_.append(b)
        se_.append(se)
        ci_.append(ci)
        di_.append(di)
        md_.append(md)
        bt_.append(bt)
        ld_.append(day_offset(ld))
        hd_.append(1 if hd else 0)
        dmg_.append(dmg)

    cols = {
        "p": np.array(p_, dtype=np.int64), "y": np.array(y_, dtype=np.int64),
        "km": np.array(km_, dtype=np.int64), "f": np.array(f_, dtype=np.int64),
        "s": np.array(s_, dtype=np.int64), "b": np.array(b_, dtype=np.int64),
        "se": np.array(se_, dtype=np.int64), "ci": np.array(ci_, dtype=np.int64),
        "di": np.array(di_, dtype=np.int64), "md": np.array(md_, dtype=np.int64),
        "bt": np.array(bt_, dtype=np.int64), "ld": np.array(ld_, dtype=np.int64),
        "hd": np.array(hd_, dtype=np.int64), "dmg": np.array(dmg_, dtype=np.int64),
    }
    n_unique, n_raw, tramer_avg, tramer_n = meta_row
    meta = {
        "n_raw": int(n_raw), "n_unique": int(n_unique), "base_date": base_date,
        "brands": BRANDS, "fuels": FUEL, "segments": SEG, "parts": PARTS,
        "snapshots": [{"date": d, "n": int(n)} for d, n in snaps],
        "tramer_avg": int(tramer_avg) if tramer_avg is not None else 0,
        "tramer_n": int(tramer_n) if tramer_n is not None else 0,
        "note": "TR-plated latest snapshot per ad_id; segment from series; server-side aggregation.",
    }
    _ROWS = {
        "cols": cols,
        "dict": {"series": series_dict, "city": city_dict, "district": dist_dict,
                 "model": model_dict, "body": body_dict},
        "meta": meta,
    }
    return _ROWS


# ── filtering (port of filterRows) ───────────────────────────────────────
def filter_rows(rows: dict, f: dict) -> np.ndarray:
    c = rows["cols"]
    n = c["p"].shape[0]
    mask = np.ones(n, dtype=bool)
    if f.get("brand", -1) >= 0:
        mask &= c["b"] == f["brand"]
    if f.get("series", -1) >= 0:
        mask &= c["se"] == f["series"]
    if f.get("fuel", -1) >= 0:
        mask &= c["f"] == f["fuel"]
    if f.get("seg", -1) >= 0:
        mask &= c["s"] == f["seg"]
    if f.get("damage", -1) == 0:
        mask &= c["hd"] == 0
    elif f.get("damage", -1) == 1:
        mask &= c["hd"] == 1
    if f.get("yearMin") is not None:
        mask &= (c["y"] >= 0) & (c["y"] >= f["yearMin"])
    if f.get("yearMax") is not None:
        mask &= (c["y"] >= 0) & (c["y"] <= f["yearMax"])
    if f.get("priceMin") is not None:
        mask &= c["p"] >= f["priceMin"]
    if f.get("priceMax") is not None:
        mask &= c["p"] <= f["priceMax"]
    if f.get("kmMin") is not None:
        mask &= (c["km"] >= 0) & (c["km"] >= f["kmMin"])
    if f.get("kmMax") is not None:
        mask &= (c["km"] >= 0) & (c["km"] <= f["kmMax"])
    return np.nonzero(mask)[0]  # preserves row order (date-desc)


def _day_to_iso(base: str, off: int) -> str:
    d = datetime.fromisoformat(base)
    return (d.fromordinal(d.toordinal() + off)).isoformat()[:10]


# ── main aggregation (port of computeAgg) ────────────────────────────────
def compute_agg(rows: dict, idx: np.ndarray) -> dict:
    c = rows["cols"]
    dic = rows["dict"]
    meta = rows["meta"]
    n = int(idx.shape[0])

    p = c["p"][idx]; y = c["y"][idx]; km = c["km"][idx]; hd = c["hd"][idx]
    b = c["b"][idx]; s = c["s"][idx]; se = c["se"][idx]; f = c["f"][idx]
    ci = c["ci"][idx]; di = c["di"][idx]; md = c["md"][idx]; bt = c["bt"][idx]

    # KPIs
    prices = p.tolist()
    kms = km[km >= 0].tolist()
    age_mask = y > 0
    age_sum = int((CURRENT_YEAR - y[age_mask]).sum()) if age_mask.any() else 0
    age_n = int(age_mask.sum())
    damaged_n = int((hd == 1).sum())
    avg_price = _round(_mean(prices))
    median_price = _round(_median(prices))
    median_km = _round(_median(kms))
    avg_age = _fixed(age_sum / age_n, 1) if age_n else 0
    clean_pct = _fixed(100 * (n - damaged_n) / n, 1) if n else 0

    # price/km by year
    by_year_p, by_year_km = {}, {}
    for i in range(n):
        yy = int(y[i])
        if yy <= 0:
            continue
        by_year_p.setdefault(yy, []).append(int(p[i]))
        if km[i] >= 0:
            by_year_km.setdefault(yy, []).append(int(km[i]))
    price_by_year = sorted(
        ({"year": yr, "price": _round(_mean(arr)), "n": len(arr)}
         for yr, arr in by_year_p.items() if len(arr) >= 3),
        key=lambda r: r["year"])
    spark_years = [r["year"] for r in price_by_year]
    price_spark = [r["price"] for r in price_by_year]
    km_spark = [_round(_median(by_year_km.get(yr, []))) for yr in spark_years]

    # segment prices
    seg_arr = [[] for _ in SEG]
    for i in range(n):
        si = int(s[i])
        if si >= 0:
            seg_arr[si].append(int(p[i]))
    segment_price = [
        {"seg": SEG[si], "avg": _round(_mean(seg_arr[si])), "median": _round(_median(seg_arr[si])), "n": len(seg_arr[si])}
        for si in range(len(SEG)) if len(seg_arr[si]) > 0
    ]

    # matrix: top-7 series x year buckets
    se_count = {}
    for i in range(n):
        se_count[int(se[i])] = se_count.get(int(se[i]), 0) + 1
    top_se = [e[0] for e in sorted(se_count.items(), key=lambda kv: -kv[1])[:7]]
    se_row_idx = {s_: r for r, s_ in enumerate(top_se)}
    cells = [[0] * len(BUCKETS) for _ in top_se]

    def bucket_idx(yy):
        for bi, (lo, hi, _) in enumerate(BUCKETS):
            if lo <= yy <= hi:
                return bi
        return -1
    for i in range(n):
        r = se_row_idx.get(int(se[i]))
        if r is None:
            continue
        bi = bucket_idx(int(y[i]))
        if bi >= 0:
            cells[r][bi] += 1
    matrix_rows = []
    for r, s_ in enumerate(top_se):
        sd = dic["series"][s_]
        matrix_rows.append({"series": sd["name"], "brand": BRANDS[sd["b"]],
                            "cells": cells[r], "total": sum(cells[r])})
    col_totals = [sum(row["cells"][bi] for row in matrix_rows) for bi in range(len(BUCKETS))]

    # recent (idx already date-desc)
    recent = []
    for k in range(min(14, n)):
        i = k
        recent.append({
            "id": k + 1, "brand": BRANDS[int(b[i])], "model": dic["model"][int(md[i])] or "—",
            "year": int(y[i]), "km": int(km[i]), "fuel": (meta["fuels"][int(f[i])] if 0 <= f[i] < len(meta["fuels"]) else "—"),
            "city": (dic["city"][int(ci[i])] if ci[i] >= 0 else "—"),
            "damaged": bool(hd[i] == 1), "price": int(p[i]),
        })

    # daily volume — last 15 days of the global window
    max_ld = int(c["ld"].max()) if c["ld"].size else 0
    win_start = max_ld - 14
    day_counts = [0] * 15
    ld_f = c["ld"][idx]
    for i in range(n):
        off = int(ld_f[i])
        if win_start <= off <= max_ld:
            day_counts[off - win_start] += 1
    highlight_idx = day_counts.index(max(day_counts)) if day_counts else 0
    daily_volume = {
        "days": [_day_to_iso(meta["base_date"], win_start + k) for k in range(15)] if meta["base_date"] else [],
        "counts": day_counts, "highlightIdx": highlight_idx,
        "thisWeek": sum(day_counts[-7:]), "lastWeek": sum(day_counts[-14:-7]),
    }

    # fuel x year (clustered)
    y_pos = {yy: k for k, yy in enumerate(MATRIX_YEARS)}
    fuel_year_grid = [[0] * len(MATRIX_YEARS) for _ in FUEL]
    for i in range(n):
        fi = int(f[i]); yp = y_pos.get(int(y[i]))
        if fi >= 0 and yp is not None:
            fuel_year_grid[fi][yp] += 1
    fuel_year = {"years": [str(yy) for yy in MATRIX_YEARS],
                 "series": [{"name": nm, "color": FUEL_COLORS[nm], "data": fuel_year_grid[fi]}
                            for fi, nm in enumerate(FUEL)]}

    # fuel donut
    fuel_count = [0] * len(FUEL)
    for i in range(n):
        fi = int(f[i])
        if fi >= 0:
            fuel_count[fi] += 1
    fuel_donut = [{"name": nm, "value": fuel_count[fi], "color": FUEL_COLORS[nm]}
                  for fi, nm in enumerate(FUEL) if fuel_count[fi] > 0]

    # brand range (box)
    brand_prices = [[] for _ in BRANDS]
    for i in range(n):
        brand_prices[int(b[i])].append(int(p[i]))
    brand_range = []
    for bi, nm in enumerate(BRANDS):
        ss = sorted(brand_prices[bi])
        if not ss:
            continue
        brand_range.append({"brand": nm, "min": ss[0], "q1": _quantile_sorted(ss, 0.25),
                            "median": _quantile_sorted(ss, 0.5), "q3": _quantile_sorted(ss, 0.75), "max": ss[-1]})

    # density km x price
    km_bin, km_max, p_bin, p_max = 25000, 500000, 500000, 6000000
    nx, ny = km_max // km_bin, p_max // p_bin
    grid = {}
    dmax = 0
    for i in range(n):
        kmv = int(km[i]); pv = int(p[i])
        if kmv < 0 or kmv >= km_max or pv < 0 or pv >= p_max:
            continue
        xi = kmv // km_bin; yi = pv // p_bin
        key = (xi, yi)
        v = grid.get(key, 0) + 1
        grid[key] = v
        if v > dmax:
            dmax = v
    density = {
        "xLabels": [f"{(i * km_bin) / 1000:g}k" for i in range(nx)],
        "yLabels": [f"{i * (p_bin / 1e6):g}M" for i in range(ny)],
        "data": [[xi, yi, v] for (xi, yi), v in grid.items()],
        "max": dmax,
    }

    # scatter (sampled per brand, cap ~1400 total)
    cap = 1400
    step = max(1, math.ceil(n / cap))
    sc_pts = [[] for _ in BRANDS]
    for k in range(0, n, step):
        if km[k] >= 0:
            sc_pts[int(b[k])].append([int(km[k]), int(p[k])])
    scatter = [{"brand": nm, "color": PAL[bi], "points": sc_pts[bi]}
               for bi, nm in enumerate(BRANDS) if sc_pts[bi]]

    # provinces
    prov_arr = {}
    for i in range(n):
        cii = int(ci[i])
        if cii < 0:
            continue
        prov_arr.setdefault(cii, []).append(int(p[i]))
    provinces = sorted(
        ({"name": dic["city"][cii], "n": len(arr), "median": _round(_median(arr))}
         for cii, arr in prov_arr.items()),
        key=lambda r: -r["n"])
    prov_max = provinces[0]["n"] if provinces else 0

    # price histogram (₺M buckets, last bucket overflow)
    ph_bin, ph_max = 500000, 6000000
    ph_n = ph_max // ph_bin
    ph_bins = [0] * ph_n
    ph_over = 0
    for i in range(n):
        bi = int(p[i]) // ph_bin
        if bi >= ph_n:
            ph_over += 1
        elif bi >= 0:
            ph_bins[bi] += 1
    price_hist = {
        "labels": [f"{(k * ph_bin / 1e6):.1f}" for k in range(ph_n)] + [f"{ph_max / 1e6:g}+"],
        "counts": ph_bins + [ph_over],
    }

    # price by body type (box) — bodies with a meaningful count
    body_arr = {}
    for i in range(n):
        bti = int(bt[i])
        if bti < 0:
            continue
        body_arr.setdefault(bti, []).append(int(p[i]))
    body_box = []
    for bti, arr in body_arr.items():
        if len(arr) < 20:
            continue
        ss = sorted(arr)
        body_box.append({"body": dic["body"][bti], "min": ss[0], "q1": _quantile_sorted(ss, 0.25),
                        "median": _quantile_sorted(ss, 0.5), "q3": _quantile_sorted(ss, 0.75),
                        "max": ss[-1], "n": len(arr)})
    body_box.sort(key=lambda r: r["median"])

    # heavy-damage price impact
    clean_p = p[hd != 1].tolist()
    dmg_p = p[hd == 1].tolist()
    damage_impact = {"clean": _round(_mean(clean_p)), "damaged": _round(_mean(dmg_p)),
                     "cleanN": len(clean_p), "damagedN": len(dmg_p)}

    # heavy-damage rate by segment
    seg_tot = [0] * len(SEG); seg_dmg = [0] * len(SEG)
    for i in range(n):
        si = int(s[i])
        if si < 0:
            continue
        seg_tot[si] += 1
        if hd[i] == 1:
            seg_dmg[si] += 1
    damage_by_seg = [
        {"seg": SEG[si], "pct": _fixed(100 * seg_dmg[si] / seg_tot[si], 1) if seg_tot[si] else 0,
         "n": seg_tot[si], "damaged": seg_dmg[si]}
        for si in range(len(SEG)) if seg_tot[si] > 0
    ]

    return {
        "n": n,
        "kpi": {"avgPrice": avg_price, "medianPrice": median_price, "avgAge": avg_age,
                "medianKm": median_km, "cleanPct": clean_pct, "damagedN": damaged_n,
                "priceSpark": price_spark, "kmSpark": km_spark, "sparkYears": spark_years},
        "segmentPrice": segment_price,
        "matrix": {"buckets": [b[2] for b in BUCKETS], "rows": matrix_rows, "colTotals": col_totals,
                   "grandTotal": sum(col_totals)},
        "recent": recent, "dailyVolume": daily_volume, "fuelYear": fuel_year, "fuelDonut": fuel_donut,
        "brandRange": brand_range, "priceByYear": [{"year": r["year"], "price": r["price"]} for r in price_by_year],
        "density": density, "scatter": scatter, "provinces": provinces, "provMax": prov_max,
        "priceHist": price_hist, "bodyBox": body_box, "damageImpact": damage_impact, "damageBySeg": damage_by_seg,
    }


def compute_districts(rows: dict, idx: np.ndarray, ci: int):
    if ci < 0:
        return []
    c = rows["cols"]
    dic = rows["dict"]
    ci_f = c["ci"][idx]; di_f = c["di"][idx]; p_f = c["p"][idx]
    dist_arr = {}
    for k in range(idx.shape[0]):
        if int(ci_f[k]) != ci or int(di_f[k]) < 0:
            continue
        dist_arr.setdefault(int(di_f[k]), []).append(int(p_f[k]))
    out = [{"name": dic["district"][d], "n": len(arr), "median": _round(_median(arr))}
           for d, arr in dist_arr.items()]
    out.sort(key=lambda r: -r["n"])
    return out[:15]


def compute_damage(rows: dict, idx: np.ndarray, dtype: int) -> dict:
    c = rows["cols"]
    dmg_f = c["dmg"][idx]
    parts = [0] * len(PARTS)
    type_totals = [0, 0, 0, 0]
    for k in range(idx.shape[0]):
        dmg = int(dmg_f[k])
        if dmg == 0:
            continue
        for pi in range(len(PARTS)):
            code = _dpart(dmg, pi)
            if code > 0:
                type_totals[code] += 1
            if (code > 0) if dtype == -1 else (code == dtype):
                parts[pi] += 1
    return {"parts": parts, "max": max(1, *parts), "typeTotals": type_totals}


# ── public entry points for the routes ───────────────────────────────────
def get_bi_meta() -> dict:
    r = load_rows()
    return {"meta": r["meta"], "dict": r["dict"]}


def get_bi_agg(filters: dict, dmg_type: int = -1, province: Optional[str] = None) -> dict:
    r = load_rows()
    idx = filter_rows(r, filters)
    agg = compute_agg(r, idx)
    prov_ci = r["dict"]["city"].index(province) if (province and province in r["dict"]["city"]) else -1
    districts = compute_districts(r, idx, prov_ci)
    damage = compute_damage(r, idx, dmg_type)
    return {"agg": agg, "districts": districts, "damage": damage}
