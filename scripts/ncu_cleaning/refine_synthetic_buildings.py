"""
Refine the 20 synthetic NCU buildings (uid 9000000001-20) injected by
locate_unmatched_buildings.py.

For each synthetic feature:
  1. Find ALL named OSM polygons within 200 m of its centroid
  2. Score each candidate via:
       a) name similarity     — rapidfuzz.partial_ratio(synth_name, candidate_name)
       b) distance             — penalty linearly from 200 m
       c) building type        — bonus if both are 'university' / academic
       d) shared CJK fragment  — bonus for ≥ 2 char overlap (科一館 ↔ 科學一館)
  3. If best total score ≥ MERGE_THRESHOLD → MERGE:
       - rename candidate (keep its osm_id) to synth_name (or chained alias)
       - delete the synthetic feature
       - record alias decision in audit
  4. Otherwise keep as synthetic

Outputs:
  buildings.geojson     (in-place; backup buildings.bak3.json)
  outputs/ncu_114/refine_synthetic_audit.csv
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
NCU_GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
BACKUP = NCU_GEOJSON.with_suffix(".bak3.json")
AUDIT_CSV = ROOT / "outputs" / "ncu_114" / "refine_synthetic_audit.csv"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

SEARCH_RADIUS_M = 200
FRAGMENT_MAX_DIST_M = 100   # second-pass merges only allowed if within this distance
MERGE_THRESHOLD = 60        # final composite score 0-160ish
FRAGMENT_THRESHOLD = 3      # second-pass: shared CJK fragment ≥ 3 chars
SYN_UID_MIN = 9_000_000_001
SYN_UID_MAX = 9_999_999_999

CJK_RE = re.compile(r"[一-鿿]+")
# Numeric-quantifier disambiguation: pairs that contain DIFFERENT
# numbered quantifiers (different dorm numbers, different building series numbers)
# must NOT be merged.
NUMBER_QUANT_RE = re.compile(r"([一二三四五六七八九十百千]+|\d+)\s*(舍|館|宿舍|號|期|棟)")


DORM_TOKEN_RE = re.compile(r"([男女])([一二三四五六七八九十百千]+|\d+)?\s*(舍|宿舍)")


def _dorm_tokens(s: str) -> set[str]:
    """Return canonical dorm tokens like '男7' / '女2' / 'X舍-no-number' for any
    bare 舍/宿舍 reference. Used to block dorm cross-merges."""
    out = set()
    for m in DORM_TOKEN_RE.finditer(s or ""):
        gender, num, _suf = m.group(1), m.group(2), m.group(3)
        out.add(f"{gender or ''}{num or 'X'}".strip() or "X")
    # Also handle bare "宿舍"/"舍" without 男/女 prefix (e.g. 國際學生宿舍, 教職員單身一舍)
    if not out and ("宿舍" in (s or "") or (s or "").endswith("舍")):
        out.add("舍")
    return out


def conflicting_numbers(a: str, b: str) -> bool:
    """True if a and b are clearly different dorms / numbered series."""
    # Numeric quantifier conflict (e.g. 工五館 vs 工三館)
    nums_a = NUMBER_QUANT_RE.findall(a or "")
    nums_b = NUMBER_QUANT_RE.findall(b or "")
    if nums_a and nums_b:
        set_a = {f"{n}{q}" for n, q in nums_a}
        set_b = {f"{n}{q}" for n, q in nums_b}
        if set_a.isdisjoint(set_b):
            return True
    # Dorm-specific conflict: 學生男六舍 vs 國際學生宿舍 (different dorm specifier)
    dorm_a = _dorm_tokens(a)
    dorm_b = _dorm_tokens(b)
    if dorm_a and dorm_b and dorm_a.isdisjoint(dorm_b):
        return True
    return False


def latlon_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polygon_centroid(geometry):
    g = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    ring = coords[0] if g == "Polygon" else coords[0][0] if g == "MultiPolygon" else None
    if not ring:
        return None
    n = len(ring)
    return sum(p[1] for p in ring) / n, sum(p[0] for p in ring) / n


def cjk_fragments(s):
    return [m for m in CJK_RE.findall(s or "") if len(m) >= 2]


def shared_substring_score(a, b):
    """Score 0-30 based on longest shared 2+ char CJK substring length."""
    a_frags = cjk_fragments(a)
    b_frags = cjk_fragments(b)
    best = 0
    for af in a_frags:
        for bf in b_frags:
            for sz in range(min(len(af), len(bf)), 1, -1):
                for start in range(0, len(af) - sz + 1):
                    sub = af[start:start + sz]
                    if sub in bf and sz > best:
                        best = sz
    return min(best * 8, 30)


def score_candidate(synth_name, synth_type, cand_name, cand_type, distance_m):
    name_sim = fuzz.partial_ratio(synth_name, cand_name)             # 0-100
    dist_pen = max(0, (SEARCH_RADIUS_M - distance_m)) / SEARCH_RADIUS_M * 30  # 0-30
    type_bonus = 0
    academic_kw = {"university", "school", "library"}
    if (synth_type or "").lower() in academic_kw and (cand_type or "").lower() in academic_kw:
        type_bonus = 15
    cjk_bonus = shared_substring_score(synth_name, cand_name)        # 0-30
    return round(name_sim * 0.5 + dist_pen + type_bonus + cjk_bonus, 1), {
        "name_sim": round(name_sim * 0.5, 1),
        "dist_pen": round(dist_pen, 1),
        "type_bonus": type_bonus,
        "cjk_bonus": cjk_bonus,
    }


def main():
    gj = json.loads(NCU_GEOJSON.read_text(encoding="utf-8"))
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
        print(f"backup: {BACKUP.name}")

    # Index features
    synth_feats = []
    named_feats = []  # (osm_id, name, building_type, cen_lat, cen_lon, feature_ref)
    for ft in gj["features"]:
        p = ft.get("properties", {})
        oid = p.get("osm_id")
        try:
            oid_i = int(oid) if oid is not None else None
        except (TypeError, ValueError):
            oid_i = None
        if oid_i is None:
            continue
        cen = polygon_centroid(ft.get("geometry") or {})
        if cen is None:
            continue
        is_synth = SYN_UID_MIN <= oid_i <= SYN_UID_MAX
        if is_synth:
            synth_feats.append((oid_i, p.get("name", ""), p.get("building_type", ""),
                                cen[0], cen[1], ft))
        elif (p.get("name") or "").strip():
            named_feats.append((oid_i, p["name"], p.get("building_type", ""),
                                cen[0], cen[1], ft))

    print(f"synthetic features: {len(synth_feats)}")
    print(f"named candidates:   {len(named_feats)}")
    print()

    audit_rows = []
    to_remove_uids = set()
    aliases_applied = 0
    for syn_uid, syn_name, syn_type, syn_lat, syn_lon, syn_ft in synth_feats:
        # PASS A: Score every named within search radius (uses Google coord)
        candidates = []
        for cand_uid, cand_name, cand_type, cand_lat, cand_lon, cand_ft in named_feats:
            d = latlon_meters(syn_lat, syn_lon, cand_lat, cand_lon)
            if d > SEARCH_RADIUS_M:
                continue
            if conflicting_numbers(syn_name, cand_name):
                continue   # different dorm number / building series → don't merge
            score, parts = score_candidate(syn_name, syn_type, cand_name, cand_type, d)
            candidates.append((score, d, cand_uid, cand_name, cand_ft, parts))
        candidates.sort(key=lambda x: -x[0])

        # PASS B (fragment-first): if pass A had no winner, search ALL named
        # buildings (no distance limit) for one sharing a ≥3-char CJK fragment
        if not (candidates and candidates[0][0] >= MERGE_THRESHOLD):
            best_frag = None
            for cand_uid, cand_name, cand_type, cand_lat, cand_lon, cand_ft in named_feats:
                if conflicting_numbers(syn_name, cand_name):
                    continue
                # Compute longest shared CJK substring length
                shared_len = max(
                    (sz for af in cjk_fragments(syn_name)
                         for bf in cjk_fragments(cand_name)
                         for sz in range(min(len(af), len(bf)), FRAGMENT_THRESHOLD - 1, -1)
                         if any(af[i:i+sz] in bf for i in range(len(af) - sz + 1))),
                    default=0,
                )
                if shared_len < FRAGMENT_THRESHOLD:
                    continue
                d = latlon_meters(syn_lat, syn_lon, cand_lat, cand_lon)
                if d > FRAGMENT_MAX_DIST_M:
                    continue   # don't allow long-distance fragment merges
                score, parts = score_candidate(syn_name, syn_type, cand_name, cand_type,
                                                min(d, SEARCH_RADIUS_M))
                # Strong fragment bonus
                score += shared_len * 10
                parts["fragment_len"] = shared_len
                parts["dist_m"] = round(d, 1)
                if best_frag is None or score > best_frag[0]:
                    best_frag = (score, d, cand_uid, cand_name, cand_ft, parts)
            if best_frag:
                candidates = [best_frag] + candidates
                candidates.sort(key=lambda x: -x[0])

        if candidates and candidates[0][0] >= MERGE_THRESHOLD:
            best_score, best_d, best_uid, best_name, best_ft, best_parts = candidates[0]
            # MERGE: keep best target, but record alias
            best_props = best_ft.setdefault("properties", {})
            existing_aliases = best_props.get("name_aliases", [])
            if isinstance(existing_aliases, str):
                existing_aliases = [existing_aliases]
            if syn_name not in existing_aliases and syn_name != best_name:
                existing_aliases.append(syn_name)
            best_props["name_aliases"] = existing_aliases
            best_props.setdefault("merged_from", []).append({
                "synthetic_uid": syn_uid,
                "score": best_score,
                "distance_m": round(best_d, 1),
            })
            to_remove_uids.add(syn_uid)
            aliases_applied += 1
            audit_rows.append({
                "synth_uid": syn_uid, "synth_name": syn_name,
                "action": "merged", "target_osm_id": best_uid,
                "target_name": best_name, "score": best_score,
                "distance_m": round(best_d, 1), "score_breakdown": str(best_parts),
            })
            print(f"  MERGE  {syn_name:24s}  →  {best_name:30s}  "
                  f"(uid {best_uid}, score {best_score}, d {best_d:.0f}m)")
        else:
            top_str = (f"top: {candidates[0][3]} score={candidates[0][0]} "
                       f"d={candidates[0][1]:.0f}m" if candidates else "no candidates")
            audit_rows.append({
                "synth_uid": syn_uid, "synth_name": syn_name,
                "action": "kept", "target_osm_id": syn_uid,
                "target_name": syn_name,
                "score": candidates[0][0] if candidates else 0,
                "distance_m": round(candidates[0][1], 1) if candidates else None,
                "score_breakdown": top_str,
            })
            print(f"  KEEP   {syn_name:24s}  ({top_str})")

    # Strip merged synth features
    new_features = []
    for ft in gj["features"]:
        oid = ft.get("properties", {}).get("osm_id")
        try:
            oid_i = int(oid) if oid is not None else None
        except (TypeError, ValueError):
            oid_i = None
        if oid_i in to_remove_uids:
            continue
        new_features.append(ft)
    gj["features"] = new_features

    NCU_GEOJSON.write_text(json.dumps(gj, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"=== Summary ===")
    print(f"  merged into existing OSM polygon: {aliases_applied}")
    print(f"  kept as synthetic:                {len(synth_feats) - aliases_applied}")
    print(f"  buildings.geojson features: {len(gj['features'])}")
    print(f"  audit: {AUDIT_CSV}")


if __name__ == "__main__":
    main()
