#!/usr/bin/env python3
"""
synpuf_generator_v2.py — Generador sintético SynPUF-inspirado (versión 2).

Responsabilidad: producir el registro de eventos, la metadata observable de reclamaciones
y la tabla de verdad de referencia (truth).

SEPARACIÓN DE RESPONSABILIDADES
  - Este módulo contiene las reglas de inyección de fraude.
  - cas_detector_v2.py NO lo importa. El detector solo recibe el registro observable.
  - synpuf_truth.csv se usa únicamente en la evaluación.

CAMBIOS RESPECTO A LA VERSIÓN 1
  1. Las tablas de especialidad del detector se derivan de los datos, no de este módulo.
  2. Parámetro `difficulty` (0-1+) que escala la magnitud de las señales de fraude.
     Permite una prueba de estrés: ¿qué tan bien detecta el sistema cuando la señal se debilita?
  3. Onset tardío de fraude por proveedor: los proveedores con onset generan deriva temporal real,
     necesaria para evaluar el componente CAS-1 y CAS-3.
  4. Lag y hora de servicio de fraude son probabilísticos y escalan con `difficulty`.
  5. La columna de diagnóstico se llama DX_CD_1 (los códigos son ICD-10, no ICD-9).
  6. Ninguna columna del registro observable contiene el perfil ni la etiqueta.

LIMITACIÓN DECLARADA
  El generador no implementa retroalimentación (un agente que cambia su comportamiento
  al detectar el sistema). El componente CAS-4 (feedback) no puede validarse con estos datos.

USO
  python synpuf_generator_v2.py --out-dir ./data --difficulty 1.0 0.5 0.25
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ── Catálogos CMS Medicare Part B (simplificados) ───────────────

CPT_CATALOG = {
    "99211": {"desc": "office_visit_minimal",        "fee": 24,   "complexity": 1, "category": "EM_amb"},
    "99212": {"desc": "office_visit_straightforward", "fee": 56,   "complexity": 1, "category": "EM_amb"},
    "99213": {"desc": "office_visit_low",             "fee": 85,   "complexity": 2, "category": "EM_amb"},
    "99214": {"desc": "office_visit_moderate",        "fee": 130,  "complexity": 3, "category": "EM_amb"},
    "99215": {"desc": "office_visit_high",            "fee": 180,  "complexity": 4, "category": "EM_amb"},
    "99231": {"desc": "hospital_subseq_low",          "fee": 75,   "complexity": 2, "category": "EM_hosp"},
    "99232": {"desc": "hospital_subseq_mod",          "fee": 110,  "complexity": 3, "category": "EM_hosp"},
    "99233": {"desc": "hospital_subseq_high",         "fee": 160,  "complexity": 4, "category": "EM_hosp"},
    "99221": {"desc": "hospital_admit_low",           "fee": 130,  "complexity": 2, "category": "EM_hosp"},
    "99222": {"desc": "hospital_admit_mod",           "fee": 190,  "complexity": 3, "category": "EM_hosp"},
    "99223": {"desc": "hospital_admit_high",          "fee": 260,  "complexity": 4, "category": "EM_hosp"},
    "85025": {"desc": "cbc_differential",             "fee": 11,   "complexity": 1, "category": "lab"},
    "80053": {"desc": "metabolic_panel_comp",         "fee": 14,   "complexity": 1, "category": "lab"},
    "93000": {"desc": "ecg_routine",                  "fee": 20,   "complexity": 1, "category": "diag"},
    "71046": {"desc": "chest_xray_2view",             "fee": 45,   "complexity": 2, "category": "diag"},
    "70553": {"desc": "mri_brain_contrast",           "fee": 520,  "complexity": 4, "category": "diag"},
    "27447": {"desc": "knee_arthroplasty_total",      "fee": 1520, "complexity": 5, "category": "surg"},
    "27130": {"desc": "hip_arthroplasty_total",       "fee": 1380, "complexity": 5, "category": "surg"},
    "43239": {"desc": "upper_gi_endoscopy_biopsy",    "fee": 480,  "complexity": 3, "category": "surg"},
    "66984": {"desc": "cataract_extraction",          "fee": 710,  "complexity": 3, "category": "surg"},
    "99395": {"desc": "preventive_18_39",             "fee": 68,   "complexity": 1, "category": "prev"},
    "99396": {"desc": "preventive_40_64",             "fee": 72,   "complexity": 1, "category": "prev"},
}

# Perfiles de especialidad: SOLO para el generador. El detector no los importa.
SPECIALTY_PROFILES = {
    "medicina_general": {
        "normal_cpts": ["99213", "99214", "85025", "80053", "93000", "99395"],
        "aux_cpts":    ["71046", "99212"],
        "typical_dx":  ["Z00.00", "J06.9", "I10", "E11.9", "M54.5"],
    },
    "medicina_interna": {
        "normal_cpts": ["99213", "99214", "99215", "85025", "80053", "93000"],
        "aux_cpts":    ["71046", "70553", "99221"],
        "typical_dx":  ["I10", "E11.9", "N18.3", "J18.9", "K21.0"],
    },
    "hospitalist": {
        "normal_cpts": ["99221", "99222", "99231", "99232", "85025"],
        "aux_cpts":    ["93000", "71046", "99223"],
        "typical_dx":  ["J18.9", "I50.9", "N17.9", "K57.30", "A41.9"],
    },
    "ortopedia": {
        "normal_cpts": ["99213", "99214", "71046", "27447", "27130"],
        "aux_cpts":    ["99215", "85025"],
        "typical_dx":  ["M17.11", "M16.11", "M51.16", "S72.001A", "M54.4"],
    },
    "oftalmologia": {
        "normal_cpts": ["99213", "66984", "99214"],
        "aux_cpts":    ["85025"],
        "typical_dx":  ["H26.9", "H40.1110", "H33.001", "H35.30"],
    },
    "gastroenterologia": {
        "normal_cpts": ["99213", "99214", "43239", "80053"],
        "aux_cpts":    ["99215", "85025", "71046"],
        "typical_dx":  ["K57.30", "K21.0", "K92.1", "K51.90", "K74.60"],
    },
}

STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]

PROCESS_FLOWS = {
    "ambulatory": ["registro_ingreso", "triage", "evaluacion_medica",
                   "procedimiento_cpt", "indicaciones_alta", "submission_factura"],
    "inpatient":  ["registro_ingreso", "admision_hospitalaria", "evaluacion_medica",
                   "procedimiento_cpt", "seguimiento_clinico", "alta_medica", "submission_factura"],
    "surgical":   ["registro_ingreso", "evaluacion_preoperatoria", "consentimiento_informado",
                   "procedimiento_cpt", "recuperacion_postop", "alta_medica", "submission_factura"],
}

SPECIALTY_TO_FLOW = {
    "medicina_general": "ambulatory",
    "medicina_interna": "ambulatory",
    "hospitalist":      "inpatient",
    "ortopedia":        "surgical",
    "oftalmologia":     "surgical",
    "gastroenterologia": "ambulatory",
}

PROFILE_NAMES = ["normal", "upcoder_leve", "upcoder_severo",
                 "phantom_biller", "unbundler", "mixed_fraud"]


@dataclass
class GenConfig:
    n_providers: int = 800
    n_claims: int = 50_000
    profile_probs: tuple = (0.72, 0.10, 0.07, 0.04, 0.04, 0.03)
    difficulty: float = 1.0          # 1.0 = línea base; 0 = fraude sin magnitud
    late_onset_share: float = 0.5    # proporción de proveedores fraudulentos con onset tardío
    onset_day_range: tuple = (90, 300)
    seed: int = 2024


def _amp(d, m):
    """Escala la magnitud de una señal de fraude. d=1 → m; d=0 → 1 (sin señal de magnitud)."""
    return 1.0 + d * (m - 1.0)


# ── Proveedores ─────────────────────────────────────────────────

def generate_providers(cfg, rng):
    n = cfg.n_providers
    profiles = rng.choice(PROFILE_NAMES, size=n, p=cfg.profile_probs)
    specialties = list(SPECIALTY_PROFILES)
    late = rng.random(n) < cfg.late_onset_share
    onsets = rng.integers(cfg.onset_day_range[0], cfg.onset_day_range[1], size=n)
    rows = []
    for i in range(n):
        prof = str(profiles[i])
        rows.append({
            "AT_PHYSN_NPI":   f"NPI{i + 1:07d}",
            "specialty":      str(rng.choice(specialties)),
            "profile":        prof,
            "onset_day":      int(onsets[i]) if (prof != "normal" and late[i]) else 0,
            "years_active":   int(rng.integers(1, 30)),
            "PRVDR_STATE_CD": str(rng.choice(STATES)),
            "group_practice": bool(rng.random() < 0.6),
            "base_volume":    int(rng.integers(40, 200)),
            "billing_speed":  str(rng.choice(["fast", "normal", "slow"], p=[0.2, 0.6, 0.2])),
        })
    return pd.DataFrame(rows)


# ── Inyección de fraude ─────────────────────────────────────────

def _inject(rng, profile, specialty, cpt_normal, amount_normal, d):
    """Devuelve (cpt, monto, tipo_fraude). tipo 'none' si la reclamación es normal."""
    spec = SPECIALTY_PROFILES[specialty]
    cpt, amount, ftype = cpt_normal, amount_normal * rng.uniform(0.92, 1.08), "none"

    if profile == "upcoder_leve":
        if cpt_normal == "99213" and rng.random() < 0.25:
            cpt = "99214"
            amount = amount_normal * (130 / 85) * _amp(d, rng.uniform(0.95, 1.15))
            ftype = "F5_upcoding_leve"

    elif profile == "upcoder_severo":
        r = rng.random()
        if r < 0.45 and cpt_normal in ("99213", "99212", "99211"):
            cpt = "99215"
            amount = CPT_CATALOG["99215"]["fee"] * _amp(d, rng.uniform(1.1, 2.5))
            ftype = "F1_upcoding_severo"
        elif r < 0.65 and cpt_normal in ("99231", "99221"):
            cpt = "99223"
            amount = CPT_CATALOG["99223"]["fee"] * _amp(d, rng.uniform(1.2, 2.0))
            ftype = "F1_upcoding_severo"

    elif profile == "phantom_biller":
        if rng.random() < 0.55:
            cpt = str(rng.choice(list(CPT_CATALOG)))
            amount = CPT_CATALOG[cpt]["fee"] * _amp(d, rng.uniform(1.0, 1.5))
            ftype = "F2_phantom_billing"

    elif profile == "unbundler":
        if rng.random() < 0.50:
            amount = amount_normal * _amp(d, rng.uniform(1.3, 2.2))
            ftype = "F3_unbundling"

    elif profile == "mixed_fraud":
        r = rng.random()
        if r < 0.30:
            surg = [c for c in spec["aux_cpts"] if CPT_CATALOG.get(c, {}).get("category") == "surg"] or ["27447"]
            cpt = str(rng.choice(surg))
            amount = CPT_CATALOG[cpt]["fee"] * _amp(d, rng.uniform(1.0, 1.4))
            ftype = "F4_upcoding_quirurgico"
        elif r < 0.55 and cpt_normal in ("99213", "99212"):
            cpt = "99215"
            amount = CPT_CATALOG["99215"]["fee"] * _amp(d, rng.uniform(1.1, 1.8))
            ftype = "F1_upcoding_severo"

    return cpt, round(float(amount), 2), ftype


def _submission_lag(rng, profile, billing_speed, is_fraud, d):
    lo, hi = {"fast": (1, 5), "normal": (3, 21), "slow": (10, 45)}[billing_speed]
    if is_fraud and profile in ("phantom_biller", "unbundler") and rng.random() < d:
        return int(rng.integers(0, 3))
    if is_fraud and profile == "mixed_fraud" and rng.random() < d:
        return int(rng.integers(0, 7))
    return int(rng.integers(lo, hi))


def _service_hour(rng, profile, is_fraud, d):
    if is_fraud and profile == "phantom_biller" and rng.random() < 0.35 * d:
        return int(rng.choice([0, 1, 2, 3, 22, 23]))
    return int(rng.integers(7, 19))


# ── Registro de eventos ─────────────────────────────────────────

def generate_event_log(cfg, providers, rng):
    records, meta, truth = [], [], []
    base = datetime(2023, 1, 1)

    vol = providers["base_volume"].to_numpy()
    counts = np.floor(vol / vol.sum() * cfg.n_claims).astype(int)
    counts[: cfg.n_claims - counts.sum()] += 1

    counter = 0
    for idx, prov in providers.iterrows():
        spec = SPECIALTY_PROFILES[prov.specialty]
        flow = PROCESS_FLOWS[SPECIALTY_TO_FLOW[prov.specialty]]

        for _ in range(int(counts[idx])):
            counter += 1
            clm = f"CLM{counter:08d}"
            ben = f"BEN{rng.integers(1, 200_000):07d}"

            cpt_base = str(rng.choice(spec["normal_cpts"]))
            fee_base = CPT_CATALOG.get(cpt_base, {}).get("fee", 85)
            amt_base = fee_base * rng.uniform(0.90, 1.20)

            svc_day = base + timedelta(days=int(rng.integers(0, 364)))
            # Fraude solo tras el onset del proveedor
            active = prov.profile != "normal" and svc_day.timetuple().tm_yday >= prov.onset_day
            if active:
                cpt, amount, ftype = _inject(rng, prov.profile, prov.specialty, cpt_base, amt_base,
                                             cfg.difficulty)
            else:
                cpt, amount, ftype = cpt_base, round(float(amt_base), 2), "none"
            is_fraud = int(ftype != "none")

            svc_hour = _service_hour(rng, prov.profile, is_fraud, cfg.difficulty)
            svc_dt = svc_day.replace(hour=svc_hour, minute=int(rng.integers(0, 59)))
            lag = _submission_lag(rng, prov.profile, prov.billing_speed, is_fraud, cfg.difficulty)
            sub_dt = svc_dt + timedelta(days=lag)

            dx = str(rng.choice(spec["typical_dx"]))
            nat_fee = CPT_CATALOG.get(cpt, {}).get("fee", fee_base)
            paid = round(float(nat_fee * rng.uniform(0.75, 0.85)), 2)

            if ftype == "F2_phantom_billing":
                events = (["procedimiento_cpt", "submission_factura"] if rng.random() < 0.6
                          else ["procedimiento_cpt", "indicaciones_alta", "submission_factura"])
            else:
                events = list(flow)
                if not is_fraud and rng.random() < 0.15:
                    for opt in ("triage", "indicaciones_alta", "seguimiento_clinico"):
                        if opt in events and rng.random() < 0.4:
                            events.remove(opt)

            t = svc_dt
            for i, act in enumerate(events):
                if i > 0:
                    t = t + timedelta(minutes=int(rng.integers(5, 60)))
                if act == "submission_factura":
                    t = sub_dt
                records.append({
                    "CLM_ID":            clm,
                    "DESYNPUF_ID":       ben,
                    "AT_PHYSN_NPI":      prov.AT_PHYSN_NPI,
                    "PRVDR_STATE_CD":    prov.PRVDR_STATE_CD,
                    "DX_CD_1":           dx,
                    "CLM_FROM_DT":       svc_dt.strftime("%Y%m%d"),
                    "NCH_WKLY_PROC_DT":  sub_dt.strftime("%Y%m%d"),
                    "CLM_PMT_AMT":       paid,
                    "case:concept:name": clm,
                    "concept:name":      act,
                    "time:timestamp":    t,
                    "org:resource":      prov.AT_PHYSN_NPI,
                    "HCPCS_CD":          cpt if act == "procedimiento_cpt" else "N/A",
                    "specialty":         prov.specialty,
                    "amount_billed":     round(float(amount), 2),
                    "submission_lag":    lag,
                    "service_hour":      svc_hour,
                    "years_provider":    int(prov.years_active),
                    "group_practice":    int(prov.group_practice),
                    "cpt_complexity":    CPT_CATALOG.get(cpt, {}).get("complexity", 2),
                    "cpt_category":      CPT_CATALOG.get(cpt, {}).get("category", "EM_amb"),
                    "event_index":       i,
                    "flow_length":       len(events),
                })

            meta.append({
                "CLM_ID":           clm,
                "DESYNPUF_ID":      ben,
                "AT_PHYSN_NPI":     prov.AT_PHYSN_NPI,
                "specialty":        prov.specialty,
                "HCPCS_CD":         cpt,
                "DX_CD_1":          dx,
                "CLM_PMT_AMT":      paid,
                "amount_billed":    round(float(amount), 2),
                "CLM_FROM_DT":      svc_dt.strftime("%Y%m%d"),
                "NCH_WKLY_PROC_DT": sub_dt.strftime("%Y%m%d"),
                "submission_lag":   lag,
                "service_hour":     svc_hour,
                "years_provider":   int(prov.years_active),
                "group_practice":   int(prov.group_practice),
                "cpt_complexity":   CPT_CATALOG.get(cpt, {}).get("complexity", 2),
                "cpt_category":     CPT_CATALOG.get(cpt, {}).get("category", "EM_amb"),
                "flow_length":      len(events),
                "PRVDR_STATE_CD":   prov.PRVDR_STATE_CD,
            })

            truth.append({
                "CLM_ID":      clm,
                "AT_PHYSN_NPI": prov.AT_PHYSN_NPI,
                "is_fraud":    is_fraud,
                "fraud_type":  ftype,
                "profile":     prov.profile,
            })

    df_log = pd.DataFrame(records).sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    return df_log, pd.DataFrame(meta), pd.DataFrame(truth)


# ── Orquestación ────────────────────────────────────────────────

def generate(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    providers = generate_providers(cfg, rng)
    df_log, df_meta, df_truth = generate_event_log(cfg, providers, rng)

    df_log.to_csv(os.path.join(out_dir, "synpuf_event_log.csv"), index=False)
    df_meta.to_csv(os.path.join(out_dir, "synpuf_claim_meta.csv"), index=False)
    df_truth.to_csv(os.path.join(out_dir, "synpuf_truth.csv"), index=False)
    with open(os.path.join(out_dir, "gen_config.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    n = len(df_truth)
    n_fraud = int(df_truth.is_fraud.sum())
    print(f"[generador] difficulty={cfg.difficulty:.2f}  claims={n:,}  eventos={len(df_log):,}")
    print(f"[generador] fraude={n_fraud:,} ({100 * n_fraud / n:.2f}%)")
    for ft, c in df_truth.loc[df_truth.is_fraud == 1, "fraud_type"].value_counts().items():
        print(f"    {ft:<26}: {c:>6,}")
    print(f"[generador] salida → {out_dir}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Generador SynPUF-inspirado v2")
    ap.add_argument("--out-dir", default=os.path.join(here, "data"))
    ap.add_argument("--n-claims", type=int, default=50_000)
    ap.add_argument("--n-providers", type=int, default=800)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--difficulty", type=float, nargs="+", default=[1.0],
                    help="Uno o más niveles; cada uno genera un subdirectorio difficulty_X.XX")
    args = ap.parse_args()

    for d in args.difficulty:
        cfg = GenConfig(n_providers=args.n_providers, n_claims=args.n_claims,
                        difficulty=d, seed=args.seed)
        generate(cfg, os.path.join(args.out_dir, f"difficulty_{d:.2f}"))


if __name__ == "__main__":
    main()
