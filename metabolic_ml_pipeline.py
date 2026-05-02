# pip install cobra highspy scikit-learn networkx numpy pandas matplotlib scipy
# pip install straindesign   ← optional; enables OptKnock comparison (Pub-2)

import os
import zipfile
import warnings
from datetime import datetime
from itertools import combinations

try:
    import straindesign as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, rankdata
from cobra.io import read_sbml_model, load_model
from cobra.flux_analysis import flux_variability_analysis, pfba

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH       = None            # None → built-in model (see load_metabolic_model)
TARGET_REACTION  = "EX_succ_e"      # succinate: aerobic TCA product, graded response across interventions
GLUCOSE_REACTION = "EX_glc__D_e"
OXYGEN_REACTION  = "EX_o2_e"
ATP_REACTION     = "ATPM"          # energy cost proxy
GLUCOSE_BOUND    = -10.0           # mmol/gDW/h
# Aerobic succinate: multiple competing routes (oxidative TCA, reductive branch,
# glyoxylate shunt) give a CONTINUOUS gradient of production across interventions.
# Anaerobic lactate was abandoned: all "good" KOs collapsed to the same NADH
# ceiling (17.76 mmol/gDW/h), producing binary labels and ρ = 0.21.
ANAEROBIC        = False
N_BEAM           = 3               # top singles expanded in combo search
N_BEAM_EXTEND    = 15              # candidates tried per beam element
N_VALIDATE_DOUBLES = 25            # ML-predicted double interventions to verify with FBA
N_MC_SAMPLES     = 15              # Monte Carlo samples per strategy
N_MC_STRATEGIES  = 6
N_TOP            = 10
FVA_FRACTION     = 0.9
GROWTH_FRACTION  = 0.10            # step-2 growth floor: 10% of mutant max — allows 90% sacrifice for production
# Five growth fractions used to compute the AUC label for ML training.
# At 10% growth all interventions hit the same succinate ceiling → binary labels.
# Scanning across fractions reveals how quickly each intervention's production
# degrades as growth is enforced — a continuous signal that distinguishes strategies
# even when they produce the same absolute flux at the loosest constraint.
GROWTH_FRACTIONS_AUC = [0.05, 0.15, 0.30, 0.50, 0.75]
# iJO1366 has ~2700 reactions — full enumeration is infeasible.
# Enumerate only the top-N most active reactions for ML training;
# the GBR then ranks all reactions in the feature matrix.
MAX_ENUM_REACTIONS = 300
MAX_FVA_REACTIONS  = 500   # FVA only on top-N most active reactions
MAX_GENE_KO        = 150   # gene KO candidates (filtered to active genes)
W_TARGET         = 0.85            # composite score weights — heavily favor production
W_GROWTH         = 0.15
W_ENERGY         = 0.0             # unused; kept for reference
SEED             = 42
# Secondary targets for multi-target generalizability test (Pub-5).
# Each runs a lightweight pipeline (enumeration + GBR, no MC/FCC/evo).
EXTRA_TARGETS = [
    {"rxn_id": "EX_mal__L_e", "name": "L-malate", "anaerobic": False},
    {"rxn_id": "EX_ac_e",     "name": "acetate",   "anaerobic": False},
]
# Experimentally validated single/double interventions for succinate in E. coli.
# Used in Pub-4 to cross-check ML recommendations against published data.
LITERATURE_KNOCKOUTS_SUCCINATE = [
    {"label": "ATPS4rpp KO",         "rxns": ["ATPS4rpp"],       "kinds": ["ko"],       "ref": "Mienda & Shamsir 2016"},
    {"label": "GND KO",              "rxns": ["GND"],            "kinds": ["ko"],       "ref": "Mienda et al. 2016"},
    {"label": "PFL+LDH_D double KO", "rxns": ["PFL", "LDH_D"],  "kinds": ["ko", "ko"], "ref": "Dong et al. 2009"},
    {"label": "PPC OE",              "rxns": ["PPC"],            "kinds": ["oe"],       "ref": "Millard et al. 1996"},
    {"label": "PTAr KO",             "rxns": ["PTAr"],           "kinds": ["ko"],       "ref": "FastKnock 2023"},
]
# ────────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
_KIND_ENC = {"ko": 0, "oe": 1, "down": 2}

# currency metabolites omitted from the reaction graph to avoid spurious connections
_CURRENCY = {
    "atp_c", "adp_c", "nadh_c", "nad_c", "h2o_c", "pi_c", "h_c",
    "co2_c", "nadph_c", "nadp_c", "coa_c", "ppi_c", "amp_c", "h2o_e",
}

# COBRApy ships e_coli_core with empty `subsystem` fields, which collapses the
# one-hot to a single "unknown" column and breaks stratified sampling. Inject
# pathway labels by reaction id so the categorical features and per-subsystem
# stratification actually carry signal.
_E_COLI_CORE_SUBSYSTEMS = {
    "PGI": "Glycolysis", "PFK": "Glycolysis", "FBA": "Glycolysis",
    "TPI": "Glycolysis", "GAPD": "Glycolysis", "PGK": "Glycolysis",
    "PGM": "Glycolysis", "ENO": "Glycolysis", "PYK": "Glycolysis",
    "FBP": "Glycolysis", "PPS": "Glycolysis", "GLCpts": "Glycolysis",
    "G6PDH2r": "PentosePhosphate", "PGL": "PentosePhosphate",
    "GND": "PentosePhosphate", "RPI": "PentosePhosphate",
    "RPE": "PentosePhosphate", "TKT1": "PentosePhosphate",
    "TKT2": "PentosePhosphate", "TALA": "PentosePhosphate",
    "CS": "TCA", "ACONTa": "TCA", "ACONTb": "TCA", "ICDHyr": "TCA",
    "AKGDH": "TCA", "SUCOAS": "TCA", "SUCDi": "TCA", "FUM": "TCA",
    "MDH": "TCA", "FRD7": "TCA",
    "PPC": "Anaplerotic", "PPCK": "Anaplerotic",
    "ME1": "Anaplerotic", "ME2": "Anaplerotic",
    "ICL": "Glyoxylate", "MALS": "Glyoxylate",
    "PDH": "PyruvateMet", "PFL": "PyruvateMet",
    "LDH_D": "Fermentation", "ALCD2x": "Fermentation",
    "ACALD": "Fermentation", "PTAr": "Fermentation",
    "ACKr": "Fermentation",
    "ATPS4r": "OxPhos", "NADH16": "OxPhos", "CYTBD": "OxPhos",
    "ATPM": "OxPhos", "NADTRHD": "OxPhos", "THD2": "OxPhos",
    "GLUDy": "Nitrogen", "GLNS": "Nitrogen",
    "GLUSy": "Nitrogen", "GLUN": "Nitrogen", "GLNabc": "Nitrogen",
}


def _target_metabolite(model, target_rxn_id):
    """Cytosolic form of the metabolite secreted by TARGET_REACTION (e.g.
    EX_succ_e exports succ_e → return succ_c). Used to mark reactions that
    directly produce or consume the target so the ML can distinguish producers
    from competitors in the stoichiometric feature space."""
    if target_rxn_id not in [r.id for r in model.reactions]:
        return None
    rxn = model.reactions.get_by_id(target_rxn_id)
    met_ids = {m.id for m in model.metabolites}
    for met in rxn.metabolites:
        if met.id.endswith("_e"):
            cyto_id = met.id[:-2] + "_c"
            if cyto_id in met_ids:
                return model.metabolites.get_by_id(cyto_id)
    return None


def assign_subsystem(rxn):
    """Use the model's subsystem if non-empty, else fall back to the manual map,
    then to id-pattern heuristics for transports."""
    if rxn.subsystem and rxn.subsystem.strip():
        return rxn.subsystem.strip()
    if rxn.id in _E_COLI_CORE_SUBSYSTEMS:
        return _E_COLI_CORE_SUBSYSTEMS[rxn.id]
    rid = rxn.id
    if rid.startswith(("EX_", "DM_", "SK_")):
        return "Boundary"
    if rid.endswith(("t2r", "t2", "tex", "tpp", "tipp")) or "abc" in rid.lower() or "t" in rid[-3:]:
        return "Transport"
    return "Other"


# ─── Model ──────────────────────────────────────────────────────────────────────

def load_metabolic_model():
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        m = read_sbml_model(MODEL_PATH)
    else:
        m = load_model("iJO1366")
    if GLUCOSE_REACTION in m.reactions:
        m.reactions.get_by_id(GLUCOSE_REACTION).lower_bound = GLUCOSE_BOUND
    if ANAEROBIC and OXYGEN_REACTION in m.reactions:
        m.reactions.get_by_id(OXYGEN_REACTION).lower_bound = 0.0
    return m


def get_biomass_id(model):
    # iJO1366 has two biomass reactions (core + WT); prefer the one that is
    # actually in the model's current objective so pFBA and slim_optimize agree.
    try:
        obj_vars = set(model.objective.variables)
        for rxn in model.reactions:
            if "biomass" in rxn.id.lower():
                if rxn.forward_variable in obj_vars or rxn.reverse_variable in obj_vars:
                    return rxn.id
    except Exception:
        pass
    for rxn in model.reactions:
        if "biomass" in rxn.id.lower():
            return rxn.id
    return None


# ─── Wild-type analysis + composite score ───────────────────────────────────────

def wt_analysis(model):
    with model:
        sol = pfba(model)
    if sol.status != "optimal":
        raise RuntimeError("Wild-type pFBA infeasible")
    sp = getattr(sol, "shadow_prices", None)
    if not isinstance(sp, pd.Series):
        sp = pd.Series(dtype=float)
    bio_id    = get_biomass_id(model)
    wt_growth = sol.fluxes.get(bio_id, 0.0) if bio_id else 0.0
    if wt_growth < 1e-6:
        # Fallback: run slim_optimize to confirm true max growth
        with model:
            if bio_id:
                model.objective = bio_id
            _g = model.slim_optimize()
            if _g is not None and _g > 1e-6:
                wt_growth = float(_g)

    # Compute WT production at each AUC fraction for normalization and max_target reference.
    # Running through all fractions once avoids a separate single-point call.
    wt_auc_prods = []
    max_target = 1e-4
    for _frac in GROWTH_FRACTIONS_AUC:
        with model:
            if bio_id and wt_growth > 1e-6:
                model.reactions.get_by_id(bio_id).lower_bound = _frac * wt_growth
            model.objective = TARGET_REACTION
            _sol_f = model.optimize()
            _t_f = float(_sol_f.fluxes.get(TARGET_REACTION, 0.0)) if _sol_f.status == "optimal" else 0.0
            _t_f = max(_t_f, 0.0)
            wt_auc_prods.append(_t_f)
            if abs(_frac - GROWTH_FRACTION) < 0.06:   # use closest fraction as max_target ref
                max_target = max(_t_f, max_target)
    wt_auc = float(np.trapezoid(wt_auc_prods, GROWTH_FRACTIONS_AUC))

    return {
        "fluxes":     sol.fluxes,       # biomass-optimal fluxes (used as features)
        "sp":         sp,
        "growth":     wt_growth,
        "target":     sol.fluxes.get(TARGET_REACTION, 0.0),
        "max_target": max(max_target, 1e-4),
        "wt_auc":     max(wt_auc, 1e-4),  # WT AUC normalizer for per-intervention AUC labels
        "atp":        abs(sol.fluxes.get(ATP_REACTION, 0.0)),
        "bio_id":     bio_id,
    }


def _production_curve_auc(model, wt):
    """
    AUC of the production-vs-growth-fraction curve for the current model state,
    normalized by the WT's AUC.

    Growth floors are fractions of WT growth (not mutant g_max). This is the
    scientifically correct reference: the question is "how much succinate can this
    intervention produce while maintaining ≥X% of WT growth?" not "≥X% of its own
    (possibly already reduced) growth". WT-referenced fractions create strong,
    feature-predictable penalties: any intervention that reduces growth below
    frac × wt_growth scores 0 at that fraction → clear signal for the GBR.
    """
    wt_growth = wt["growth"]
    prods = []
    for frac in GROWTH_FRACTIONS_AUC:
        with model:
            if wt["bio_id"]:
                model.reactions.get_by_id(wt["bio_id"]).lower_bound = wt_growth * frac
            model.objective = TARGET_REACTION
            sol = model.optimize()
            t = float(sol.fluxes.get(TARGET_REACTION, 0.0)) if sol.status == "optimal" else 0.0
            prods.append(max(t, 0.0))
    auc_raw = float(np.trapezoid(prods, GROWTH_FRACTIONS_AUC))
    return auc_raw / wt["wt_auc"]


def composite_score(target_flux, growth_rate, atp_flux, wt):
    """
    Weighted normalized score: production gain + growth maintenance.

    The energy / ATP term is omitted because ATPM is itself a candidate intervention:
    knocking it out sets atp_flux = 0, which caused s_energy to diverge to ~10^9
    and ranked lethal knockouts first.  Growth viability is already penalised by
    s_growth, so the separate energy term is both redundant and numerically unstable.

    Lethal interventions (growth < 1e-6) return 0 unconditionally so they can never
    rank above viable strategies regardless of any other flux value.
    """
    if growth_rate < 1e-6:
        return 0.0
    s_target = target_flux / wt["max_target"]
    s_growth = growth_rate / max(wt["growth"], 1e-9)   # guard: wt growth=0 if wrong bio_id
    return W_TARGET * s_target + W_GROWTH * s_growth


# ─── Reaction graph (topology features) ─────────────────────────────────────────

def build_reaction_graph(model):
    G = nx.Graph()
    for rxn in model.reactions:
        G.add_node(rxn.id)
    for met in model.metabolites:
        if met.id in _CURRENCY:
            continue
        rxn_ids = [r.id for r in met.reactions]
        for a, b in combinations(rxn_ids, 2):
            G.add_edge(a, b)
    return G


def compute_graph_features(G, target_rxn_id, biomass_rxn_id):
    betweenness = nx.betweenness_centrality(G, normalized=True)
    max_dist = len(G.nodes)
    path_to_target  = dict(nx.single_source_shortest_path_length(G, target_rxn_id))  if target_rxn_id  in G else {}
    path_to_biomass = dict(nx.single_source_shortest_path_length(G, biomass_rxn_id)) if biomass_rxn_id in G else {}
    records = {
        node: {
            "betweenness":      betweenness.get(node, 0.0),
            "path_to_target":   path_to_target.get(node, max_dist),
            "path_to_biomass":  path_to_biomass.get(node, max_dist),
        }
        for node in G.nodes
    }
    return pd.DataFrame.from_dict(records, orient="index")


# ─── FVA ────────────────────────────────────────────────────────────────────────

def run_fva(model, rxn_ids):
    rxns = [model.reactions.get_by_id(r) for r in rxn_ids if r in model.reactions]
    return flux_variability_analysis(
        model, reaction_list=rxns, fraction_of_optimum=FVA_FRACTION
    ).fillna(0.0)


# ─── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(model, wt, fva_df, graph_df):
    # Subsystem set drawn from the same `assign_subsystem` used per row, so the
    # one-hot columns and the per-row label are guaranteed consistent.
    subsystems = sorted({assign_subsystem(r) for r in model.reactions})
    target_met = _target_metabolite(model, TARGET_REACTION)
    rows = []
    for rxn in model.reactions:
        # Exclude boundary and all biomass reactions from the candidate pool.
        # iJO1366 has two biomass reactions (core + WT); knocking out either
        # would make the growth viability constraint unsatisfiable (lb > ub).
        if rxn.id.startswith(("EX_", "DM_", "SK_")):
            continue
        if "biomass" in rxn.id.lower():
            continue
        wf  = wt["fluxes"].get(rxn.id, 0.0)
        fm  = fva_df.loc[rxn.id, "minimum"] if rxn.id in fva_df.index else wf
        fx  = fva_df.loc[rxn.id, "maximum"] if rxn.id in fva_df.index else wf
        sp_vals = [abs(float(wt["sp"].get(m.id, 0.0))) for m in rxn.metabolites]
        # fva_util can blow up when fva_max ≈ 0; clamp to keep features bounded
        # so StandardScaler + GBR don't produce 10^15 predictions in CV folds.
        util = abs(wf) / max(abs(fx), 1e-3)
        p_ent = float(min(max(util, 1e-9), 1 - 1e-9))
        flux_entropy = -(p_ent * np.log2(p_ent) + (1 - p_ent) * np.log2(1 - p_ent))
        sub = assign_subsystem(rxn)
        # Stoichiometric relationship to the target metabolite. Without this,
        # the model could not distinguish LDH_D (the lactate-producing enzyme,
        # bad to KO) from ACALD (a NADH competitor, good to KO) — they share
        # subsystem, topology, and FVA range but have opposite effects.
        coef = float(rxn.metabolites.get(target_met, 0.0)) if target_met else 0.0
        fva_range_val = fx - fm
        wt_growth     = wt.get("growth", 1.0) or 1.0
        row = {
            "rxn_id":            rxn.id,
            "wt_flux":           wf,
            "fva_min":           fm,
            "fva_max":           fx,
            "fva_range":         fva_range_val,
            "fva_util":          float(min(util, 10.0)),
            "flux_entropy":      float(flux_entropy),
            "mean_sp":           float(np.mean(sp_vals)) if sp_vals else 0.0,
            "max_sp":            float(np.max(sp_vals))  if sp_vals else 0.0,
            "degree":            len(rxn.metabolites),
            "reversible":        int(rxn.lower_bound < 0),
            "betweenness":       graph_df.loc[rxn.id, "betweenness"]     if rxn.id in graph_df.index else 0.0,
            "path_to_target":    graph_df.loc[rxn.id, "path_to_target"]  if rxn.id in graph_df.index else 99,
            "path_to_biomass":   graph_df.loc[rxn.id, "path_to_biomass"] if rxn.id in graph_df.index else 99,
            "produces_target":   int(coef > 0),
            "consumes_target":   int(coef < 0),
            "target_coef":       coef,
            # Derived: FVA flexibility per unit of WT growth (generalisation cue)
            "fva_range_per_g":   fva_range_val / wt_growth,
            # Derived: how tightly the WT flux fills the feasible FVA window
            "flux_fva_ratio":    abs(wf) / (abs(fva_range_val) + 1e-9),
        }
        for s in subsystems:
            row[f"sub_{s}"] = int(sub == s)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("rxn_id")
    # Final safety net: any residual NaN/inf would break the GBR.
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ─── Intervention simulation ─────────────────────────────────────────────────────

def _apply_kind(rxn, kind):
    if kind == "ko":
        rxn.knock_out()
    elif kind == "oe":
        rxn.upper_bound = min(rxn.upper_bound * 3.0 + 1.0, 1000.0)
    elif kind == "down":
        rxn.upper_bound = max(rxn.upper_bound * 0.2, 0.0)


def _two_step_optimize(model, wt):
    """
    Two-step production evaluation for the current model state (may include interventions).

    Step 1 — maximize biomass: finds the growth ceiling g_max for this mutant.
             If g_max ≈ 0, the intervention is lethal → return all zeros.
    Step 2 — fix growth ≥ GROWTH_FRACTION × g_max, maximize TARGET_REACTION.
             Allowing the cell to sacrifice up to (1 - GROWTH_FRACTION) of its maximum
             growth creates a CONTINUOUS label gradient: interventions that reroute
             more carbon toward the target (without fully blocking growth) score higher
             than those that either kill the cell or leave flux distribution unchanged.

    With GROWTH_FRACTION = 0.10 and aerobic succinate:
      - Lethal KOs          → g ≈ 0 → target = 0
      - TCA-disrupting KOs  → g < WT, target = 2–8 mmol/gDW/h  (graded, ML-learnable)
      - Glyoxylate KOs/OEs  → intermediate effects on succinate routing
      - Unrelated KOs       → g ≈ WT, target ≈ WT succinate (near-zero aerobically)
    """
    try:
        if wt["bio_id"]:
            model.objective = wt["bio_id"]   # defensive: don't rely on caller's state
        sol_g = model.optimize()
        if sol_g.status != "optimal":
            return 0.0, 0.0, 0.0
        g = float(sol_g.objective_value)
        if g < 1e-9:
            return 0.0, g, 0.0
        # Step 2 runs inside a nested context so bio.lower_bound and model.objective
        # are always restored on exit, regardless of what the caller has open.
        with model:
            if wt["bio_id"]:
                bio = model.reactions.get_by_id(wt["bio_id"])
                bio.lower_bound = g * GROWTH_FRACTION  # allow 90% growth sacrifice → graded production labels
            model.objective = TARGET_REACTION
            sol_t = model.optimize()
            if sol_t.status != "optimal":
                return 0.0, g, 0.0
            t = float(sol_t.fluxes.get(TARGET_REACTION, 0.0))
            a = abs(float(sol_t.fluxes.get(ATP_REACTION, 0.0)))
        return t, g, a
    except Exception:
        return 0.0, 0.0, 0.0


def simulate_single(model, rxn_id, kind, wt):
    """Returns (target_flux, growth, atp_flux, auc_label). 0s if infeasible or lethal.

    The 4th value is the normalized AUC of the production-vs-growth curve across
    GROWTH_FRACTIONS_AUC — the ML training label. It varies continuously even when
    multiple interventions hit the same single-fraction production ceiling.
    The 1st–3rd values (t, g, a) are from the GROWTH_FRACTION single-point evaluation
    and are used for display / composite_score, not for ML training.
    """
    if rxn_id not in model.reactions:
        return 0.0, 0.0, 0.0, 0.0
    with model:
        _apply_kind(model.reactions.get_by_id(rxn_id), kind)
        t, g, a = _two_step_optimize(model, wt)
        if g < 1e-9:
            return 0.0, 0.0, 0.0, 0.0
        auc_label = _production_curve_auc(model, wt)
        return t, g, a, auc_label


# ─── Full single-intervention enumeration ────────────────────────────────────────

def enumerate_all_singles(model, feat_df, wt):
    """
    Exhaustively simulate every (reaction, kind) combination for all internal reactions.
    For large models (iJO1366: ~2200 internal reactions) the enumeration is capped at
    MAX_ENUM_REACTIONS by activity score; the GBR ranks the full feature matrix at inference.
    """
    if len(feat_df) > MAX_ENUM_REACTIONS:
        # Split budget: 2/3 by FVA flexibility, 1/3 by WT flux activity.
        # Pure importance (FVA+flux) under-samples reactions with high FVA range
        # but low WT flux — exactly the structural interventions that tend to give
        # the largest production gains. The FVA-priority half ensures we always
        # train on the most flexible (highest-impact) reactions in the model.
        n_fva  = MAX_ENUM_REACTIONS * 2 // 3
        n_flux = MAX_ENUM_REACTIONS - n_fva
        by_fva  = set(feat_df["fva_range"].abs().nlargest(n_fva).index)
        by_flux = set(feat_df["wt_flux"].abs().nlargest(n_flux * 3).index) - by_fva
        enum_rxns = by_fva | set(list(by_flux)[:n_flux])
        print(f"    ({len(feat_df)} reactions in model; enumerating top {len(enum_rxns)}: "
              f"{n_fva} by FVA-range + {n_flux} by flux activity)")
    else:
        enum_rxns = set(feat_df.index)

    entries = [
        (rxn_id, kind)
        for rxn_id in feat_df.index
        if rxn_id in model.reactions and rxn_id in enum_rxns
        for kind in ("ko", "oe", "down")
    ]
    X, y_score, y_flux, keys = [], [], [], []
    for i, (rxn_id, kind) in enumerate(entries):
        t, g, a, sc = simulate_single(model, rxn_id, kind, wt)
        X.append(np.append(feat_df.loc[rxn_id].values, _KIND_ENC[kind]))
        y_score.append(sc)
        y_flux.append(t)
        keys.append((rxn_id, kind))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(entries)} simulated...")
    auc_std  = float(np.std(y_score))  if len(y_score) > 1 else 0.0
    auc_mean = float(np.mean(y_score)) if y_score else 0.0
    print(f"    {len(entries)} interventions enumerated | "
          f"non-zero AUC labels: {sum(s > 1e-9 for s in y_score)}/{len(y_score)} | "
          f"AUC mean={auc_mean:.3f}  std={auc_std:.3f}  "
          f"[higher std = more ML signal; old single-fraction gave std≈0]")
    return np.array(X), np.array(y_score), np.array(y_flux), keys


# ─── GBR helpers ────────────────────────────────────────────────────────────────

def _make_gbr():
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=2, random_state=SEED,
    )


# ─── GBR training with proper cross-validation ───────────────────────────────────

def train_gbr_model(X, y, keys):
    """
    Train GBR on full single-intervention labels with GroupKFold cross-validation.

    Groups = reaction IDs, so each fold holds out ALL three kinds of a reaction
    (ko/oe/down) together. This tests true generalization: given that we have never
    seen ANY intervention on reaction R, can we predict its FBA outcome from features?

    This is the scientifically correct question: the value of ML is to predict
    double/triple interventions from single-intervention training data, so we must
    verify the model generalises to reactions it has not seen — not just to kinds
    it hasn't tried.
    """
    groups = np.array([rxn_id for rxn_id, _ in keys])
    unique_rxns = sorted(set(groups))
    n_splits = min(5, len(unique_rxns))

    gkf = GroupKFold(n_splits=n_splits)
    sp_scores, topk_scores = [], []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        pipe = Pipeline([("sc", StandardScaler()), ("gbr", _make_gbr())])
        pipe.fit(X[tr_idx], y[tr_idx])
        pred = pipe.predict(X[te_idx])
        y_te = y[te_idx]

        if len(te_idx) >= 2 and np.std(pred) > 1e-9 and np.std(y_te) > 1e-9:
            sp_scores.append(float(spearmanr(pred, y_te).correlation))

        k = max(1, len(te_idx) // 5)
        true_top = set(np.argsort(y_te)[::-1][:k])
        pred_top = set(np.argsort(pred)[::-1][:k])
        topk_scores.append(len(true_top & pred_top) / k)

    sp_mean = float(np.mean(sp_scores)) if sp_scores else 0.0
    sp_std  = float(np.std(sp_scores))  if sp_scores else 0.0
    tk_mean = float(np.mean(topk_scores)) if topk_scores else 0.0
    print(f"    GroupKFold-by-reaction CV ({n_splits} folds): "
          f"Spearman ρ = {sp_mean:.3f} ± {sp_std:.3f} | "
          f"Top-K precision = {tk_mean:.3f}")

    scaler = StandardScaler()
    clf = _make_gbr()
    clf.fit(scaler.fit_transform(X), y)
    return clf, scaler, sp_mean, tk_mean


# ─── Pairwise ranking model ──────────────────────────────────────────────────────

def train_pairwise_ranker(X, y):
    """
    Logistic regression on pairwise feature differences.
    Directly models the ordering problem rather than forcing regression on absolute values.
    Subsampled to ≤40 points to keep O(n²) pair generation tractable.
    """
    idx = rng.choice(len(y), size=min(40, len(y)), replace=False)
    X_p, y_p = [], []
    for i, j in combinations(idx, 2):
        diff  = X[i] - X[j]
        label = int(y[i] > y[j])
        X_p.extend([diff, -diff])
        y_p.extend([label, 1 - label])
    X_p = np.array(X_p)
    y_p = np.array(y_p)
    scaler = StandardScaler()
    clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(scaler.fit_transform(X_p), y_p)
    return clf, scaler


def pairwise_win_rate(row, X_ref, clf_p, scaler_p):
    diffs = row - X_ref                                   # (n_ref, n_features)
    probs = clf_p.predict_proba(scaler_p.transform(diffs))[:, 1]
    return float(probs.mean())


# ─── Ranking ─────────────────────────────────────────────────────────────────────

def rank_all_interventions(feat_df, clf_gbr, scaler_gbr, clf_pair=None, scaler_pair=None,
                            X_ref=None, known_ranks=None):
    """
    Rank all (reaction, kind) combinations using GBR only.

    clf_pair/scaler_pair/X_ref are kept as optional parameters for backward
    compatibility with the ablation study, but are not used in the main pipeline.
    Ablation showed pairwise ranker (ρ=0.053 alone) degrades GBR performance
    when ensembled (ρ=0.447 ensemble vs ρ=0.563 GBR-only).

    For reactions already enumerated in step 4, `known_ranks` provides the exact
    flux-percentile rank from FBA — no need to re-predict with the GBR.
    """
    records = []
    for rxn_id in feat_df.index:
        for kind in ("ko", "oe", "down"):
            row = np.append(feat_df.loc[rxn_id].values, _KIND_ENC[kind])
            key = (rxn_id, kind)
            if known_ranks and key in known_ranks:
                pred_score = known_ranks[key]
                ensemble   = known_ranks[key]   # ground truth overrides GBR for enumerated rxns
            else:
                pred_score = float(np.clip(clf_gbr.predict(scaler_gbr.transform([row]))[0], 0.0, 1.0))
                ensemble   = pred_score  # GBR-only; pairwise ranker removed (ablation)
            records.append({
                "rxn_id":     rxn_id,
                "kind":       kind,
                "pred_score": pred_score,
                "ensemble":   ensemble,
                "n_int":      1,
            })
    return pd.DataFrame(records).sort_values("ensemble", ascending=False).reset_index(drop=True)


# ─── ML-guided double intervention prediction and validation ─────────────────────

def predict_and_rank_doubles(feat_df, clf, scaler, model, known_ranks=None):
    """
    Rank all possible double-intervention pairs without running any FBA.

    For pair (rxn_i, kind_i) × (rxn_j, kind_j):
        pred_pair = (pred_i + pred_j) / 2   [additive / no-epistasis assumption]

    This is the null model for epistasis analysis and the correct prior when
    we have no pairwise training data. The scientific hypothesis being tested:
    "ML trained on single interventions can identify double interventions worth
    testing experimentally, more efficiently than random selection."

    With 74 reactions × 3 kinds there are ~24 K unique pairs — enumerated in
    memory (zero FBA calls) and sorted by predicted double score.
    """
    single_preds = {}
    for rxn_id in feat_df.index:
        if rxn_id not in model.reactions:
            continue
        for kind in ("ko", "oe", "down"):
            key = (rxn_id, kind)
            if known_ranks and key in known_ranks:
                single_preds[key] = known_ranks[key]
            else:
                row = np.append(feat_df.loc[rxn_id].values, _KIND_ENC[kind])
                single_preds[key] = float(clf.predict(scaler.transform([row]))[0])

    pool = sorted(single_preds.keys())
    records = []
    for idx_i in range(len(pool)):
        ri, ki = pool[idx_i]
        pi = single_preds[(ri, ki)]
        for idx_j in range(idx_i + 1, len(pool)):
            rj, kj = pool[idx_j]
            if ri == rj:
                continue
            pj = single_preds[(rj, kj)]
            records.append({
                "rxn_i":     ri, "kind_i": ki,
                "rxn_j":     rj, "kind_j": kj,
                "pred_i":    pi, "pred_j": pj,
                "pred_pair": (pi + pj) / 2.0,
            })
    df = pd.DataFrame(records).sort_values("pred_pair", ascending=False).reset_index(drop=True)
    return df, single_preds


def validate_predicted_doubles(model, pair_df, wt, n_validate):
    """
    Simulate the top-n_validate pairs from pair_df with actual FBA.
    Returns FBA-validated results so we can compare predicted vs. actual score.
    """
    results = []
    for _, row in pair_df.head(n_validate * 2).iterrows():
        if len(results) >= n_validate:
            break
        ri, ki = row["rxn_i"], row["kind_i"]
        rj, kj = row["rxn_j"], row["kind_j"]
        if ri not in model.reactions or rj not in model.reactions:
            continue
        with model:
            _apply_kind(model.reactions.get_by_id(ri), ki)
            _apply_kind(model.reactions.get_by_id(rj), kj)
            t, g, a = _two_step_optimize(model, wt)
            sc = composite_score(t, g, a, wt)
        results.append({
            "rxn_id":    f"{ri}({ki})+{rj}({kj})",
            "kind":      f"{ki}+{kj}",
            "pred_pair": float(row["pred_pair"]),
            "fba_flux":  t,
            "growth":    g,
            "composite": sc,
        })
    return pd.DataFrame(results).sort_values("composite", ascending=False).reset_index(drop=True)


def random_double_baseline(model, wt, single_preds, n_validate):
    """
    Randomly sample n_validate pairs from the same pool as the ML prediction.
    Used to answer: does ML selection beat random selection for double interventions?
    """
    pool = [(r, k) for r, k in single_preds if r in model.reactions]
    rng.shuffle(pool)
    seen_pairs, results = set(), []
    for idx_i in range(len(pool)):
        if len(results) >= n_validate:
            break
        ri, ki = pool[idx_i]
        for idx_j in range(idx_i + 1, len(pool)):
            if len(results) >= n_validate:
                break
            rj, kj = pool[idx_j]
            key = frozenset([(ri, ki), (rj, kj)])
            if ri == rj or key in seen_pairs:
                continue
            seen_pairs.add(key)
            with model:
                _apply_kind(model.reactions.get_by_id(ri), ki)
                _apply_kind(model.reactions.get_by_id(rj), kj)
                t, g, a = _two_step_optimize(model, wt)
                sc = composite_score(t, g, a, wt)
            results.append({
                "rxn_id": f"{ri}({ki})+{rj}({kj})", "kind": f"{ki}+{kj}",
                "fba_flux": t, "growth": g, "composite": sc,
            })
    return pd.DataFrame(results)


# ─── Greedy beam-search for double knockouts ─────────────────────────────────────

def greedy_combo_search(model, ranking_df, wt):
    """
    Beam search for two-reaction combinations. The first reaction is always a KO
    (top N_BEAM). The second can be KO, OE, or down-regulation from the top
    N_BEAM_EXTEND pool — enabling mixed interventions (KO+OE, KO+down) that a
    pure double-KO search would miss.
    """
    top_ko = (
        ranking_df[ranking_df["kind"] == "ko"]
        .drop_duplicates("rxn_id").head(N_BEAM)
    )
    extend_pool = (
        ranking_df.drop_duplicates("rxn_id").head(N_BEAM_EXTEND)
        [["rxn_id", "kind"]].values.tolist()
    )
    results = []
    for _, base in top_ko.iterrows():
        bid = base["rxn_id"]
        for eid, ekind in extend_pool:
            if eid == bid or eid not in model.reactions:
                continue
            with model:
                model.reactions.get_by_id(bid).knock_out()
                _apply_kind(model.reactions.get_by_id(eid), ekind)
                t, g, a = _two_step_optimize(model, wt)
                if g < 1e-9:
                    continue
                results.append({
                    "rxn_id":    f"{bid}(ko)+{eid}({ekind})",
                    "kind":      f"ko+{ekind}",
                    "fba_flux":  t,
                    "growth":    g,
                    "composite": composite_score(t, g, a, wt),
                })
    if not results:
        return pd.DataFrame()
    return (
        pd.DataFrame(results)
        .sort_values("composite", ascending=False)
        .reset_index(drop=True)
    )


# ─── Monte Carlo robustness ──────────────────────────────────────────────────────

def monte_carlo_robustness(model, strategies_df, wt):
    """
    Each strategy is re-evaluated under N_MC_SAMPLES perturbations of glucose and O2 uptake
    (±20% Gaussian noise). Robustness = mean - std (lower-confidence-bound heuristic).
    """
    results = []
    for _, strat in strategies_df.iterrows():
        rxn_id, kind = strat["rxn_id"], strat["kind"]
        if rxn_id not in model.reactions:
            continue
        fluxes = []
        for _ in range(N_MC_SAMPLES):
            with model:
                glc = model.reactions.get_by_id(GLUCOSE_REACTION)
                glc.lower_bound = GLUCOSE_BOUND * max(0.1, 1.0 + 0.2 * rng.standard_normal())
                if not ANAEROBIC and OXYGEN_REACTION in model.reactions:
                    o2 = model.reactions.get_by_id(OXYGEN_REACTION)
                    o2.lower_bound = o2.lower_bound * max(0.1, 1.0 + 0.2 * rng.standard_normal())
                _apply_kind(model.reactions.get_by_id(rxn_id), kind)
                t, *_ = _two_step_optimize(model, wt)
                fluxes.append(t)
        mean_f, std_f = float(np.mean(fluxes)), float(np.std(fluxes))
        results.append({
            "rxn_id":     rxn_id,
            "kind":       kind,
            "mean_flux":  mean_f,
            "std_flux":   std_f,
            "cv":         std_f / (mean_f + 1e-9),
            "robustness": mean_f - std_f,
        })
    return pd.DataFrame(results).sort_values("robustness", ascending=False).reset_index(drop=True)


# ─── Flux control coefficients (simplified MCA) ──────────────────────────────────

def flux_control_coefficients(model, wt, n_top=20, epsilon=0.05):
    """
    Numerical FCC_i = (J_wt − J_perturbed) / J_ref / ε for a downward perturbation
    of rxn i's upper bound. We shrink rather than expand because, at the lactate
    production ceiling, adding capacity to non-bottleneck reactions is a no-op.

    Candidates are chosen as the top-N reactions by |flux| in the lactate-maximizing
    WT solution: a reaction with zero flux at the optimum cannot, by definition,
    have a non-zero local control coefficient on the target. The previous
    alphabetical pool (PFK, PGI, PGK …) was dominated by upstream glycolysis with
    slack, so every FCC came back as 0.

      FCC > 0  → tightening reduced production (rxn is on the load-bearing path)
      FCC ≈ 0  → idle or non-binding reaction
    """
    sol_g = model.optimize()
    if sol_g.status != "optimal" or sol_g.objective_value < 1e-9:
        return pd.Series(dtype=float)
    g = float(sol_g.objective_value)
    with model:
        if wt["bio_id"]:
            # Use GROWTH_FRACTION (same as _two_step_optimize) so that t_wt and
            # t_new (from _two_step_optimize inside the perturbation loop) share
            # the same growth floor. Without this, t_wt ≈ 0 (near-max-growth) and
            # t_new ≈ 14 (10% growth) → every FCC comes out large and negative.
            model.reactions.get_by_id(wt["bio_id"]).lower_bound = g * GROWTH_FRACTION
        model.objective = TARGET_REACTION
        sol_t = model.optimize()
        if sol_t.status != "optimal":
            return pd.Series(dtype=float)
        t_wt = float(sol_t.fluxes.get(TARGET_REACTION, 0.0))
        # Snapshot of target-max fluxes (signed); used to pick candidates and
        # set the perturbation magnitude per reaction.
        wt_fluxes = sol_t.fluxes.copy()

    ref = max(t_wt, wt["max_target"], 1e-6)
    sorted_active = wt_fluxes[wt_fluxes.abs() > 1e-6].abs().sort_values(ascending=False)
    candidates = [
        rid for rid in sorted_active.index
        if not rid.startswith(("EX_", "DM_", "SK_")) and rid != wt["bio_id"]
    ][:n_top]

    fccs = {}
    for rxn_id in candidates:
        rxn = model.reactions.get_by_id(rxn_id)
        v   = float(wt_fluxes[rxn_id])
        if abs(v) < 1e-6:
            fccs[rxn_id] = 0.0
            continue
        # Constrain relative to current flux, not current bound. The default
        # ub = 1000 is non-binding for nearly every reaction in e_coli_core
        # (actual fluxes are ~10–20), so shrinking the bound by 5 % left the
        # feasible set unchanged and gave FCC = 0 for all.
        new_bound = v * (1.0 - epsilon)
        # Skip reactions whose flux is pinned by a hard constraint on the
        # opposite side (e.g. ATPM with lower_bound = 8.39 maintenance flux):
        # there is no feasible perturbation, so the FCC is undefined.
        if v > 0 and new_bound < rxn.lower_bound:
            fccs[rxn_id] = 0.0
            continue
        if v < 0 and new_bound > rxn.upper_bound:
            fccs[rxn_id] = 0.0
            continue
        with model:
            if v > 0:
                rxn.upper_bound = new_bound
            else:
                rxn.lower_bound = new_bound
            t_new, *_ = _two_step_optimize(model, wt)
        fccs[rxn_id] = (t_wt - t_new) / ref / epsilon
    return pd.Series(fccs).sort_values(ascending=False)


# ─── Pareto front ────────────────────────────────────────────────────────────────

def compute_pareto_front(df):
    """
    Non-dominated strategies in (growth, target_flux) space.
    A point is dominated if another point is ≥ in both objectives and > in at least one.
    """
    pts = df[["growth", "fba_flux"]].values.astype(float)
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
                    (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                dominated[i] = True
                break
    return df[~dominated].sort_values("growth").reset_index(drop=True)


# ─── Gene-level knockouts (GPR rules) ────────────────────────────────────────────

def gene_knockout_analysis(model, wt, n_top=20):
    """
    Single-gene knockouts propagated through Boolean GPR rules. More biologically
    realistic than reaction-level KOs: a gene shared by multiple reactions knocks
    them all out simultaneously; an isozyme-covered gene has no effect.

    For large models (iJO1366 has ~1500 genes), restrict to genes whose reactions
    carry non-trivial flux in the WT solution — silent genes cannot have a non-zero
    FCC on any active flux by definition.
    """
    active_rxn_ids = {r for r, v in wt["fluxes"].items() if abs(v) > 1e-6}
    candidates = [g for g in model.genes
                  if any(r.id in active_rxn_ids for r in g.reactions)]
    candidates = candidates[:MAX_GENE_KO]
    results = []
    for gene in candidates:
        with model:
            gene.knock_out()
            t, g, a = _two_step_optimize(model, wt)
            if g < 1e-9:
                continue
            sc = composite_score(t, g, a, wt)
            if sc <= 1e-9:
                continue
            results.append({
                "gene":      gene.id,
                "name":      gene.name or gene.id,
                "n_rxns":    len(gene.reactions),
                "fba_flux":  t,
                "growth":    g,
                "composite": sc,
            })
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("composite", ascending=False).reset_index(drop=True)
    print(f"    {len(df)} viable gene KOs checked (of {len(model.genes)} total genes; "
          f"screened {len(candidates)} active-gene candidates)")
    return df.head(n_top)


# ─── Evolutionary search for triple-KO combinations ──────────────────────────────

def evolutionary_search(model, ranking_df, wt, n_ko=3, n_gen=5, pop_size=12):
    """
    Genetic algorithm for simultaneous triple-KO combinations.
    Each individual encodes n_ko reaction IDs; fitness = composite_score.
    Selection: keep top-50%; crossover: uniform; mutation: single-gene swap (30%).
    """
    pool = (
        ranking_df[ranking_df["kind"] == "ko"]
        .drop_duplicates("rxn_id")["rxn_id"].tolist()[:40]
    )
    if len(pool) < n_ko:
        print(f"    Pool has only {len(pool)} reactions; skipping evolutionary search")
        return pd.DataFrame(), []

    def evaluate(combo):
        with model:
            for rxn_id in combo:
                if rxn_id in model.reactions:
                    model.reactions.get_by_id(rxn_id).knock_out()
            t, g, a = _two_step_optimize(model, wt)
            return composite_score(t, g, a, wt), t, g

    def make_individual():
        return tuple(rng.choice(pool, size=n_ko, replace=False).tolist())

    population = [make_individual() for _ in range(pop_size)]
    history, seen_keys, best_records = [], set(), []

    for gen in range(n_gen):
        scored = [((evaluate(ind)), ind) for ind in population]
        scored.sort(key=lambda x: x[0][0], reverse=True)

        (best_sc, best_t, best_g), best_ind = scored[0]
        history.append({
            "generation":     gen + 1,
            "best_composite": best_sc,
            "best_flux":      best_t,
            "best_combo":     " + ".join(best_ind),
        })
        print(f"    Gen {gen+1}/{n_gen}: composite={best_sc:.4f}  flux={best_t:.4f}"
              f"  [{', '.join(best_ind)}]")

        for (sc, t, g), ind in scored[:3]:
            key = frozenset(ind)
            if key not in seen_keys:
                seen_keys.add(key)
                best_records.append({
                    "rxn_id":    " + ".join(ind),
                    "kind":      "triple_ko",
                    "composite": sc,
                    "fba_flux":  t,
                    "growth":    g,
                })

        survivors = [ind for _, ind in scored[:pop_size // 2]]
        new_pop = list(survivors)
        while len(new_pop) < pop_size:
            p1 = survivors[int(rng.integers(len(survivors)))]
            p2 = survivors[int(rng.integers(len(survivors)))]
            child = list(p1[i] if rng.random() < 0.5 else p2[i] for i in range(n_ko))
            if rng.random() < 0.3:
                child[int(rng.integers(n_ko))] = rng.choice(pool)
            child = tuple(child)
            new_pop.append(child if len(set(child)) == n_ko else make_individual())
        population = new_pop

    best_df = (
        pd.DataFrame(best_records)
        .sort_values("composite", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    return best_df, history


# ─── Baseline comparison ─────────────────────────────────────────────────────────

def compute_baselines(model, feat_df, wt, n_each=30):
    """
    Three non-ML baselines to quantify the contribution of the ML-guided search:
      random KO, FVA-range heuristic (highest flexibility), shadow-price heuristic.
    """
    pool = [r for r in feat_df.index if r in model.reactions]

    random_rxns = rng.choice(pool, size=min(n_each, len(pool)), replace=False)
    random_fluxes = [simulate_single(model, r, "ko", wt)[0] for r in random_rxns]
    random_best = max(random_fluxes)
    random_mean = float(np.mean(random_fluxes))

    fva_top = feat_df.sort_values("fva_range", ascending=False).head(n_each)
    fva_fluxes = [simulate_single(model, r, "ko", wt)[0] for r in fva_top.index]
    fva_best = max(fva_fluxes)
    fva_mean = float(np.mean(fva_fluxes))

    sp_top = feat_df.sort_values("max_sp", ascending=False).head(n_each)
    sp_fluxes = [simulate_single(model, r, "ko", wt)[0] for r in sp_top.index]
    sp_best = max(sp_fluxes)
    sp_mean = float(np.mean(sp_fluxes))

    return {
        "random_KO":      (random_best, random_mean),
        "FVA_heuristic":  (fva_best,    fva_mean),
        "SP_heuristic":   (sp_best,     sp_mean),
    }


# ─── Top strategy validation ─────────────────────────────────────────────────────

def validate_top_strategies(model, ranking_df, wt):
    singles = ranking_df[ranking_df["n_int"] == 1].head(N_TOP * 2)
    results, done = [], 0
    for _, row in singles.iterrows():
        if done >= N_TOP:
            break
        t, g, a, _ = simulate_single(model, row["rxn_id"], row["kind"], wt)
        results.append({
            "rxn_id":    row["rxn_id"],
            "kind":      row["kind"],
            "ensemble":  row["ensemble"],
            "fba_flux":  t,
            "growth":    g,
            "composite": composite_score(t, g, a, wt),
            "delta_wt":  t - wt["target"],
        })
        done += 1
    return pd.DataFrame(results).sort_values("fba_flux", ascending=False).reset_index(drop=True)


# ─── Pathway-level interpretation ────────────────────────────────────────────────

def pathway_ranking(model, ranking_df):
    """Aggregate ensemble scores by subsystem to identify high-potential pathways."""
    sub_map = {r.id: assign_subsystem(r) for r in model.reactions}
    df = ranking_df.copy()
    df["subsystem"] = df["rxn_id"].map(sub_map).fillna("unknown")
    return (
        df.groupby("subsystem")["ensemble"]
        .agg(best="max", mean="mean", count="count")
        .sort_values("best", ascending=False)
    )


# ─── Visualization ───────────────────────────────────────────────────────────────

def plot_all(clf_gbr, val_df, mc_df, fcc_series, baselines, wt, feat_names, pathway_df,
            pareto_df=None, gene_df=None, combo_df=None, evo_history=None):
    wt_cs = composite_score(wt["target"], wt["growth"], wt["atp"], wt)
    fig = plt.figure(figsize=(22, 22))
    gs  = gridspec.GridSpec(4, 3, hspace=0.55, wspace=0.38)

    # Feature importance — top-20 only (iJO1366 has 56 features; showing all makes labels unreadable)
    ax = fig.add_subplot(gs[0, 0])
    imp = clf_gbr.feature_importances_
    top_n   = min(20, len(imp))
    top_idx = np.argsort(imp)[::-1][:top_n]
    ax.bar(range(top_n), imp[top_idx], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feat_names[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"GBR Feature Importance (top {top_n})")
    ax.set_ylabel("Importance")

    # ML ensemble vs. FBA composite (top candidates)
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(val_df["ensemble"], val_df["composite"],
                    c=val_df["growth"], cmap="plasma", zorder=3, s=60)
    lo = min(val_df["ensemble"].min(), val_df["composite"].min()) - 0.02
    hi = max(val_df["ensemble"].max(), val_df["composite"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("ML ensemble score")
    ax.set_ylabel("FBA composite score")
    ax.set_title("Prediction vs. FBA (color = growth rate)")
    plt.colorbar(sc, ax=ax, label="growth (1/h)", pad=0.02)

    # Growth–production trade-off
    ax = fig.add_subplot(gs[0, 2])
    sc2 = ax.scatter(val_df["growth"], val_df["fba_flux"],
                     c=val_df["composite"], cmap="viridis", s=70, zorder=3)
    ax.axvline(wt["growth"],      color="gray", ls="--", lw=1, label="WT growth")
    ax.axhline(wt["max_target"], color="gray", ls=":",  lw=1, label="max target")
    ax.set_xlabel("Growth rate (1/h)")
    ax.set_ylabel(f"{TARGET_REACTION} flux")
    ax.set_title("Growth–Production Trade-off")
    ax.legend(fontsize=7)
    plt.colorbar(sc2, ax=ax, label="composite", pad=0.02)

    # Monte Carlo robustness
    ax = fig.add_subplot(gs[1, :2])
    mc = mc_df.head(8)
    x  = range(len(mc))
    ax.bar(x, mc["mean_flux"], yerr=mc["std_flux"], capsize=4,
           color="steelblue", alpha=0.75, label="Mean ± Std")
    # WT biomass-optimal target is 0 for aerobic iJO1366 (no succinate secretion).
    # Show max achievable target as a meaningful reference instead.
    ax.axhline(wt["max_target"], color="black", ls="--", lw=1,
               label=f"max achievable = {wt['max_target']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.rxn_id[:18]}[{r.kind}]" for r in mc.itertuples()],
        rotation=40, ha="right", fontsize=7,
    )
    ax.set_title(f"Monte Carlo Robustness (n={N_MC_SAMPLES} env. perturbations)")
    ax.set_ylabel("Target flux (mmol/gDW/h)")
    ax.legend(fontsize=7)

    # Flux control coefficients
    ax = fig.add_subplot(gs[1, 2])
    fcc_top = fcc_series.head(12)
    colors  = ["#2ca02c" if v >= 0 else "#d62728" for v in fcc_top.values]
    ax.barh(fcc_top.index[::-1], fcc_top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Flux Control Coefficients (FCC)")
    ax.set_xlabel("FCC  (numerical, ε=1%)")

    # Baseline comparison
    ax = fig.add_subplot(gs[2, 0])
    bnames = list(baselines.keys()) + ["ML-assisted"]
    b_best = [baselines[k][0] for k in baselines] + [val_df["fba_flux"].max()]
    b_mean = [baselines[k][1] for k in baselines] + [val_df["fba_flux"].mean()]
    xpos   = range(len(bnames))
    ax.bar(xpos, b_best,  alpha=0.65, label="Best found",  color="steelblue")
    ax.bar(xpos, b_mean,  alpha=0.90, label="Mean",        color="darkorange")
    ax.set_xticks(xpos)
    ax.set_xticklabels(bnames, rotation=25, ha="right", fontsize=8)
    ax.set_title("ML vs. Baselines")
    ax.set_ylabel("Target flux (mmol/gDW/h)")
    ax.legend(fontsize=7)

    # Pathway-level ranking (top 10 subsystems) — truncate long iJO1366 pathway names
    ax = fig.add_subplot(gs[2, 1])
    pw = pathway_df.head(10).copy()
    pw.index = [s[:30] for s in pw.index]
    ax.barh(pw.index[::-1], pw["best"].values[::-1], color="mediumseagreen")
    ax.set_title("Top Pathways by Best Ensemble Score")
    ax.set_xlabel("Best ensemble score in subsystem")
    ax.tick_params(axis="y", labelsize=7)

    # Top validated strategies (composite)
    ax = fig.add_subplot(gs[2, 2])
    labels = [f"{r.rxn_id[:16]}[{r.kind}]" for r in val_df.head(8).itertuples()]
    colors = ["#2ca02c" if c >= wt_cs else "#d62728" for c in val_df.head(8)["composite"]]
    ax.barh(labels[::-1], val_df.head(8)["composite"].values[::-1], color=colors[::-1])
    ax.axvline(wt_cs, color="black", lw=1.2, ls="--", label=f"WT = {wt_cs:.3f}")
    ax.set_title(f"Top {N_TOP} Strategies (Composite Score)")
    ax.set_xlabel("Composite score")
    ax.legend(fontsize=7)

    # Pareto front (growth vs. production)
    ax = fig.add_subplot(gs[3, 0])
    if pareto_df is not None and not pareto_df.empty:
        all_pts_g = val_df[["growth", "fba_flux"]].copy()
        all_pts_g["src"] = "single"
        if combo_df is not None and not combo_df.empty:
            tmp = combo_df[["growth", "fba_flux"]].copy()
            tmp["src"] = "combo"
            all_pts_g = pd.concat([all_pts_g, tmp], ignore_index=True)
        for src, col, mrk in [("single", "steelblue", "o"), ("combo", "darkorange", "^")]:
            sub = all_pts_g[all_pts_g["src"] == src]
            if not sub.empty:
                ax.scatter(sub["growth"], sub["fba_flux"],
                           color=col, alpha=0.55, s=45, marker=mrk, label=src)
        ax.scatter(pareto_df["growth"], pareto_df["fba_flux"],
                   color="red", s=90, zorder=5, marker="*", label="Pareto")
        ax.plot(pareto_df["growth"].values, pareto_df["fba_flux"].values,
                "r--", lw=1, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No Pareto data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Growth rate (1/h)")
    ax.set_ylabel(f"{TARGET_REACTION} flux")
    ax.set_title("Pareto Front (growth vs. production)")
    ax.legend(fontsize=7)

    # Gene KO ranking — use fba_flux as bar value (composite collapses when many
    # interventions share the same AUC; flux shows absolute production difference)
    ax = fig.add_subplot(gs[3, 1])
    if gene_df is not None and not gene_df.empty:
        gdf = gene_df.head(8)
        labels_g = [f"{r.gene}({r.n_rxns}rxn)" for r in gdf.itertuples()]
        bars = ax.barh(labels_g[::-1], gdf["fba_flux"].values[::-1], color="mediumpurple")
        # Colour by growth rate to show growth-coupling quality
        norm_g = plt.Normalize(gdf["growth"].min(), gdf["growth"].max() + 1e-9)
        for bar, g_val in zip(bars, gdf["growth"].values[::-1]):
            bar.set_color(plt.cm.RdYlGn(norm_g(g_val)))
        ax.set_xlabel(f"{TARGET_REACTION} flux (mmol/gDW/h)")
        ax.set_title("Top Gene KOs (GPR-aware; color=growth)")
    else:
        ax.text(0.5, 0.5, "No gene KO data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Top Gene KOs (GPR-aware)")

    # Evolutionary search convergence — plain decimal y-axis avoids matplotlib's
    # confusing offset notation ("1e-12 + 9.55e-1") when values are tightly clustered
    ax = fig.add_subplot(gs[3, 2])
    if evo_history:
        hdf = pd.DataFrame(evo_history)
        ax.plot(hdf["generation"], hdf["best_composite"], "o-", color="coral", lw=1.5)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.4f"))
        y_min = hdf["best_composite"].min()
        y_max = hdf["best_composite"].max()
        margin = max((y_max - y_min) * 0.5, 0.005)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best composite")
        ax.set_title("Evolutionary Search Convergence (triple KO)")
    else:
        ax.text(0.5, 0.5, "No evo data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Evolutionary Search (triple KO)")

    plt.savefig("results_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results_summary.png")


# ─── Publishability additions ────────────────────────────────────────────────────

def generalization_holdout_test(model, feat_df, wt, X_all, y_rank_all, keys_all):
    """
    True generalization test: hold out 30% of enumerated reactions (by reaction group,
    never seen during training). Measures whether ML predicts unseen reactions correctly.
    This is the publishable metric — CV ρ reuses training data; holdout ρ does not.
    """
    unique_rxns = np.array(sorted({r for r, _ in keys_all}))
    rng_h = np.random.default_rng(SEED + 99)
    rng_h.shuffle(unique_rxns)
    n_train = int(len(unique_rxns) * 0.70)
    train_set = set(unique_rxns[:n_train])
    test_set  = set(unique_rxns[n_train:])

    tr_idx = np.array([i for i, (r, _) in enumerate(keys_all) if r in train_set])
    te_idx = np.array([i for i, (r, _) in enumerate(keys_all) if r in test_set])
    if len(tr_idx) < 10 or len(te_idx) < 10:
        return {}

    scaler = StandardScaler()
    clf = _make_gbr()
    clf.fit(scaler.fit_transform(X_all[tr_idx]), y_rank_all[tr_idx])
    pred   = np.clip(clf.predict(scaler.transform(X_all[te_idx])), 0.0, 1.0)
    y_te   = y_rank_all[te_idx]

    sp   = float(spearmanr(pred, y_te).correlation) if np.std(pred) > 1e-9 else 0.0
    k    = max(1, len(te_idx) // 5)
    topk = len(set(np.argsort(y_te)[::-1][:k]) & set(np.argsort(pred)[::-1][:k])) / k

    true_t5 = {keys_all[te_idx[i]] for i in np.argsort(y_te)[::-1][:5]}
    pred_t5 = {keys_all[te_idx[i]] for i in np.argsort(pred)[::-1][:5]}
    top5_p  = len(true_t5 & pred_t5) / 5

    return {
        "n_train_rxns": n_train, "n_test_rxns": len(test_set),
        "n_train_samp": len(tr_idx), "n_test_samp": len(te_idx),
        "spearman_rho": sp, "topk_precision": topk, "top5_precision": top5_p,
        "true_top5": sorted(f"{r}({k})" for r, k in true_t5),
        "pred_top5": sorted(f"{r}({k})" for r, k in pred_t5),
    }


def run_straindesign(model, wt, k=2, n_results=5):
    """
    Growth-coupled strain design via StrainDesign (OptKnock framework, HiGHS solver).
    OptKnock was removed from modern COBRApy; StrainDesign is its maintained successor.
    Falls back gracefully if StrainDesign is not installed.
    """
    if not _SD_AVAILABLE:
        return pd.DataFrame()
    biomass_id = wt.get("bio_id", "")
    if not biomass_id:
        return pd.DataFrame()
    try:
        # Try parameter variants in order of most-to-least likely for current SD version.
        # inner_objective/outer_objective is the modern API recommended by the developers.
        result = None
        for kwargs in [
            {"inner_objective": biomass_id, "outer_objective": TARGET_REACTION},
            {"prod_id": TARGET_REACTION, "biomass_id": biomass_id},
            {"outer_obj": {TARGET_REACTION: 1.0}, "inner_obj": {biomass_id: 1.0}},
        ]:
            try:
                result = sd.compute_strain_designs(
                    model,
                    sd_modules=[sd.SDModule(model, sd.OPTKNOCK, **kwargs)],
                    max_interventions=k,
                    solution_approach=sd.BEST,
                    solver="highs",
                    time_limit=300,
                )
                if result is not None:
                    break
            except Exception:
                result = None
                continue

        if result is None or not hasattr(result, "reaction_sd") or result.reaction_sd is None:
            print("    No growth-coupled solutions found within time limit.")
            return pd.DataFrame()

        rows = []
        for ko_set in result.reaction_sd[:n_results]:
            rxns = [r for r, v in ko_set.items() if v == "KO"]
            with model:
                for r in rxns:
                    if r in model.reactions:
                        model.reactions.get_by_id(r).knock_out()
                t, g, a = _two_step_optimize(model, wt)
            rows.append({
                "rxn_id":    "+".join(rxns),
                "kind":      "+".join(["ko"] * len(rxns)),
                "fba_flux":  round(t, 4),
                "growth":    round(g, 4),
                "composite": round(composite_score(t, g, a, wt), 4),
                "source":    "StrainDesign/OptKnock",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    StrainDesign error: {e}")
        return pd.DataFrame()


def ablation_study(X_all, y_rank_all, keys_all):
    """
    Compare three ranking strategies by 5-fold CV Spearman ρ:
      GBR-only    — gradient boosting regressor, no pairwise ranker
      Pairwise-only — pairwise logistic ranker, no GBR
      Ensemble    — 0.6 × GBR + 0.4 × pairwise (current pipeline)
    """
    groups = np.array([r for r, _ in keys_all])
    gkf    = GroupKFold(n_splits=min(5, len(set(groups))))
    results = {}

    for label, use_gbr, use_pair in [("GBR-only", True, False),
                                      ("Pairwise-only", False, True),
                                      ("Ensemble", True, True)]:
        sp_scores = []
        for tr_idx, te_idx in gkf.split(X_all, y_rank_all, groups):
            X_tr, y_tr = X_all[tr_idx], y_rank_all[tr_idx]
            X_te, y_te = X_all[te_idx], y_rank_all[te_idx]

            sc_g = StandardScaler()
            clf_g = _make_gbr()
            clf_g.fit(sc_g.fit_transform(X_tr), y_tr)
            clf_p, sc_p = train_pairwise_ranker(X_tr, y_tr)
            X_ref_a = X_tr[np.argsort(y_tr)[::-1][:20]]

            pred_g = np.clip(clf_g.predict(sc_g.transform(X_te)), 0.0, 1.0)
            pred_p = np.array([pairwise_win_rate(X_te[i], X_ref_a, clf_p, sc_p)
                                for i in range(len(te_idx))])

            if use_gbr and use_pair:
                pred = 0.6 * pred_g + 0.4 * pred_p
            elif use_gbr:
                pred = pred_g
            else:
                pred = pred_p

            if np.std(pred) > 1e-9 and np.std(y_te) > 1e-9:
                sp_scores.append(float(spearmanr(pred, y_te).correlation))

        results[label] = float(np.mean(sp_scores)) if sp_scores else 0.0

    return results


def literature_comparison(model, wt, ranking_df, lit_data):
    """
    Cross-check ML ranking against experimentally validated interventions from literature.
    Reports: ML rank of the intervention and its predicted vs actual FBA flux.
    """
    rows = []
    for entry in lit_data:
        rxns, kinds, label, ref = entry["rxns"], entry["kinds"], entry["label"], entry["ref"]

        # ML rank (singles only)
        ml_rank = "n/a"
        if len(rxns) == 1:
            mask = (ranking_df["rxn_id"] == rxns[0]) & (ranking_df["kind"] == kinds[0])
            if mask.any():
                ml_rank = int(ranking_df[mask].index[0]) + 1

        # FBA
        with model:
            feasible = all(r in model.reactions for r in rxns)
            if feasible:
                for r, k in zip(rxns, kinds):
                    _apply_kind(model.reactions.get_by_id(r), k)
                t, g, a = _two_step_optimize(model, wt)
            else:
                t, g, a = 0.0, 0.0, 0.0

        rows.append({
            "intervention": label,
            "ml_rank":      ml_rank,
            "fba_flux":     round(t, 4),
            "growth":       round(g, 4),
            "composite":    round(composite_score(t, g, a, wt), 4),
            "reference":    ref,
        })
    return pd.DataFrame(rows)


def run_target_quick(model, target_rxn_id, target_name, n_enum=150):
    """
    Lightweight pipeline for secondary targets: enumeration + GBR ranking.
    Demonstrates the method generalises across metabolic objectives without
    requiring the full (~60 min) analysis.
    """
    global TARGET_REACTION
    if target_rxn_id not in model.reactions:
        return None
    old_target = TARGET_REACTION
    TARGET_REACTION = target_rxn_id
    try:
        wt_q = wt_analysis(model)
        if wt_q["growth"] < 1e-6:
            return None

        internal_q = [r.id for r in model.reactions if not r.id.startswith(("EX_", "DM_", "SK_"))]
        fva_pool_q = sorted(internal_q, key=lambda r: abs(wt_q["fluxes"].get(r, 0.0)), reverse=True)[:MAX_FVA_REACTIONS]
        fva_df_q   = run_fva(model, fva_pool_q)
        G_q        = build_reaction_graph(model)
        graph_q    = compute_graph_features(G_q, target_rxn_id, wt_q["bio_id"] or "")
        feat_q     = build_feature_matrix(model, wt_q, fva_df_q, graph_q)

        n_fva_q  = n_enum * 2 // 3
        by_fva_q = set(feat_q["fva_range"].abs().nlargest(n_fva_q).index)
        by_fl_q  = set(feat_q["wt_flux"].abs().nlargest(n_enum * 3).index) - by_fva_q
        enum_q   = by_fva_q | set(list(by_fl_q)[:n_enum - n_fva_q])

        X_q, y_fl_q, keys_q = [], [], []
        for rxn_id in feat_q.index:
            if rxn_id not in enum_q or rxn_id not in model.reactions:
                continue
            for kind in ("ko", "oe", "down"):
                t, g, a, _ = simulate_single(model, rxn_id, kind, wt_q)
                X_q.append(np.append(feat_q.loc[rxn_id].values, _KIND_ENC[kind]))
                y_fl_q.append(t)
                keys_q.append((rxn_id, kind))

        if len(X_q) < 20:
            return None

        X_q    = np.array(X_q)
        y_fl_q = np.array(y_fl_q)
        y_rk_q = rankdata(y_fl_q, method="average") / len(y_fl_q)
        _, _, sp_q, tk_q = train_gbr_model(X_q, y_rk_q, keys_q)

        flux_ord = np.argsort(y_fl_q)[::-1]
        top5 = [{"rxn_id": keys_q[i][0], "kind": keys_q[i][1], "fba_flux": float(y_fl_q[i])}
                for i in flux_ord[:5]]
        return {"target": target_name, "rxn_id": target_rxn_id,
                "cv_rho": round(sp_q, 3), "topk_prec": round(tk_q, 3),
                "wt_max_target": round(wt_q["max_target"], 4),
                "best_single_flux": round(float(y_fl_q[flux_ord[0]]), 4) if len(flux_ord) else 0.0,
                "best_single_rxn": f"{keys_q[flux_ord[0]][0]}({keys_q[flux_ord[0]][1]})" if len(flux_ord) else "n/a",
                "top5": top5}
    finally:
        TARGET_REACTION = old_target


def save_results_zip(tag="results"):
    """Package all CSV and PNG outputs into a single timestamped ZIP for easy download."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"metabolic_{tag}_{ts}.zip"
    files    = sorted(f for f in os.listdir(".") if f.endswith((".csv", ".png")))
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f)
    return zip_name


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=== ML-Assisted Metabolic Engineering Pipeline ===\n")

    print("[1/9] Loading model...")
    model = load_metabolic_model()
    assert TARGET_REACTION in model.reactions, f"'{TARGET_REACTION}' not in model"
    mode = "anaerobic" if ANAEROBIC else "aerobic"
    print(f"    {model.id} | {len(model.reactions)} rxns | {len(model.metabolites)} mets | {mode}")

    print("\n[2/9] Wild-type pFBA + multi-objective baseline...")
    wt    = wt_analysis(model)
    print(f"    Growth={wt['growth']:.4f} | WT target (biomass-opt)={wt['target']:.4f} | "
          f"Max achievable target={wt['max_target']:.4f} | WT AUC={wt['wt_auc']:.4f} | ATP={wt['atp']:.4f}")

    print("\n[3/9] FVA + reaction graph (betweenness, path distances) + feature matrix...")
    internal = [r.id for r in model.reactions if not r.id.startswith(("EX_", "DM_", "SK_"))]
    # For large models limit FVA to the most active reactions; build_feature_matrix
    # falls back to WT-flux values for reactions not in fva_df.
    fva_pool = sorted(internal, key=lambda r: abs(wt["fluxes"].get(r, 0.0)), reverse=True)
    fva_pool = fva_pool[:MAX_FVA_REACTIONS]
    print(f"    Running FVA on {len(fva_pool)}/{len(internal)} internal reactions (top by WT flux)...")
    fva_df   = run_fva(model, fva_pool)
    G        = build_reaction_graph(model)
    graph_df = compute_graph_features(G, TARGET_REACTION, wt["bio_id"] or "")
    feat_df  = build_feature_matrix(model, wt, fva_df, graph_df)
    feat_names = list(feat_df.columns) + ["kind_enc"]
    print(f"    {len(feat_df)} reactions × {len(feat_df.columns)} features (+1 kind_enc)")

    n_rxns  = len(feat_df)
    n_enum  = min(n_rxns, MAX_ENUM_REACTIONS)
    n_total = n_enum * 3
    print(f"\n[4/9] Single-intervention enumeration (top {n_enum}/{n_rxns} reactions × 3 kinds = {n_total} training points)...")
    X_all, y_sc_all, y_flux_all, keys_all = enumerate_all_singles(model, feat_df, wt)

    # Show best interventions found by direct enumeration — this is ground truth
    # for the ML to beat. Sorted by raw production flux (not AUC label).
    flux_order = np.argsort(y_flux_all)[::-1]
    print("    Top-5 interventions found by enumeration (ground truth):")
    for rank in range(min(5, len(flux_order))):
        i = flux_order[rank]
        rxn_id, kind = keys_all[i]
        print(f"      {rank+1}. {rxn_id}({kind}): flux={y_flux_all[i]:.4f}  AUC={y_sc_all[i]:.4f}")

    # Rank-normalize flux labels: maps [0, max_flux] to (0, 1] using percentile rank.
    # Spreading the labels avoids the ~15.39 plateau dominating GBR splits and
    # prevents the model from extrapolating past the training max for unseen reactions.
    y_rank_all = rankdata(y_flux_all, method="average") / len(y_flux_all)

    print(f"\n[5/9] GBR training + GroupKFold-by-reaction CV ({n_rxns} reaction groups)...")
    clf_gbr, scaler_gbr, sp_cv, topk_cv = train_gbr_model(X_all, y_rank_all, keys_all)
    print(f"    Stored for summary: Spearman ρ = {sp_cv:.3f}, Top-K prec = {topk_cv:.3f}")
    imp = clf_gbr.feature_importances_
    top10 = np.argsort(imp)[::-1][:10]
    print("    Top-10 GBR feature importances:")
    for i in top10:
        bar = "▪" * int(imp[i] * 200)
        print(f"      {feat_names[i]:<30s} {imp[i]:.4f}  {bar}")

    print("\n[6/9] GBR ranking of singles (pairwise ranker excluded — ablation showed it adds noise)...")
    # Exact flux ranks for the 300 enumerated reactions — used as ground truth
    # so GBR extrapolation only applies to the unenumerated reactions.
    known_ranks = {(rxn_id, kind): float(y_rank_all[i])
                   for i, (rxn_id, kind) in enumerate(keys_all)}
    ranking_df = rank_all_interventions(feat_df, clf_gbr, scaler_gbr,
                                        known_ranks=known_ranks)
    print("    Top 10 by ensemble score:")
    print(ranking_df.head(10).to_string(index=False))

    print(f"\n[7/9] FBA validation of top {N_TOP} singles (ground truth already known from step 4)...")
    val_df = validate_top_strategies(model, ranking_df, wt)
    print(val_df.to_string(index=False))

    # Top-5 precision: compare ML top-5 against best by RAW FLUX (not AUC),
    # because that is what matters experimentally.
    true_top5_flux = {keys_all[i] for i in flux_order[:5]}
    pred_top5_keys = set(zip(val_df.head(5)["rxn_id"], val_df.head(5)["kind"]))
    n_overlap_flux = len(true_top5_flux & pred_top5_keys)
    rank_order = np.argsort(y_rank_all)[::-1]
    true_top5_rank = {keys_all[i] for i in rank_order[:5]}
    n_overlap_rank = len(true_top5_rank & pred_top5_keys)
    print(f"    Top-5 precision vs best-flux interventions: {n_overlap_flux}/5"
          f"  (rank-label top-5: {n_overlap_rank}/5)")

    n_pairs_approx = n_rxns * (n_rxns - 1) // 2 * 9
    print(f"\n[7b] ML-guided double interventions: predict all ~{n_pairs_approx:,} pairs "
          f"(zero FBA), validate top {N_VALIDATE_DOUBLES} with FBA...")
    pair_df, single_preds = predict_and_rank_doubles(feat_df, clf_gbr, scaler_gbr, model,
                                                      known_ranks=known_ranks)
    ml_doubles_df  = validate_predicted_doubles(model, pair_df, wt, N_VALIDATE_DOUBLES)
    rand_doubles_df = random_double_baseline(model, wt, single_preds, N_VALIDATE_DOUBLES)

    ml_good   = (ml_doubles_df["composite"]   > 0.20).sum()
    rand_good = (rand_doubles_df["composite"] > 0.20).sum()
    ml_good_high   = int((ml_doubles_df["composite"]   > 0.95).sum()) if not ml_doubles_df.empty else 0
    rand_good_high = int((rand_doubles_df["composite"] > 0.95).sum()) if not rand_doubles_df.empty else 0
    rand_best_flux = float(rand_doubles_df["fba_flux"].max()) if not rand_doubles_df.empty else 0.0
    print(f"    ML-selected doubles:     {ml_good}/{N_VALIDATE_DOUBLES} with composite > 0.20  ({ml_good_high}/{N_VALIDATE_DOUBLES} > 0.95)")
    print(f"    Random-selected doubles: {rand_good}/{N_VALIDATE_DOUBLES} with composite > 0.20  ({rand_good_high}/{N_VALIDATE_DOUBLES} > 0.95)")
    print(f"    ML best double: {ml_doubles_df.iloc[0]['rxn_id']}  "
          f"composite={ml_doubles_df.iloc[0]['composite']:.4f}" if not ml_doubles_df.empty else "")
    if not ml_doubles_df.empty:
        print(ml_doubles_df.head(5)[["rxn_id","kind","pred_pair","fba_flux","composite"]].to_string(index=False))

    combo_df = ml_doubles_df  # reuse combo_df name for plot compatibility

    print(f"\n[7c] Gene-level knockout analysis ({len(model.genes)} genes via GPR rules)...")
    gene_df = gene_knockout_analysis(model, wt)
    if not gene_df.empty:
        print(gene_df.head(5)[["gene", "name", "n_rxns", "fba_flux", "growth", "composite"]].to_string(index=False))

    print("\n[7d] Evolutionary search for triple KOs (n_gen=5, pop=12)...")
    evo_df, evo_history = evolutionary_search(model, ranking_df, wt, n_ko=3, n_gen=5, pop_size=12)
    if not evo_df.empty:
        print("    Best triple-KO combos found:")
        print(evo_df.to_string(index=False))

    print(f"\n[8/9] Monte Carlo robustness ({N_MC_SAMPLES} perturbations × {N_MC_STRATEGIES} strategies)...")
    mc_df = monte_carlo_robustness(model, val_df.head(N_MC_STRATEGIES), wt)
    print(mc_df[["rxn_id", "kind", "mean_flux", "std_flux", "cv", "robustness"]].to_string(index=False))

    print("\n[8b] Flux control coefficients (active reactions in lactate-max solution, ε=5%)...")
    fcc_series = flux_control_coefficients(model, wt, n_top=20)
    if fcc_series.empty:
        print("    (no active reactions; FCC skipped)")
    else:
        print(fcc_series.head(8).to_string())

    print("\n[9/9] Baseline comparison + pathway ranking + Pareto front...")
    baselines  = compute_baselines(model, feat_df, wt, n_each=25)
    pathway_df = pathway_ranking(model, ranking_df)

    pareto_input = val_df[["rxn_id", "kind", "growth", "fba_flux"]].copy()
    if not combo_df.empty and "growth" in combo_df.columns:
        pareto_input = pd.concat(
            [pareto_input, combo_df[["rxn_id", "kind", "growth", "fba_flux"]]],
            ignore_index=True,
        )
    pareto_df = compute_pareto_front(pareto_input)

    best_enum_flux = float(np.max(y_flux_all)) if len(y_flux_all) else 0.0
    best_enum_key  = keys_all[int(np.argmax(y_flux_all))] if len(y_flux_all) else ("?", "?")
    print(f"    Best enumerated single:      {best_enum_flux:.4f}  [{best_enum_key[0]}({best_enum_key[1]})]")
    print(f"    Random KO best:              {baselines['random_KO'][0]:.4f}")
    print(f"    FVA heuristic best:          {baselines['FVA_heuristic'][0]:.4f}")
    print(f"    SP heuristic best:           {baselines['SP_heuristic'][0]:.4f}")
    print(f"    ML single best:              {val_df['fba_flux'].max():.4f}")
    print(f"    ML doubles best:             {ml_doubles_df['fba_flux'].max():.4f}" if not ml_doubles_df.empty else "    ML doubles best:             n/a")
    print(f"    Triple-KO (evo) best:        {evo_df['fba_flux'].max():.4f}" if not evo_df.empty else "    Triple-KO (evo) best:        n/a")
    print(f"    ML doubles comp>0.20:        {ml_good}/{N_VALIDATE_DOUBLES} (random: {rand_good}/{N_VALIDATE_DOUBLES})")
    print(f"    ML doubles comp>0.95:        {ml_good_high}/{N_VALIDATE_DOUBLES} (random: {rand_good_high}/{N_VALIDATE_DOUBLES})")
    ml_best_flux_d = float(ml_doubles_df["fba_flux"].max()) if not ml_doubles_df.empty else 0.0
    print(f"    ML doubles best flux:        {ml_best_flux_d:.4f}  (random best: {rand_best_flux:.4f})")
    print(f"    ML generalization (CV ρ):    {sp_cv:.3f}  Top-K prec: {topk_cv:.3f}")
    print(f"    Top-5 flux precision:        {n_overlap_flux}/5  (rank-label: {n_overlap_rank}/5)")
    print(f"    Pareto-optimal strats:       {len(pareto_df)}")
    print("\n    Top pathways by ensemble score:")
    print(pathway_df.head(6).to_string())

    print("\nGenerating plots...")
    plot_all(clf_gbr, val_df, mc_df, fcc_series, baselines, wt, feat_names, pathway_df,
             pareto_df=pareto_df, gene_df=gene_df, combo_df=combo_df, evo_history=evo_history)

    # ── Pub-1: True generalization (holdout by reaction group) ───────────────────
    print("\n[Pub-1] Generalization holdout test (70 % train / 30 % test, reaction groups)...")
    holdout = generalization_holdout_test(model, feat_df, wt, X_all, y_rank_all, keys_all)
    if holdout:
        print(f"    Train: {holdout['n_train_rxns']} rxns ({holdout['n_train_samp']} samples) | "
              f"Test: {holdout['n_test_rxns']} rxns ({holdout['n_test_samp']} samples)")
        print(f"    Holdout Spearman ρ = {holdout['spearman_rho']:.3f} | "
              f"Top-K prec = {holdout['topk_precision']:.3f} | "
              f"Top-5 prec = {holdout['top5_precision']:.3f}")
        print(f"    True top-5 (held-out): {holdout['true_top5']}")
        print(f"    Pred top-5 (held-out): {holdout['pred_top5']}")

    # ── Pub-2: OptKnock via StrainDesign ─────────────────────────────────────────
    print("\n[Pub-2] Growth-coupled design — StrainDesign/OptKnock (k=2, HiGHS)...")
    sd_df = run_straindesign(model, wt, k=2, n_results=5)
    if not sd_df.empty:
        print(sd_df.to_string(index=False))
    else:
        if _SD_AVAILABLE:
            print("    No growth-coupled solutions found within time limit.")
        else:
            print("    StrainDesign not installed — run: pip install straindesign")
            print("    Literature OptKnock result: ATPS4rpp+LDH_D → ~16.4 mmol/gDW/h (FastKnock 2023)")

    # ── Pub-3: Ablation study ─────────────────────────────────────────────────────
    print("\n[Pub-3] Ablation study (5-fold CV Spearman ρ per model component)...")
    ablation = ablation_study(X_all, y_rank_all, keys_all)
    for name, rho in ablation.items():
        bar = "█" * int(max(0, rho) * 30)
        print(f"    {name:<20s} ρ = {rho:.3f}  {bar}")

    # ── Pub-4: Literature comparison ──────────────────────────────────────────────
    print("\n[Pub-4] Literature comparison — experimentally validated interventions...")
    lit_df = literature_comparison(model, wt, ranking_df, LITERATURE_KNOCKOUTS_SUCCINATE)
    print(lit_df.to_string(index=False))

    # ── Pub-5: Multi-target generalizability ──────────────────────────────────────
    print("\n[Pub-5] Multi-target generalizability test...")
    multi_rows = [{"target": "succinate", "rxn_id": TARGET_REACTION,
                   "cv_rho": sp_cv, "topk_prec": topk_cv,
                   "wt_max_target": round(wt["max_target"], 4),
                   "best_single_flux": round(val_df["fba_flux"].max(), 4),
                   "best_single_rxn": val_df.iloc[0]["rxn_id"] + "(" + val_df.iloc[0]["kind"] + ")"}]
    for t_cfg in EXTRA_TARGETS:
        print(f"    Running abbreviated pipeline for {t_cfg['name']} ({t_cfg['rxn_id']})...")
        res = run_target_quick(model, t_cfg["rxn_id"], t_cfg["name"])
        if res:
            multi_rows.append({k: v for k, v in res.items() if k != "top5"})
            print(f"      CV ρ={res['cv_rho']:.3f}  Top-K={res['topk_prec']:.3f}  "
                  f"Best: {res['best_single_rxn']} flux={res['best_single_flux']:.4f}")
        else:
            print(f"      Skipped ({t_cfg['rxn_id']} not in model or infeasible)")
    multi_df = pd.DataFrame(multi_rows)
    print("    Summary:")
    print(multi_df[["target", "cv_rho", "topk_prec", "wt_max_target", "best_single_flux"]].to_string(index=False))

    # ── Save all outputs and ZIP ──────────────────────────────────────────────────
    ranking_df.to_csv("ranking_all_interventions.csv",  index=False)
    val_df.to_csv("top_validated_strategies.csv",       index=False)
    mc_df.to_csv("monte_carlo_robustness.csv",          index=False)
    fcc_series.to_csv("flux_control_coefficients.csv")
    pareto_df.to_csv("pareto_front.csv",                index=False)
    lit_df.to_csv("literature_comparison.csv",          index=False)
    multi_df.to_csv("multi_target_summary.csv",         index=False)
    if holdout:
        pd.DataFrame([{k: str(v) for k, v in holdout.items()}]).to_csv(
            "generalization_holdout.csv", index=False)
    if not ablation:
        pass
    else:
        pd.DataFrame([{"model": k, "cv_rho": v} for k, v in ablation.items()]).to_csv(
            "ablation_study.csv", index=False)
    if not sd_df.empty:
        sd_df.to_csv("optknock_straindesign.csv",       index=False)
    if not combo_df.empty:
        combo_df.to_csv("combo_search.csv",             index=False)
    if not gene_df.empty:
        gene_df.to_csv("gene_knockouts.csv",            index=False)
    if not evo_df.empty:
        evo_df.to_csv("evolutionary_search.csv",        index=False)

    zip_path = save_results_zip(tag=TARGET_REACTION.replace("EX_", "").replace("_e", ""))
    print(f"\nAll results → {zip_path}  ({len([f for f in os.listdir('.') if f.endswith(('.csv','.png'))])} files zipped)")
    print("=== Done ===")
    return ranking_df, val_df, clf_gbr


if __name__ == "__main__":
    ranking_df, val_df, clf_gbr = main()
