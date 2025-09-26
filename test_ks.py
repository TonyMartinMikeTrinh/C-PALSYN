import pandas as pd
import numpy as np

def lengths_to_prob_series(csv_path: str) -> pd.Series:
    """
    Read a CSV with columns: trace_length, count
    Return a probability Series indexed by integer trace_length.
    """
    df = pd.read_csv(csv_path, usecols=["trace_length", "count"])
    df["trace_length"] = pd.to_numeric(df["trace_length"], errors="coerce").astype("Int64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    df = df.dropna(subset=["trace_length"])
    grouped = df.groupby(df["trace_length"].astype(int))["count"].sum()
    total = grouped.sum()
    probs = (grouped / total) if total > 0 else grouped
    probs = probs.sort_index()
    probs.name = "p"
    return probs

def ks_distance_discrete(real: pd.Series, synth: pd.Series) -> float:
    """
    KS distance for two discrete, numeric distributions (support = integers).
    Series index = support values (ints), values = probabilities (sum≈1).
    """
    # vereinheitlichte, sortierte Stützmenge
    support = np.array(sorted(set(real.index).union(set(synth.index))), dtype=int)

    p1 = np.array([float(real.get(k, 0.0)) for k in support], dtype=np.float64)
    p2 = np.array([float(synth.get(k, 0.0)) for k in support], dtype=np.float64)

    # Defensiv renormalisieren
    if p1.sum() > 0 and not np.isclose(p1.sum(), 1.0): p1 /= p1.sum()
    if p2.sum() > 0 and not np.isclose(p2.sum(), 1.0): p2 /= p2.sum()

    F1 = np.cumsum(p1)
    F2 = np.cumsum(p2)
    return float(np.max(np.abs(F1 - F2)))

real = lengths_to_prob_series("road_fines_plot/length_real.csv")
uncond = lengths_to_prob_series("road_fines_plot/length_unconditional.csv")
tlist = lengths_to_prob_series("road_fines_plot/length_transition_list.csv")
sim = lengths_to_prob_series("road_fines_plot/length_simulation.csv")

print("KS real vs unconditional:", ks_distance_discrete(real, uncond))
print("KS real vs transition_list:", ks_distance_discrete(real, tlist))
print("KS real vs simulation:", ks_distance_discrete(real, sim))



