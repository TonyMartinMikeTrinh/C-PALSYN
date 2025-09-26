import pandas as pd
import numpy as np
from pyemd import emd
from decimal import Decimal, ROUND_HALF_UP




def calculate_emd_trace_length(real: pd.Series, synth: pd.Series) -> float:
    # get all values from both distributions
    values = sorted(set(real.index).union(synth.index))

    p1 = np.array([real.get(k, 0.0) for k in values], dtype=np.float64)
    p2 = np.array([synth.get(k, 0.0) for k in values], dtype=np.float64)

    if not np.isclose(p1.sum(), 1.0): p1 /= p1.sum()
    if not np.isclose(p2.sum(), 1.0): p2 /= p2.sum()

    # ground distance matrix (|real_value_i - synth_value_j|)
    distance_matrix = np.abs(np.subtract.outer(values, values)).astype(np.float64)
    print(distance_matrix)
    
    return float(emd(p1, p2,distance_matrix))

def normalize_df(df: pd.DataFrame):
    # runden
    vals = pd.to_numeric(df["value"], errors="coerce").to_numpy(np.float64)
    df["value"] = np.sign(vals) * (np.floor(np.abs(vals) * 10.0 + 0.5) / 10.0)

    out = (
        df["value"]
        .value_counts(normalize=True)      # Anteile
        .rename("percentage")
        .reset_index()
        .rename(columns={"index": "value"})
    )

    # sortieren
    out = out.sort_values("value").reset_index(drop=True)

    print(out)
    # ggf. speichern
    out.to_csv("value_percentage.csv", index=False)
    return out

# XES-Dateien (Pfad ggf. anpassen)
total_payment_amount_real = normalize_df(pd.read_csv("total_payment_amount_real.csv"))

total_payment_amount_unconditional = normalize_df(pd.read_csv("total_payment_amount_unconditional.csv"))
total_payment_amount_transition_list = normalize_df(pd.read_csv("total_payment_amount_transition-list.csv"))
total_payment_amount_simulation = normalize_df(pd.read_csv("total_payment_amount_simulation.csv"))

logs = {
    "real": total_payment_amount_real,
    "unconditional": total_payment_amount_unconditional,
    "transition_list": total_payment_amount_transition_list,
    "total_payment_amount_simulation": total_payment_amount_simulation
}

rows = []

for name, log_synth in logs.items():
    row = {"synthetic_log": name}

    print("start")
    
    row["earth_mover_distance_length_distribution"] = calculate_emd_trace_length(
    total_payment_amount_real.set_index("value")["percentage"],
    log_synth.set_index("value")["percentage"]
    )
    print(row["earth_mover_distance_length_distribution"])
    rows.append(row)

# Als CSV speichern
df = pd.DataFrame(rows)
df.to_csv("total_payment_amount_attribute_emd.csv", index=False)
print("✅ Ergebnisse gespeichert in total_payment_amount_attribute_emd.csv")


