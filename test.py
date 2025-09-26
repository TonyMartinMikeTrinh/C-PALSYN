import pandas as pd

# CSV einlesen
df = pd.read_csv("synthetic_vs_real_evaluation_road_fines.csv")

# echte Logs (z. B. 'real') rausfiltern, wenn du nur synthetische willst
df_synth = df[df["synthetic_log"] != "real"].copy()

# Gruppennamen extrahieren (alles vor '_run')
df_synth["group"] = df_synth["synthetic_log"].str.replace(r"_run\d+", "", regex=True)

# Mittelwerte je Gruppe berechnen
df_means = df_synth.groupby("group").mean(numeric_only=True).reset_index()

df_means.to_csv("synthetic_log_group_means_rf.csv", index=False)

print("Saved group averages to synthetic_log_group_means_rf.csv")
