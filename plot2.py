import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ------------------------
# 1. Define file paths
# ------------------------

event_files = [
    ("sepsis_plot/event_real.csv", "Event Real Log"),
    ("sepsis_plot/event_unconditional.csv", "Event Unconditional Log"),
    ("sepsis_plot/event_transition_list.csv", "Event Transition List Log"),
    ("sepsis_plot/event_simulation.csv", "Event Simulation Log")
]

length_files = [
    ("sepsis_plot/length_real.csv", "Trace Length Real Log"),
    ("sepsis_plot/length_unconditional.csv", "Trace Length Unconditional Log"),
    ("sepsis_plot/length_transition_list.csv", "Trace Length Transition List Log"),
    ("sepsis_plot/length_simulation.csv", "Trace Length Simulation Log")
]

hue_order_lengths = [
    "Trace Length Real Log",
    "Trace Length Unconditional Log",
    "Trace Length Transition List Log",
    "Trace Length Simulation Log",
]

selected_events = [
    "Add penalty",
    "Create Fine",
    "Insert Fine Notification",
    "Payment",
    "Send Fine",
    "Send for Credit Collection",
]

# ------------------------
# 1. Event Probability Comparison
# ------------------------
# Alle Events einlesen und kombinieren
df_events_all = []
for file, label in event_files:
    df = pd.read_csv(file)
    if not df.empty and "count" in df.columns:
        df["source"] = label
        # falls du lieber normalisieren willst:
        df["probability"] = df["count"] / df["count"].sum()
        df_events_all.append(df)

df_events_all = pd.concat(df_events_all, ignore_index=True)

plt.figure(figsize=(14, 7))
sns.barplot(
    data=df_events_all,
    x="concept:name",
    y="probability",   # oder "count", wenn du absolute Werte willst
    hue="source",
    order=selected_events
)
plt.xticks(rotation=90)
plt.title("Event Distribution Comparison (4 Logs)")
plt.xlabel("Event")
plt.ylabel("Probability")  # ändere auf "Count", falls absolute Werte
plt.legend(title="Log Source")
plt.tight_layout()
plt.savefig("sepsis_plot/event_comparison.png")
plt.show()


# ------------------------
# 2. Trace Length Probability Comparison
# ------------------------
plt.figure(figsize=(10, 6))
for file, label in length_files:
    df = pd.read_csv(file)
    if not df.empty and "count" in df.columns:
        # expandiere counts zu values
        expanded = df.loc[df.index.repeat(df["count"])]
        expanded = expanded.rename(columns={"trace_length": "value"})
        sns.kdeplot(expanded["value"], fill=False, label=label)

plt.title("Trace Length Probability Comparison (4 Logs)")
plt.xlabel("Trace Length")
plt.ylabel("Probability")
plt.xlim(0, 30)   # anpassen an deine Daten
plt.legend()
plt.tight_layout()
plt.savefig("sepsis_plot/trace_length_Probability_comparison.png")
plt.show()

"""

df_lengths_all = []
for file, label in length_files:
    df = pd.read_csv(file)
    if not df.empty and "count" in df.columns:
        df["source"] = label
        df["probability"] = df["count"] / df["count"].sum()  # Normierung
        df_lengths_all.append(df)

df_lengths_all = pd.concat(df_lengths_all, ignore_index=True)


plt.figure(figsize=(14, 7))
sns.barplot(
    data=df_lengths_all,
    x="trace_length",
    y="probability",   # oder "count" für absolute Werte
    hue="source",
    hue_order=hue_order_lengths
)
plt.xticks(rotation=90)
plt.title("Trace Length Distribution Comparison (4 Logs)")
plt.xlabel("Trace Length")
plt.ylabel("Probability")
plt.xlim(0, 6)   # anpassen an deine Daten
plt.legend(title="Log Source")
plt.tight_layout()
plt.savefig("sepsis_plot/trace_length_comparison.png")
plt.show()

"""