from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py
from pm4py.statistics.variants.log import get as variants_module
# ------------------------
# 1. List of log paths
# ------------------------
log_paths = [

    "example_logs/Sepsis_Cases_Event_Log.xes"

]

# ------------------------
# 2. Process all logs and collect CRP and Leucocytes values
# ------------------------
event_rows = []
length_rows = []

def event_distribution(log):
    df = pm4py.convert_to_dataframe(log)
    count_data = df['concept:name'].value_counts()
    result = count_data.reset_index(drop=False)
    result.columns = ["concept:name", "count"]
    print(result)
    return result

def calculate_trace_length_distribution(log):

    df = pm4py.convert_to_dataframe(log)
    count_data = df['case:concept:name'].value_counts()
    # Get the count of each trace length
    count_data = count_data.value_counts()
    # Sort the series by the index
    count_data = count_data.sort_index()
    result = count_data.reset_index(drop=False)
    result.columns = ["trace_length", "count"]
    print(result)
    return result



event_rows = []
length_rows = []

for path in log_paths:
    log = xes_importer.apply(path)



    event_rows.append(event_distribution(log))

    length_rows.append(calculate_trace_length_distribution(log))

# ------------------------
# 3. Create DataFrames
# ------------------------
merged_events = pd.concat(event_rows, axis=0)        # untereinander stapeln
df_event = merged_events.groupby("concept:name", as_index=False)["count"].sum()

# Trace lengths: addiere alle Series
merged_lengths = pd.concat(length_rows, axis=1).fillna(0).sum(axis=1).astype(int)
df_length = (
    pd.concat(length_rows, axis=0, ignore_index=True)     # alle Verteilungen untereinander
      .groupby("trace_length", as_index=False)["count"].sum()
      .sort_values("trace_length")
)

#df_event.to_csv("road_fines_plot/event_real.csv", index=False)
df_length.to_csv("sepsis_plot/length_real.csv", index=False)

# ------------------------
# 5. Plot combined KDE curves (all logs together)
# ------------------------
sns.set(style="whitegrid")


event_dist = df_event["concept:name"].value_counts()  # falls noch nicht aggregiert

plt.figure(figsize=(12,6))
sns.barplot(
    x="concept:name",
    y="count",
    data=df_event,
    color="skyblue",
    edgecolor="black"
)
plt.xticks(rotation=90)
plt.title("Event Distribution")
plt.xlabel("Event")
plt.ylabel("Count")
plt.tight_layout()
#plt.savefig("road_fines_plot/event_unconditional.png")
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(
    data=df_length,
    x="trace_length",
    weights=df_length["count"] / df_length["count"].sum(),  # normiert = Wahrscheinlichkeiten
    bins=len(df_length),
    discrete=True,
    color="skyblue",
    edgecolor="black"
)

plt.title("Trace Length Distribution")
plt.xlabel("Trace Length")
plt.ylabel("Probability")
plt.tight_layout()
#plt.savefig("road_fines_plot/length_unconditional.png")
plt.show()



"""
"example_logs/Sepsis_Cases_Event_Log.xes"


    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_no-petri.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_no-petri.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_no-petri.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_no-petri.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_no-petri.xes",


    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_transition-list.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_transition-list.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_transition-list.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_transition-list.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_transition-list.xes",

    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_simulation.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_simulation.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_simulation.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_simulation.xes",
    "outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_simulation.xes",

"example_logs/Road_Traffic_Fine_Management_Process_short.xes"

"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_no-petri.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_no-petri.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_no-petri.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_no-petri.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_no-petri.xes",

"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_transition-list.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_transition-list.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_transition-list.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_transition-list.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_transition-list.xes",

"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_simulation.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_simulation.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_simulation.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_simulation.xes",
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_simulation.xes",
"""