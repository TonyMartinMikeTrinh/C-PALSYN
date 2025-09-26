from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 1. List of log paths
# ------------------------
log_paths = [

    "outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_simulation.xes",
    "outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_simulation.xes",
    "outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_simulation.xes",
    "outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_simulation.xes",
    "outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_simulation.xes"

]

# ------------------------
# 2. Process all logs and collect CRP and Leucocytes values
# ------------------------
crp_rows = []
leuco_rows = []

for path in log_paths:
    log = xes_importer.apply(path)
    
    for trace in log:
        case_id = trace.attributes.get("concept:name", None)

        for event in trace:
            timestamp = event.get("time:timestamp")
            activity = event.get("concept:name")

            if "amount" in event:
                crp_rows.append({
                    "case_id": case_id,
                    "value": event["amount"],
                    "timestamp": timestamp,
                    "activity": activity,
                    "source_log": path
                })

            if "totalPaymentAmount" in event:
                leuco_rows.append({
                    "case_id": case_id,
                    "value": event["totalPaymentAmount"],
                    "timestamp": timestamp,
                    "activity": activity,
                    "source_log": path
                })

# ------------------------
# 3. Create DataFrames
# ------------------------
df_crp = pd.DataFrame(crp_rows)
df_leucocytes = pd.DataFrame(leuco_rows)

# ------------------------
# 4. Save combined values as CSV
# ------------------------
df_crp.to_csv("fine_amount_simulation.csv", index=False)
df_leucocytes.to_csv("total_payment_amount_simulation.csv", index=False)

# ------------------------
# 5. Plot combined KDE curves (all logs together)
# ------------------------
sns.set(style="whitegrid")

# CRP KDE
plt.figure()
sns.kdeplot(df_crp["value"], fill=True)
plt.title("Density Curve of fine amount")
plt.xlabel("amount")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("fine_amount_simulation.png")
plt.show()

# Leucocytes KDE
plt.figure()
sns.kdeplot(df_leucocytes["value"], fill=True)
plt.title("Density Curve of total payement amount")
plt.xlabel("total fine payement amount")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("total_payment_amount_simulation.png")
plt.show()



"""
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



"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_no-petri.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_no-petri.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_no-petri.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_no-petri.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_no-petri.xes"

"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_transition-list.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_transition-list.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_transition-list.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_transition-list.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_transition-list.xes"

"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_simulation.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_simulation.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_simulation.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_simulation.xes"
"outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_simulation.xes"
"""