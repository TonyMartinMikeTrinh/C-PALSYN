from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py

log = xes_importer.apply("outputs/Sepsis/Sepsis_log_1050_simulation.xes")

df = pm4py.convert_to_dataframe(log)
df["time:timestamp"] = df["time:timestamp"].astype(str)

df.to_excel("outputs/Sepsis/Sepsis_log_1050_simulation.xlsx", index=False)
