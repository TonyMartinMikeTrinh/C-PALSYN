import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from PALSYN.petrinet_handler.simulate_to_adjacency import generate_adjacency_dict
from pm4py.objects.petri_net.importer import importer as pnml_importer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file

# Load Model
palsyn_model = DPEventLogSynthesizer()
#palsyn_model.load("models/FINAL_MODELS/FINAL_MODELS/ROAD_FINES/LSTM_Road_Fine_u=32_e=0.1_ep=5")
#palsyn_model.load("models/FINAL_MODELS/FINAL_MODELS/HOSPITAL_BILLING/LSTM_hospital_billing_u=32_e=0.1_ep=10")
palsyn_model.load("models/FINAL_MODELS/FINAL_MODELS/SEPSIS_CASE/LSTM_Sepsis Case_u=32_e=inf_ep=10")

# === KONFIGURATION ===
#pnml_path = "normative_models/road_fine_normative_model.pnml"
#pnml_path = "normative_models/road_fine_normative_model 2.pnml"
pnml_path = "normative_models/sepsis_case_normative_model.pnml"

# === PETRI-NETZ LADEN ===
net, im, fm = pnml_importer.apply(pnml_path)

# Sample
#event_log = palsyn_model.sample(sample_size=1050, batch_size=3, petri_net=(net, im, fm))
event_log = palsyn_model.sample(sample_size=1050, batch_size=3)

event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
#xes_filename = "outputs/Road_Fine/Road_Fine_log-5_simulation.xes"
#xes_filename = "outputs/Road_Fine/Road_Fine_log-10_transiton_matrix.xes"

#xes_filename = "outputs/Road_Fine/hospital_billing-1000_simulation.xes"
#xes_filename = "outputs/Road_Fine/hospital_billing-1000_transiton_matrix.xes"

xes_filename = "outputs/Sepsis/final/Sepsis_log_1050.xes"
#xes_filename = "outputs/Sepsis/final/Sepsis_log_1050_simulation.xes"
#xes_filename = "outputs/Sepsis/final/Sepsis_log_1050_transition_list.xes"

pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)


alignments = pm4py.conformance_diagnostics_alignments(event_log_xes, net, im, fm)

fitness_values = [alignment.get("fitness") for alignment in alignments]
fitness = sum(fitness_values) / len(fitness_values)

#print("align: ", alignments)
#print("FV ",fitness_values)
print("F ",fitness)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)

df.to_excel("outputs/Sepsis/final/Sepsis_log_1050.xlsx", index=False)
#df.to_excel("outputs/Sepsis/final/Sepsis_log_1050_simulation.xlsx", index=False)
#df.to_excel("outputs/Sepsis/final/Sepsis_log_1050_transition.xlsx", index=False)


#df.to_excel("outputs/Road_Fine_log_1000_simulation.xlsx", index=False)
#df.to_excel("outputs/Road_Fine_log_1000_transiton_matrix.xlsx", index=False)

#df.to_excel("outputs/hospital_billing_1000_simulation.xlsx", index=False)
#df.to_excel("outputs/hospital_billing_1000_transiton_matrix.xlsx", index=False)
