import os
import pm4py
from typing import Optional, Tuple
from pm4py.objects.petri_net.importer import importer as pnml_importer

from PALSYN.synthesizer import DPEventLogSynthesizer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file

# ------------------------------------------------------------
# Helper: führt n Runs mit/ohne Petri-Net durch
# ------------------------------------------------------------
def run_experiment(
    *,
    title: str,
    model_path: str,
    pnml_path: str,
    sample_size: int,
    batch_size: int,
    repeats: int,
    out_dir: str,
):
    print(f"\n=== {title}: Lade Modell ===")
    palsyn_model = DPEventLogSynthesizer()
    palsyn_model.load(model_path)

    print(f"=== {title}: Lade PNML ===")
    net, im, fm = pnml_importer.apply(pnml_path)

    os.makedirs(out_dir, exist_ok=True)

    # 3 Modi: ohne Petri, mit Simulation, mit Transition-List
    modes = [
        ("no-petri", None, "none"),              # ohne Petri-Net
        ("simulation", (net, im, fm), "simulation"),
        ("transition-list", (net, im, fm), "transition-list"),
    ]

    for mode_name, petri_net, mode in modes:
        print(f"\n>>> {title}: Starte {repeats} Runs im Mode = {mode_name}")

        for i in range(1, repeats + 1):
            print(f"\n--- {title} Run {i}/{repeats} [{mode_name}] ---")

            # Sample
            if petri_net is None:
                event_df = palsyn_model.sample(
                    sample_size=sample_size,
                    batch_size=batch_size,
                )
            else:
                event_df = palsyn_model.sample(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    petri_net=petri_net,
                    mode=mode,  # "simulation" oder "transition-list"
                )

            # EventLog konvertieren
            event_log = pm4py.convert_to_event_log(event_df)

            # Dateinamen
            xes_path = os.path.join(out_dir, f"{title}_log_{sample_size}_run-{i}_{mode_name}.xes")
            xlsx_path = os.path.join(out_dir, f"{title}_log_{sample_size}_run-{i}_{mode_name}.xlsx")

            # XES speichern & säubern
            pm4py.write_xes(event_log, xes_path)
            clean_xes_file(xes_path, xes_path)

            # Fitness nur berechnen, wenn PN existiert
            #if petri_net is not None:
            #    alignments = pm4py.conformance_diagnostics_alignments(event_log, net, im, fm)
            #    fitness_values = [al.get("fitness") for al in alignments if al is not None]
            #    fitness = sum(fitness_values) / len(fitness_values) if fitness_values else float("nan")
            #    print(f"{title} Run {i} [{mode_name}]: Fitness = {fitness:.4f}")

            # XLSX Export
            #df_out = pm4py.convert_to_dataframe(event_log)
            #df_out["time:timestamp"] = df_out["time:timestamp"].astype(str)
            #df_out.to_excel(xlsx_path, index=False)

    print(f"\n=== {title}: Alle Runs fertig. Ergebnisse in: {out_dir} ===")


# =========================
# Konfiguration & Aufrufe
# =========================

# Road Fines: 11,390 – 5× pro Mode
'''
run_experiment(
    title="Road_Fine",
    model_path="models/FINAL_MODELS/FINAL_MODELS/ROAD_FINES/LSTM_Road_Fine_u=32_e=inf_ep=5",
    pnml_path="normative_models/road_fine_normative_model.pnml",
    sample_size=11390,
    batch_size=3,
    repeats=5,
    out_dir="outputs/Road_Fine/evaluation",
)
'''
# Sepsis Case: 1,050 – 5× pro Mode
run_experiment(
    title="Sepsis",
    model_path="models/FINAL_MODELS/FINAL_MODELS/SEPSIS_CASE/LSTM_Sepsis Case_u=32_e=inf_ep=10",
    pnml_path="normative_models/sepsis_case_normative_model.pnml",
    sample_size=1050,
    batch_size=3,
    repeats=5,
    out_dir="outputs/Sepsis/evaluation",
)
