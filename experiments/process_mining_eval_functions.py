import numpy as np
import pm4py
import pandas as pd
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.earth_mover_distance.variants.pyemd import normalized_levensthein
from pyemd import emd


# Maximal viele Zeilen und Spalten anzeigen
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Maximale Breite der Ausgabe erhöhen
pd.set_option("display.width", 0)

# Abschneiden langer Inhalte vermeiden
pd.set_option("display.max_colwidth", None)

def event_distribution(log):
    """Return the count of each event type in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each event type in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df['concept:name'].value_counts()

    prob_data = count_data / count_data.sum()
    return prob_data


def calculate_trace_length_distribution(log):
    """Return the count of each trace length in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each trace length in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df['case:concept:name'].value_counts()
    # Get the count of each trace length
    count_data = count_data.value_counts()
    # Sort the series by the index
    count_data = count_data.sort_index()

    print(count_data)

    # Make index numeric
    #count_data.index = pd.to_numeric(count_data.index)
    prob_data = count_data / count_data.sum()
    return prob_data


def calc_hellinger(real_data, synthetic_data, input_type = "column"):
    """
    Calculate Hellinger distance between two distributions or columns.

    Args:
        data1: First distribution/column (pandas Series or column)
        data2: Second distribution/column (pandas Series or column)

    Returns:
        float: Hellinger distance
    """
    if input_type == "column":
        dist1 = real_data.value_counts()
        dist2 = synthetic_data.value_counts()
    else:
        dist1 = real_data
        dist2 = synthetic_data

    # Align distributions
    all_indices = sorted(set(dist1.index) | set(dist2.index))
    dist1_aligned = pd.Series(0, index=all_indices)
    dist2_aligned = pd.Series(0, index=all_indices)

    dist1_aligned[dist1.index] = dist1
    dist2_aligned[dist2.index] = dist2

    # Convert to probabilities
    p1 = dist1_aligned / dist1_aligned.sum()
    p2 = dist2_aligned / dist2_aligned.sum()

    return np.sqrt(np.sum((np.sqrt(p1.values) - np.sqrt(p2.values)) ** 2)) / np.sqrt(2)


def calculate_throughput_time(log):
    """Calculate the throughput time of a PM4Py event log. The throughput time is defined as the time between the
    first and the last event of a trace.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The throughput time of the event log.
    :rtype: float
    """
    # Remove all timestamps that are NaN
    df = pm4py.convert_to_dataframe(log)
    df = df.dropna(subset=['time:timestamp'])
    # Transform df back to log
    log = pm4py.convert_to_event_log(df)

    all_case_durations = pm4py.get_all_case_durations(log)

    return all_case_durations


def calculate_fitness(alignments):
    """Calculate the fitness of an event log according to the calculated alignments.

    :param alignments: A list of PM4PY alignments.
    :type alignments: List[object]
    :return: The average fitness of the alignments.
    :rtype: float
    """
    fitness_values = [alignment.get("fitness") for alignment in alignments]
    fitness = sum(fitness_values) / len(fitness_values)

    return fitness


def calculate_earth_mover_distance(real_log, synthetic_log):
    """Calculate the earth mover distance between two event logs. The earth mover distance is defined as the
    minimum cost of turning one distribution into the other.

    :param real_log: The real event log.
    :type real_log: pm4py.objects.log.log.EventLog
    :param synthetic_log: The synthetic event log.
    :type synthetic_log: pm4py.objects.log.log.EventLog
    :return: The earth mover distance between the two event logs.
    :rtype: float
    """
    real_language = variants_module.get_language(real_log)
    synthetic_language = variants_module.get_language(synthetic_log)
    earth_mover_distance = emd_evaluator.apply(synthetic_language, real_language)

    return earth_mover_distance


def calculate_petri_nets(log, threshold):
    """Discover Petri nets using inductive and heuristic mining algorithms.

    :param threshold:
    :param log: A process event log.
    :type log: pm4py.objects.log.log.EventLog
    :return: A dictionary containing the discovered Petri nets, initial markings, and final markings.
    :rtype: dict
    """
    net_inductive, initial_marking_inductive, final_marking_inductive = pm4py.discover_petri_net_inductive(log)
    net_heuristics, initial_marking_heuristics, final_marking_heuristics = \
        pm4py.discover_petri_net_heuristics(log, dependency_threshold=threshold)
    petri_net_list = [
                      [net_inductive, initial_marking_inductive, final_marking_inductive],
                      [net_heuristics, initial_marking_heuristics, final_marking_heuristics]]
    petri_net_list_names = ["Inductive",
                            "Heuristics"]
    petri_net_dict = dict(zip(petri_net_list_names, petri_net_list))
    return petri_net_dict


def compare_logs(real_event_log, synthetic_event_log, threshold):
    """Compare the fitness of a real event Log with a synthetic event log. In this case the Petri Nets discovered
    from the real even log are used to calculate the alignments of the synthetic event log.

    :param real_event_log: The real event log.
    :type real_event_log: pm4py.objects.log.log.EventLog
    :param synthetic_event_log: The synthetic event log.
    :type synthetic_event_log: pm4py.objects.log.log.EventLog
    :param threshold: The threshold for the heuristic mining algorithm.
    :type threshold: float
    :return: Dictionary containing results for both real and synthetic data with prefixed keys
    """
    petri_net_dict = calculate_petri_nets(real_event_log, threshold)
    petri_net_dict_synth = calculate_petri_nets(synthetic_event_log, threshold)

    results = {}

    # Calculate metrics for real event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        prec = pm4py.precision_alignments(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        gen = generalization_evaluator.apply(real_event_log, petri_net[0], petri_net[1], petri_net[2])
        fitness = calculate_fitness(alignments)
        simp = simplicity_evaluator.apply(petri_net[0])

        results[f"real_{key}_Fitness"] = fitness
        results[f"real_{key}_Precision"] = prec
        results[f"real_{key}_Generalization"] = gen
        results[f"real_{key}_Simplicity"] = simp

    # Calculate metrics for synthetic event log
    for key, petri_net in petri_net_dict.items():
        alignments = pm4py.conformance_diagnostics_alignments(synthetic_event_log, petri_net[0], petri_net[1],
                                                              petri_net[2])
        prec = pm4py.precision_alignments(synthetic_event_log, petri_net[0], petri_net[1], petri_net[2])
        gen = generalization_evaluator.apply(synthetic_event_log, petri_net[0], petri_net[1], petri_net[2])
        fitness = calculate_fitness(alignments)
        simp = simplicity_evaluator.apply(petri_net_dict_synth[key][0])

        results[f"synthetic_{key}_Fitness"] = fitness
        results[f"synthetic_{key}_Precision"] = prec
        results[f"synthetic_{key}_Generalization"] = gen
        results[f"synthetic_{key}_Simplicity"] = simp

    return results


from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer


def calculate_emd_trace_length(real: pd.Series, synth: pd.Series) -> float:
    # get all values from both distributions
    values = sorted(set(real.index).union(synth.index))

    p1 = np.array([real.get(k, 0.0) for k in values], dtype=np.float64)
    p2 = np.array([synth.get(k, 0.0) for k in values], dtype=np.float64)

    if not np.isclose(p1.sum(), 1.0): p1 /= p1.sum()
    if not np.isclose(p2.sum(), 1.0): p2 /= p2.sum()

    # ground distance matrix (|real_value_i - synth_value_j|)
    distance_matrix = np.abs(np.subtract.outer(values, values)).astype(np.float64)
    
    return float(emd(p1, p2, distance_matrix))




# XES-Dateien (Pfad ggf. anpassen)
log_real = xes_importer.apply("example_logs/Sepsis_Cases_Event_Log.xes")

log_no_petri_net_1 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_no-petri.xes")
log_no_petri_net_2 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_no-petri.xes")
log_no_petri_net_3 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_no-petri.xes")
log_no_petri_net_4 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_no-petri.xes")
log_no_petri_net_5 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_no-petri.xes")

log_transition_list_1 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_transition-list.xes")
log_transition_list_2 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_transition-list.xes")
log_transition_list_3 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_transition-list.xes")
log_transition_list_4 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_transition-list.xes")
log_transition_list_5 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_transition-list.xes")

log_petrinet_playout_1 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-1_simulation.xes")
log_petrinet_playout_2 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-2_simulation.xes")
log_petrinet_playout_3 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-3_simulation.xes")
log_petrinet_playout_4 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-4_simulation.xes")
log_petrinet_playout_5 = xes_importer.apply("outputs/Sepsis/evaluation/Sepsis_log_1050_run-5_simulation.xes")

logs = {
    "real": log_real,

    "no_petri_net_run1": log_no_petri_net_1,
    "no_petri_net_run2": log_no_petri_net_2,
    "no_petri_net_run3": log_no_petri_net_3,
    "no_petri_net_run4": log_no_petri_net_4,
    "no_petri_net_run5": log_no_petri_net_5,

    "transition_list_run1": log_transition_list_1,
    "transition_list_run2": log_transition_list_2,
    "transition_list_run3": log_transition_list_3,
    "transition_list_run4": log_transition_list_4,
    "transition_list_run5": log_transition_list_5,

    "petrinet_playout_run1": log_petrinet_playout_1,
    "petrinet_playout_run2": log_petrinet_playout_2,
    "petrinet_playout_run3": log_petrinet_playout_3,
    "petrinet_playout_run4": log_petrinet_playout_4,
    "petrinet_playout_run5": log_petrinet_playout_5,

}

rows = []

for name, log_synth in logs.items():
    row = {"synthetic_log": name}

    print("start")
    
    pnml_path = "normative_models/sepsis_case_normative_model.pnml"

    # === PETRI-NETZ LADEN ===
    #net, im, fm = pnml_importer.apply(pnml_path)
    #alignments = pm4py.conformance_diagnostics_alignments(log_synth, net, im, fm)
    #fitness_values = [alignment.get("fitness") for alignment in alignments]
    #fitness = sum(fitness_values) / len(fitness_values)
    #row["fitness"] = fitness
    
    real_ed = event_distribution(log_real)
    synth_ed = event_distribution(log_synth)
    row["hellinger_event_distribution"] = calc_hellinger(real_ed, synth_ed, input_type="distribution")
    
    # Trace Length Distribution
    td1 = calculate_trace_length_distribution(log_real)
    td2 = calculate_trace_length_distribution(log_synth)
    row["earth_mover_distance_length_distribution"] = calculate_emd_trace_length(td1,td2)
    
    # Variant Distribution
    row["earth_mover_distance_trace_variant"] = calculate_earth_mover_distance(log_real, log_synth)

    rows.append(row)

# Als CSV speichern
df = pd.DataFrame(rows)
df.to_csv("synthetic_vs_real_evaluation_sepsis_2.csv", index=False)
print("✅ Ergebnisse gespeichert in synthetic_vs_real_evaluation_sepsis.csv")






# XES-Dateien (Pfad ggf. anpassen)
log_real = xes_importer.apply("example_logs/Road_Traffic_Fine_Management_Process_short.xes")

log_no_petri_net_1 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_no-petri.xes")
log_no_petri_net_2 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_no-petri.xes")
log_no_petri_net_3 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_no-petri.xes")
log_no_petri_net_4 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_no-petri.xes")
log_no_petri_net_5 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_no-petri.xes")

log_transition_list_1 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_transition-list.xes")
log_transition_list_2 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_transition-list.xes")
log_transition_list_3 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_transition-list.xes")
log_transition_list_4 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_transition-list.xes")
log_transition_list_5 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_transition-list.xes")

log_petrinet_playout_1 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-1_simulation.xes")
log_petrinet_playout_2 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-2_simulation.xes")
log_petrinet_playout_3 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-3_simulation.xes")
log_petrinet_playout_4 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-4_simulation.xes")
log_petrinet_playout_5 = xes_importer.apply("outputs/Road_Fine/evaluation/Road_Fine_log_11390_run-5_simulation.xes")

logs = {
    "real": log_real,

    "no_petri_net_run1": log_no_petri_net_1,
    "no_petri_net_run2": log_no_petri_net_2,
    "no_petri_net_run3": log_no_petri_net_3,
    "no_petri_net_run4": log_no_petri_net_4,
    "no_petri_net_run5": log_no_petri_net_5,

    "transition_list_run1": log_transition_list_1,
    "transition_list_run2": log_transition_list_2,
    "transition_list_run3": log_transition_list_3,
    "transition_list_run4": log_transition_list_4,
    "transition_list_run5": log_transition_list_5,

    "petrinet_playout_run1": log_petrinet_playout_1,
    "petrinet_playout_run2": log_petrinet_playout_2,
    "petrinet_playout_run3": log_petrinet_playout_3,
    "petrinet_playout_run4": log_petrinet_playout_4,
    "petrinet_playout_run5": log_petrinet_playout_5,

}

rows = []

for name, log_synth in logs.items():
    row = {"synthetic_log": name}

    print("start")
    
    pnml_path = "normative_models/road_fine_normative_model.pnml"

    # === PETRI-NETZ LADEN ===
    #net, im, fm = pnml_importer.apply(pnml_path)
    #alignments = pm4py.conformance_diagnostics_alignments(log_synth, net, im, fm)
    #fitness_values = [alignment.get("fitness") for alignment in alignments]
    #fitness = sum(fitness_values) / len(fitness_values)
    #row["fitness"] = fitness
    
    real_ed = event_distribution(log_real)
    synth_ed = event_distribution(log_synth)
    row["hellinger_event_distribution"] = calc_hellinger(real_ed, synth_ed, input_type="distribution")
    
    # Trace Length Distribution
    td1 = calculate_trace_length_distribution(log_real)
    td2 = calculate_trace_length_distribution(log_synth)
    row["earth_mover_distance_length_distribution"] = calculate_emd_trace_length(td1,td2)
    
    # Variant Distribution
    row["earth_mover_distance_trace_variant"] = calculate_earth_mover_distance(log_real, log_synth)

    rows.append(row)


# Als CSV speichern
df = pd.DataFrame(rows)
df.to_csv("synthetic_vs_real_evaluation_road_fines_2.csv", index=False)
print("✅ Ergebnisse gespeichert in synthetic_vs_real_evaluation_road_fines.csv")

