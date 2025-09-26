#from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.analysis import check_soundness
# Lies ein Petri-Netz aus einer .pnml-Datei ein

#net, initial_marking, final_marking = pnml_importer.apply("normative_models/hospital_billing_normative_model.pnml")
#net, initial_marking, final_marking = pnml_importer.apply("normative_models/road_fines_normative_model.pnml")
net, initial_marking, final_marking = pnml_importer.apply("normative_models/sepsis_case_normative_model.pnml")

import pm4py# Reads a Petri net from a filenet,
from split_join_analyzer import SplitJoinAnalyzerPM4Py
# Soundness-Check durchführen
#soundness_result = check_soundness(net, initial_marking, final_marking)
from transition_depths import compute_structural_transition_depths, find_direct_input_places_for_and_joins, find_direct_output_places_for_and_splits, find_and_split_to_join_paths

#print(compute_structural_transition_depths("normative_models/sepsis_case_normative_model.pnml"))
df_transitions = compute_structural_transition_depths("normative_models/sepsis_case_normative_model.pnml")
print(df_transitions)
df_needed_places = find_direct_input_places_for_and_joins("normative_models/sepsis_case_normative_model.pnml", df_transitions)
print(df_needed_places)
df_out_places = find_direct_output_places_for_and_splits("normative_models/sepsis_case_normative_model.pnml", df_transitions)
print(df_out_places)

matches = find_and_split_to_join_paths(net, df_transitions)
print("Realistisch erlaufene AND-Split → AND-Join Matches:")
for s, j in matches:
    print(f"{s} → {j}")


# Liste aller ANDSplit-Transitions (einmalig)
and_split_transitions = df_out_places["ANDSplit_Transition"].unique().tolist()

# Liste aller daraus entstehenden Places (kann mehrfach vorkommen, aber hier eindeutig)
out_places = df_out_places["out_place"].unique().tolist()

print("AND-Split-Transitions:", and_split_transitions)
print("Resultierende Places:", out_places)

out_places = set(
    p for p in net.places if p.name in df_out_places["out_place"].unique()
)

# Liste aller ANDJoin-Transitions (einmalig)
and_join_transitions = df_needed_places["ANDJoin_Transition"].unique().tolist()

# Liste aller benötigten Places (direkte Inputs, einmalig)
in_places = df_needed_places["needed_Place"].unique().tolist()

print("AND-Join-Transitions:", and_join_transitions)
print("Benötigte Places:", in_places)

# Konvertiere Place-Namen zu echten Place-Objekten
in_places = set(
    p for p in net.places if p.name in df_needed_places["needed_Place"].unique()
)


print("Init: ", initial_marking)
print("End: ", final_marking)
#print(net.transitions)
#print("Net: ", net)

print(net.arcs)

from collections import defaultdict

transition_to_places = defaultdict(list)
for arc in net.arcs:
    if isinstance(arc.source, net.__class__.Transition):
        transition_to_places[arc.source].append(arc.target)

place_to_visible_transitions = defaultdict(set)

# baue place -> sichtbare Transitions
def collect_visible_from_place(place, visited_places=None):
    if visited_places is None:
        visited_places = set()
    if place in visited_places:
        return set()
    if place in final_marking:
        return {"END"}
    if place in in_places:
        print("Join Label ", place)
        return {("AND-Join", place)}
    visited_places.add(place)
    result = set()
    for arc in net.arcs:
        if arc.source == place and isinstance(arc.target, net.__class__.Transition):
            t = arc.target
            print(arc.source,"  ", arc.target)
            if t.label is not None: 
                result.add(t.label)
            else:
                # stille Transition -> Output-Places weiterverfolgen
                for arc2 in net.arcs:
                    if arc2.source == t and isinstance(arc2.target, net.__class__.Place):
                        print(arc2.source, "    ", arc2.target)
                        result |= collect_visible_from_place(arc2.target, visited_places)
    return result

# Fülle Mapping
for place in net.places:
    print("Start: ", place)
    place_to_visible_transitions[place] = collect_visible_from_place(place)

print(place_to_visible_transitions)

visible_transition_map = defaultdict(set)

start_transitions = set()
for place in initial_marking:
    start_transitions |= collect_visible_from_place(place)
visible_transition_map["START"] = start_transitions

for t in net.transitions:
    if t.label is None:
        continue
    places = transition_to_places[t]
    for p in places:
        visible_transition_map[t.label] |= place_to_visible_transitions.get(p, set())

for src, targets in visible_transition_map.items():
    formatted = ', '.join(sorted(str(t) for t in targets))
    print(f"{src}: {formatted}")

# exausitv