from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.semantics import enabled_transitions
from pm4py.objects.petri_net.utils import petri_utils
from copy import deepcopy

# === KONFIGURATION ===
pnml_path = "normative_models/sepsis_case_normative_model.pnml"
no_traces = 1000
max_trace_length = 100
debug = False

from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from copy import deepcopy
import random

# Beispiel-Setup
net, im, fm = pnml_importer.apply("normative_models/sepsis_case_normative_model.pnml")

# Konfiguration
max_steps = 100
marking = deepcopy(im)
trace = []
stochastic_map = {
    "CRP": 0.2,
    "Leucocytes": 0.2,
    "LacticAcid": 0.2,
    "IV Antibiotics": 0.15,
    "IV Liquid": 0.15,
    "Admission NC": 0.1,
    "Return ER": 0.1,
    "ER Triage": 0.1,
    "ER Sepsis Triage": 0.1,
    "Admission IC": 0.1,
    "Release A": 0.1,
    "Release B": 0.1,
    "Release C": 0.1,
    "Release D": 0.1,
    "Release E": 0.1,
    "Transfer NC": 0.1,
    "ER Registration": 0.3
}

# Schrittweise Simulation
for step in range(max_steps):
    enabled = enabled_transitions(net, marking)
    if not enabled:
        print("🔚 Keine weiteren Transitionen aktiviert.")
        break

    # Nur sichtbare Transitionen wählen
    visible = [t for t in enabled if t.label]
    if not visible:
        chosen = random.choice(enabled)  # z.B. silent
    else:
        weights = [stochastic_map.get(t.label, 0.01) for t in visible]
        chosen = random.choices(visible, weights=weights, k=1)[0]

        # Logging
        trace.append(chosen.label)

        # Wahrscheinlichkeiten z. B. dynamisch anpassen
        stochastic_map[chosen.label] *= 0.95

    marking = execute(chosen, net, marking)

print("📋 Trace:", " → ".join(trace))
print("📍 Endmarkierung:", {p.name: c for p, c in marking.items()})
