import random
from copy import deepcopy
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.semantics import enabled_transitions
from pm4py.objects.petri_net.obj import Marking

# === 1. Netz laden ===
net, im_raw, _ = pnml_importer.apply("normative_models/test.pnml")

# === 2. Initialmarkierung setzen (über Place-Name "source") ===
im = Marking()
for p in net.places:
    if getattr(p, "name", "").lower() == "source":
        im[p] = 1
initial = [f"{p.name or p.__dict__.get('id')}:{c}" for p, c in im.items()]
print(f"\n📌 Initialmarkierung: {initial}")

# === 3. Wahrscheinlichkeiten für benannte Transitionen definieren ===
transition_probs = {
    "A": 0.6,
    "B": 0.4,
}

# === 4. Stille Transitionen automatisch feuern ===
def fire_silent_transitions(net, marking):
    changed = True
    while changed:
        changed = False
        for t in enabled_transitions(net, marking):
            if not t.label:
                petri_utils.execute(t, marking)
                changed = True

# === 5. Einen Trace simulieren ===
def simulate_trace(net, im, transition_probs, max_len=10):
    trace = []
    marking = deepcopy(im)
    for _ in range(max_len):
        fire_silent_transitions(net, marking)
        enabled = enabled_transitions(net, marking)
        enabled_labeled = [t for t in enabled if t.label and t.label in transition_probs]
        if not enabled_labeled:
            break
        weights = [transition_probs[t.label] for t in enabled_labeled]
        t_selected = random.choices(enabled_labeled, weights=weights, k=1)[0]
        petri_utils.execute(t_selected, marking)
        trace.append(t_selected.label)
    fire_silent_transitions(net, marking)
    return trace, marking

# === 6. Mehrere Traces erzeugen ===
n_traces = 5
for i in range(n_traces):
    trace, end_marking = simulate_trace(net, im, transition_probs)
    print(f"\n=== 🚀 Trace {i+1} ===")
    print(" → ".join(trace) if trace else "∅ (leer)")
    print("📍 Endmarkierung:", {p.name or p.__dict__.get("id"): c for p, c in end_marking.items()})