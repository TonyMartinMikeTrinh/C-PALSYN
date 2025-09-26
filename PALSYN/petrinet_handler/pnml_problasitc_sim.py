import random
from copy import deepcopy
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.semantics import enabled_transitions

# === KONFIGURATION ===
pnml_path = "normative_models/sepsis_case_normative_model.pnml"
max_trace_length = 50
num_traces = 5  # Für Debug weniger

transition_probs = {
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
    "ER Registration": 0.3,  # WICHTIG!
}

# === NETZ LADEN ===
net, im, fm = pnml_importer.apply(pnml_path)

# === STILLE TRANSITIONEN AUSFÜHREN ===
def fire_silent_transitions(net, marking):
    changed = True
    while changed:
        changed = False
        enabled = enabled_transitions(net, marking)
        silent_transitions = [t for t in enabled if not t.label]
        for t in silent_transitions:
            print(f"⚙️  Führe stille Transition: {t.name}")
            petri_utils.execute(t, marking)
            changed = True

# === EINZELTRACE SIMULIEREN ===
def simulate_trace_with_probabilities(net, initial_marking, transition_probs, max_length=50):
    marking = deepcopy(initial_marking)
    trace = []
    first = True

    for step in range(max_length):
        print(f"\n🔁 Schritt {step+1}")
        fire_silent_transitions(net, marking)

        enabled = enabled_transitions(net, marking)
        print("🔎 Aktivierbare Transitionen:")
        for t in enabled:
            print(f"   - {t.label or '(silent)'} ({t.name})")

        # Sonderfall: ER Registration beim ersten Schritt bevorzugt
        if first:
            first = False
            t = next((t for t in enabled if t.label == "ER Registration"), None)
            if t:
                print("🚀 ERZWINGE Start mit 'ER Registration'")
                petri_utils.execute(t, marking)
                trace.append(t.label)
                continue

        enabled_labeled = [t for t in enabled if t.label and t.label in transition_probs]
        if not enabled_labeled:
            print("❌ Keine benannte Transition mehr verfügbar.")
            break

        probs = [transition_probs[t.label] for t in enabled_labeled]
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(enabled_labeled, weights=probs, k=1)[0]
        print(f"🎯 Ausgewählte Transition: {chosen.label} ({chosen.name})")
        petri_utils.execute(chosen, marking)
        trace.append(chosen.label)

    fire_silent_transitions(net, marking)
    return trace, marking

# === MEHRERE TRACES SIMULIEREN ===
all_traces = []
for i in range(num_traces):
    print(f"\n\n=== 🚀 TRACE {i+1} ===")
    trace, end_marking = simulate_trace_with_probabilities(net, im, transition_probs, max_trace_length)
    all_traces.append((trace, end_marking))

# === ERGEBNISSE AUSGEBEN ===
for i, (trace, marking) in enumerate(all_traces, 1):
    print(f"\n📌 Trace {i}:")
    print(" → ".join(trace) if trace else "∅ (leer)")
    print("📍 Endmarkierung:", {p.name: c for p, c in marking.items()})
