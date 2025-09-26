from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.semantics import ClassicSemantics
import random

net, im, fm = pnml_importer.apply("normative_models/sepsis_case_normative_model.pnml")

marking = im.copy()
semantics = ClassicSemantics()
trace = []

MAX_STEPS = 100
for step in range(MAX_STEPS):
    enabled = semantics.enabled_transitions(net, marking)
    print(f"Schritt {step+1}: aktiviert: {[t.label for t in enabled]}")

    if not enabled:
        print("Keine Transitionen mehr aktiv.")
        break

    # Eine Transition wählen
    t = random.choice(list(enabled))
    print("Choice: ", t)
    print("List: ", enabled)
    # Transition ausführen
    marking = semantics.execute(t, net, marking)

    if t.label:  # Nur sichtbare Transitionen speichern
        trace.append(t.label)
    else:
        trace.append(f"τ({t.name})")

    if marking == fm:
        print("Finalmarkierung erreicht.")
        break

print("Simulierter Pfad:", " → ".join(trace))
