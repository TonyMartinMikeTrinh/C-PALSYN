# Dummy-Klasse zur Simulation von Transitionen
class DummyTransition:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"<Transition {self.name}>"
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return isinstance(other, DummyTransition) and self.name == other.name

# Beispielhafte Concept-Index-Map
concept_index = {
    'END': 2,
    'CRP': 3,
    'Leucocytes': 33,
    'ER Triage': 64,
    'ER Registration': 93,
    'ER Sepsis Triage': 100,
    'Admission NC': 135,
    'LacticAcid': 172,
    'IV Antibiotics': 207,
    'IV Liquid': 246,
    'Release A': 278,
    'Return ER': 323,
    'Release B': 382,
    'Admission IC': 419,
    'Release C': 461,
    'Release D': 504,
    'Release E': 596
}

# Beispielhafte stille Transitionen und ihre Labels
silent_transition_map = {
    DummyTransition("n32"): {"Release E", "Release A", "Release B", "Release D", "END", "Release C", "Return ER"},
    DummyTransition("n49"): {"Release E", "Release A", "Release B", "Release D", "END", "Release C", "Return ER"},
    DummyTransition("n34"): {"Release E", "Release A", "Release B", "Release D", "END", "Release C", "Return ER"},
}

# Diese Liste entspricht enabled_silent (wie bei dir im Code): Tuple[Transition, None]
enabled_silent = [
    (DummyTransition("n32"), None),
    (DummyTransition("n49"), None),
    (DummyTransition("n34"), None)
]

# Testfunktion: Mapping auf Concept Index
enabled_silent_indexes = [
    [concept_index[label] for label in silent_transition_map.get(silent_t, set()) if label in concept_index]
    for silent_t, _ in enabled_silent
]

# Testausgabe
print("✅ Enabled silent mapped concept indexes:")
print(enabled_silent_indexes)