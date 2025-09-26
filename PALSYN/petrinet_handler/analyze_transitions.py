from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import PetriNet, Marking
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from pm4py.objects.petri_net.obj import PetriNet, Marking
from collections import defaultdict
from typing import List, Dict, Set, Tuple

def analyze_transitions_with_end_labels(
    net: PetriNet,
    final_marking: Marking
) -> Tuple[
    List[PetriNet.Transition],
    Dict[PetriNet.Transition, Set[str]]
]:
    visible_transitions: List[PetriNet.Transition] = []
    silent_transition_map: Dict[PetriNet.Transition, Set[str]] = defaultdict(set)

    for t in net.transitions:
        if t.label is not None:
            visible_transitions.append(t)

    for silent_t in net.transitions:
        if silent_t.label is not None:
            continue  # skip visible transitions

        visited = set()
        to_visit = set()

        for arc in net.arcs:
            if arc.source == silent_t:
                to_visit.add(arc.target)

        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)

            if isinstance(node, PetriNet.Place):
                if node in final_marking and final_marking[node] > 0:
                    silent_transition_map[silent_t].add("END")

                for arc in net.arcs:
                    if arc.source == node and isinstance(arc.target, PetriNet.Transition):
                        target_t = arc.target
                        if target_t.label is not None:
                            silent_transition_map[silent_t].add(target_t.label)
                        else:
                            for arc2 in net.arcs:
                                if arc2.source == target_t:
                                    to_visit.add(arc2.target)
                                    
    return visible_transitions, silent_transition_map





