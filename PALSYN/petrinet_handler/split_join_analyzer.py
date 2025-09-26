from typing import List, Dict, Set
from pm4py.objects.petri_net.importer import importer as pnml_importer

class SplitJoinAnalyzerPM4Py:
    def __init__(self, pnml_path: str):
        self.net, self.im, self.fm = pnml_importer.apply(pnml_path)

    def find_and_splits(self) -> List[str]:
        and_splits = []
        for t in self.net.transitions:
            if len(t.out_arcs) > 1:
                and_splits.append(t.name)
        return and_splits

    def get_reachable_transitions(self, place_name: str, visited=None) -> Set[str]:
        if visited is None:
            visited = set()
        result = set()

        for arc in self.net.arcs:
            if arc.source.name == place_name:
                target = arc.target
                if target.name in visited:
                    continue
                visited.add(target.name)
                if hasattr(target, "label"):  # it's a transition
                    result.add(target.name)
                    # Recurse through its outputs
                    for arc2 in self.net.arcs:
                        if arc2.source == target:
                            result |= self.get_reachable_transitions(arc2.target.name, visited)
        return result

    def find_matching_joins(self) -> Dict[str, List[str]]:
        matches = {}
        for t in self.net.transitions:
            if len(t.out_arcs) > 1:
                split_name = t.name
                num_paths = len(t.out_arcs)
                paths = []

                for arc in t.out_arcs:
                    place = arc.target
                    transitions = self.get_reachable_transitions(place.name)
                    paths.append(transitions)

                if paths:
                    intersection = set.intersection(*paths)

                    # Nur Transitionen mit passender Anzahl eingehender Arcs
                    valid_joins = []
                    for tr in self.net.transitions:
                        if tr.name in intersection and len(tr.in_arcs) == num_paths:
                            valid_joins.append(tr)

                    # Wähle die erste erreichbare Join-Transition (nach Platz im PNML-Modell)
                    if valid_joins:
                        matches[split_name] = [valid_joins[0].name]  # Nur die früheste
                    else:
                        matches[split_name] = []
        return matches
    
    def compute_node_depths_from_marking(self):
    
        from collections import deque, defaultdict
        depths = defaultdict(lambda: float('inf'))
        visited = set()
        queue = deque()

        # Alle initialen Places mit Tiefe 0
        for place in self.im:
            depths[place.name] = 0
            queue.append(place)

        # Breiten-Suche (BFS)
        while queue:
            node = queue.popleft()
            current_depth = depths[node.name]

            # Alle ausgehenden Arcs von diesem Knoten
            for arc in self.net.arcs:
                if arc.source.name == node.name:
                    target = arc.target
                    if depths[target.name] > current_depth + 1:
                        depths[target.name] = current_depth + 1
                        queue.append(target)

        return dict(depths)

