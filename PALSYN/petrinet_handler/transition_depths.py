from collections import deque
from pm4py.objects.petri_net.importer import importer as pnml_importer
import pandas as pd

def compute_structural_transition_depths(pnml_path: str):
    net, im, fm = pnml_importer.apply(pnml_path)

    Transition = net.__class__.Transition
    Place = net.__class__.Place

    # Initialisiere Tiefe und Flags
    transition_depths = {t.name: float("inf") for t in net.transitions if t.name is not None}
    is_and_split = {t.name: len(t.out_arcs) > 1 for t in net.transitions if t.name is not None}
    is_and_join = {t.name: len(t.in_arcs) > 1 for t in net.transitions if t.name is not None}

    # Finde Start-Transitions (Transitions nach Start-Places)
    start_transitions = set()
    for place in im:
        for arc in net.arcs:
            if arc.source == place and isinstance(arc.target, Transition):
                start_transitions.add(arc.target)

    # BFS-Queue: (Transition, current_depth)
    queue = deque()
    for t in start_transitions:
        transition_depths[t.name] = 0
        queue.append((t, 0))

    while queue:
        current_t, depth = queue.popleft()

        # Berechne Tiefe je nach Struktur
        out_places = [arc.target for arc in net.arcs if arc.source == current_t and isinstance(arc.target, Place)]
        for place in out_places:
            out_arcs = [arc for arc in net.arcs if arc.source == place]

            for arc in out_arcs:
                next_t = arc.target
                if not isinstance(next_t, Transition) or next_t.name is None:
                    continue

                new_depth = depth
                if is_and_split.get(current_t.name, False):
                    new_depth += 1  # AND-Split
                if is_and_join.get(next_t.name, False):
                    new_depth -= 1  # AND-Join

                if transition_depths[next_t.name] > new_depth:
                    transition_depths[next_t.name] = new_depth
                    queue.append((next_t, new_depth))

    # Ergebnis zusammenfassen
    result = []
    for t in net.transitions:
        if t.name is None:
            continue
        result.append((
            t.name,
            t.label,
            transition_depths[t.name],
            is_and_split[t.name],
            is_and_join[t.name]
        ))

    df = pd.DataFrame(result, columns=["Transition", "Label", "Depth", "AND_Split", "AND_Join"])
    df = df.sort_values(by="Depth")
    return df


def find_direct_input_places_for_and_joins(pnml_path: str, df_with_joins: pd.DataFrame):
    net, im, fm = pnml_importer.apply(pnml_path)

    Transition = net.__class__.Transition
    Place = net.__class__.Place

    # AND-Join-Transitions aus dem DataFrame
    and_join_names = df_with_joins[df_with_joins["AND_Join"] == True]["Transition"].values
    and_joins = [t for t in net.transitions if t.name in and_join_names]

    result = []

    for join_t in and_joins:
        for arc in net.arcs:
            if arc.target == join_t and isinstance(arc.source, Place):
                result.append((join_t.name, arc.source.name))

    df = pd.DataFrame(result, columns=["ANDJoin_Transition", "needed_Place"]).drop_duplicates()
    return df

def find_direct_output_places_for_and_splits(pnml_path: str, df_with_splits: pd.DataFrame):
    net, im, fm = pnml_importer.apply(pnml_path)

    Transition = net.__class__.Transition
    Place = net.__class__.Place

    # AND-Split-Transitionen aus dem DataFrame
    and_split_names = df_with_splits[df_with_splits["AND_Split"] == True]["Transition"].values
    and_splits = [t for t in net.transitions if t.name in and_split_names]

    result = []

    for split_t in and_splits:
        for arc in net.arcs:
            if arc.source == split_t and isinstance(arc.target, Place):
                result.append((split_t.name, arc.target.name))

    df = pd.DataFrame(result, columns=["ANDSplit_Transition", "out_place"]).drop_duplicates()
    return df

from collections import deque

def find_and_split_to_join_paths(net, df):
    Transition = net.__class__.Transition
    Place = net.__class__.Place

    # Mapping: Transition-Name → Tiefe, Join/Split-Markierung
    transition_info = {
        row["Transition"]: {
            "depth": row["Depth"],
            "is_join": row["AND_Join"],
            "is_split": row["AND_Split"]
        }
        for _, row in df.iterrows()
    }

    result = []

    for _, row in df[df["AND_Split"] == True].iterrows():
        split_name = row["Transition"]
        split_depth = row["Depth"]
        split_transition = next(t for t in net.transitions if t.name == split_name)

        visited = set()
        queue = deque()
        queue.append(split_transition)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if isinstance(node, Transition):
                # Transition → Places
                for arc in net.arcs:
                    if arc.source == node and isinstance(arc.target, Place):
                        queue.append(arc.target)

            elif isinstance(node, Place):
                # Place → Transitions
                for arc in net.arcs:
                    if arc.source == node and isinstance(arc.target, Transition):
                        t = arc.target
                        t_info = transition_info.get(t.name)
                        if t_info and t_info["is_join"] and t_info["depth"] == split_depth:
                            result.append((split_name, t.name))
                            # Wir stoppen bei erstem gefundenem Join auf dieser Tiefe
                            queue.clear()
                            break
                        queue.append(t)
    return result
