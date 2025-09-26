from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.semantics import ClassicSemantics
from typing import List, Optional, Tuple, Dict, Set

semantics = ClassicSemantics()


def execute_transition(
    t: Optional[PetriNet.Transition],
    net: PetriNet,
    marking: Marking
) -> Marking:
    """
    Executes the given transition (if not None) and returns the new marking.

    Raises an error if the transition is not enabled.

    Parameters
    ----------
    t : PetriNet.Transition or None
        The transition to execute. If None, returns the marking unchanged.
    net : PetriNet
        The Petri net
    marking : Marking
        The current marking

    Returns
    -------
    Marking
        The updated marking
    """
    if t is None:
        return marking

    enabled = semantics.enabled_transitions(net, marking)
    if t not in enabled:
        raise ValueError(f"The given transition {t.name} is not enabled in the current marking.")

    return semantics.execute(t, net, marking)


def get_enabled_transitions(
    net: PetriNet,
    marking: Marking
) -> List[PetriNet.Transition]:
    """
    Returns the list of currently enabled transitions.

    Parameters
    ----------
    net : PetriNet
        The Petri net
    marking : Marking
        The current marking

    Returns
    -------
    List[PetriNet.Transition]
        Enabled transitions at the given marking
    """
    return list(semantics.enabled_transitions(net, marking))


def split_enabled_transitions(
    enabled: List[PetriNet.Transition],
    visible_transitions: List[PetriNet.Transition],
    silent_transition_map: Dict[PetriNet.Transition, Set[str]]
) -> Tuple[List[PetriNet.Transition], List[PetriNet.Transition]]:
    """
    Splits the currently enabled transitions into visible and silent, based on provided transition information.

    Parameters
    ----------
    enabled : List[PetriNet.Transition]
        Currently enabled transitions (from marking)
    visible_transitions : List[PetriNet.Transition]
        All known visible transitions
    silent_transition_map : Dict[PetriNet.Transition, Set[str]]
        Mapping of known silent transitions to visible labels

    Returns
    -------
    Tuple[List[PetriNet.Transition], List[PetriNet.Transition]]
        (enabled_visible_transitions, enabled_silent_transitions)
    """
    enabled_visible = []
    enabled_silent = []

    for t in enabled:
        if t.label is not None and t in visible_transitions:
            enabled_visible.append(t)
        elif t.label is None and t in silent_transition_map:
            enabled_silent.append(t)

    return enabled_visible, enabled_silent