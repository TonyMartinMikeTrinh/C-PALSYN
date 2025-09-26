from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Tuple, Dict, Set
from collections import defaultdict
from pm4py.algo.simulation.playout.petri_net.variants.basic_playout import Parameters as Parameter_BASIC_PO
from pm4py.algo.simulation.playout.petri_net.variants.extensive import Parameters as Parameter_EXTENSIV_PO
from pm4py.algo.simulation.playout.petri_net.algorithm import Variants


def generate_adjacency_dict(
    petri_net: Tuple[PetriNet, Marking, Marking],
    no_traces: int = 10000,
    max_trace_length: int = 100,
    variant: Variants = Variants.BASIC_PLAYOUT
) -> Dict[str, Set[str]]:
    """
    Simulates the given Petri net and builds an adjacency dictionary
    representing the transition graph based on the simulated event log.

    Parameters:
        petri_net (Tuple[PetriNet, Marking, Marking]): A tuple (net, initial_marking, final_marking)
        no_traces (int): Number of traces to simulate
        max_trace_length (int): Maximum number of events per trace
        variant (Variants): Simulation variant to use (BASIC_PLAYOUT or EXTENSIVE)

    Returns:
        Dict[str, Set[str]]: An adjacency dictionary where each key is a source transition
                             and the value is a set of directly following transitions.
                             Includes "START" and "END" markers where applicable.
    """
    net, im, _ = petri_net

    if variant == Variants.BASIC_PLAYOUT:
        parameters = {
            Parameter_BASIC_PO.NO_TRACES: no_traces,
            Parameter_BASIC_PO.MAX_TRACE_LENGTH: max_trace_length
        }
    elif variant == Variants.EXTENSIVE:
        parameters = {
            #Parameter_EXTENSIV_PO.RETURN_ELEMENTS: no_traces,
            Parameter_EXTENSIV_PO.MAX_TRACE_LENGTH: max_trace_length
        }
    else:
        raise ValueError(f"Unsupported simulation variant: {variant}")

    sim_log = simulator.apply(net, im, variant=variant, parameters=parameters)
    print(f"Simulated traces: {len(sim_log)}")
    adjacency_dict = defaultdict(set)

    for trace in sim_log:
        labels = [event["concept:name"] for event in trace]

        if labels:
            adjacency_dict[None].add(labels[0])  # Start node

        for i in range(len(labels) - 1):
            adjacency_dict[labels[i]].add(labels[i + 1])

        if len(labels) < max_trace_length:
            adjacency_dict[labels[-1]].add("END")  # End marker

    return dict(adjacency_dict)
