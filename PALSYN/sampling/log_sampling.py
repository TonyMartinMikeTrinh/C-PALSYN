import time
import random
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tensorflow.keras import backend as K
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Optional, Tuple
from keras.utils import pad_sequences
from typing import List, Tuple, Dict, Set
from PALSYN.preprocessing.log_preprocessing import START_TOKEN, END_TOKEN
from PALSYN.petrinet_handler.simulate_to_adjacency import generate_adjacency_dict
from PALSYN.petrinet_handler.petri_execution import execute_transition, get_enabled_transitions, split_enabled_transitions
from PALSYN.petrinet_handler.analyze_transitions import analyze_transitions_with_end_labels
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.simulation.playout.petri_net.algorithm import Variants

def clean_sequence(sequence: List[str], max_length: int) -> List[str]:
    """Clean a generated token sequence.

    Removes special markers, strips event concept prefixes, and preserves
    case-level tokens. The returned sequence is prepended with `START_TOKEN`.
    If the input sequence length is greater than or equal to `max_length`, an
    empty list is returned to filter out runaway generations.

    Args:
        sequence: Generated tokens that may include case-level entries (prefix
            `"case:"`) and event-level tokens in the form
            `"<concept>==<column>==<value>"`.
        max_length: Maximum allowed raw sequence length; sequences at or above
            this length are discarded.

    Returns:
        A cleaned token sequence beginning with `START_TOKEN`, or an empty list
        if discarded.
    """
    if len(sequence) >= max_length:
        return []

    trace: List[str] = [START_TOKEN]
    for word in sequence:
        if not word or word in {START_TOKEN, END_TOKEN}:
            continue
        if word.startswith("case:"):
            trace.append(word)
        else:
            parts = word.split("==")
            trace.append("==".join(parts[1:]))

    return trace


def _build_token_index_maps(index_word: Dict[int, str], columns: Iterable[str]) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]]]:
    """Build lookup tables of token indices by column and by (concept, column).

    Args:
        index_word: Mapping from token index to the corresponding token string.
        columns: Iterable of column names aligned with the model outputs.

    Returns:
        A tuple with two dictionaries:
        - by_column: Maps `column` to a list of token indices whose token
          contains the substring `"=={column}=="`.
        - by_concept_and_column: Maps `(concept, column)` to a list of token
          indices whose token contains `"{concept}=={column}=="`.
    """
    col_set = set(columns)
    by_column: Dict[str, List[int]] = {c: [] for c in col_set}
    by_concept_and_column: Dict[Tuple[str, str], List[int]] = {}

    for idx, token in index_word.items():
        for col in col_set:
            marker = f"=={col}=="
            if marker in token:
                by_column[col].append(idx)
                concept = token.split("==", 1)[0]
                by_concept_and_column.setdefault((concept, col), []).append(idx)

    return by_column, by_concept_and_column


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to a probability distribution.

    If the input sums to zero (or is empty), returns a uniform distribution to
    avoid NaNs.

    Args:
        arr: Array of non-negative scores.

    Returns:
        Array of probabilities summing to 1.0.
    """
    arr = np.asarray(arr, dtype=float)
    total = float(arr.sum())
    if total > 0.0:
        return arr / total
    n = arr.size if arr.size > 0 else 1
    return np.full(n, 1.0 / n, dtype=float)


def sample_batch(
    sample_size: int,
    tokenizer: Any,
    max_sequence_len: int,
    model: Any,
    batch_size: int,
    num_cols: int,
    column_list: List[str],
) -> List[List[str]]:
    """Generate synthetic sequences using conditional per-column sampling.

    At each step, samples one token per column, constraining event-level
    attributes to match the current `concept:name`. A sequence terminates when
    `concept:name` predicts `END` or no valid candidates remain.

    Args:
        sample_size: Target number of sequences to return.
        tokenizer: Fitted tokenizer exposing `word_index` and
            `texts_to_sequences`.
        max_sequence_len: Maximum sequence length used during training; also
            bounds generation and post-processing.
        model: Trained Keras model with one softmax output per column in
            `column_list`.
        batch_size: Minimum number of sequences generated in one sweep.
        num_cols: Number of output columns per generation step.
        column_list: Ordered list of column names matching model outputs.

    Returns:
        List of cleaned token sequences (each a list of strings).

    Notes:
        - Prints a simple progress bar to stdout during generation.
        - Clears the Keras backend session before returning.
    """
    start_time = time.perf_counter()

    effective_batch_size = max(int(batch_size), int(sample_size))

    seed_sequences: List[List[str]] = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
    active_mask = np.ones(effective_batch_size, dtype=bool)

    index_word: Dict[int, str] = {index: word for word, index in tokenizer.word_index.items()}

    tokens_by_column, tokens_by_concept_column = _build_token_index_maps(index_word, column_list)

    total_sequences = effective_batch_size
    completed = 0
    last_percent = 0

    def update_progress() -> None:
        nonlocal completed, last_percent
        completed += 1
        pct = int((completed / total_sequences) * 100)
        if pct > last_percent:
            filled = pct // 2
            bar = "█" * filled + "░" * (50 - filled)
            print(f"\rProgress: |{bar}| {pct}% ", end="", flush=True)
            last_percent = pct

    while np.any(active_mask):
        token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in seed_sequences]
        padded = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")

        try:
            model.reset_states()
        except Exception:
            pass

        predictions = model.predict(padded, verbose=0)

        for i, (active, seq) in enumerate(zip(active_mask, seed_sequences)):
            if not active:
                continue

            current_concept: str | None = None
            step_tokens: List[str] = []

            for output_probs, column in zip(predictions, column_list):
                probs = _safe_normalize(output_probs[i])

                if current_concept is None:
                    candidates = tokens_by_column.get(column, [])
                else:
                    candidates = tokens_by_concept_column.get((current_concept, column), [])

                if not candidates:
                    active_mask[i] = False
                    update_progress()
                    break

                cand_probs = _safe_normalize(np.array([probs[idx] for idx in candidates], dtype=float))

                next_index = np.random.choice(candidates, p=cand_probs)
                next_word = index_word.get(int(next_index), END_TOKEN)

                if column == "concept:name":
                    current_concept = next_word.split("==")[0]
                    if current_concept == "END" or next_word == END_TOKEN:
                        active_mask[i] = False
                        update_progress()
                        break

                step_tokens.append(next_word)

            if active_mask[i]:
                seq.extend(step_tokens)
                if len(seq) >= (max_sequence_len * 2):
                    active_mask[i] = False
                    update_progress()

    K.clear_session()

    max_len_cutoff = int(round(max_sequence_len * 1.5))
    cleaned: List[List[str]] = [
        clean_sequence(sentence, max_len_cutoff)
        for sentence in seed_sequences
        if len(sentence) < max_len_cutoff
    ]

    if len(cleaned) > sample_size:
        cleaned = random.sample(cleaned, sample_size)

    duration = time.perf_counter() - start_time
    print(f"\nGenerated {len(cleaned)} sequences")
    print(f"Time to generate sequences: {duration:.2f}s")


    return clean_synthetic_event_log_sentences

def sample_batch_1(
        sample_size: int,
        tokenizer,
        max_sequence_len: int,
        model,
        batch_size: int,
        num_cols: int,
        column_list: list[str],
        petri_net: Optional[Tuple[PetriNet, Marking, Marking]] = None
) -> list[list[str]]:
    """
    Generate synthetic event log sentences using a trained DP-BiLSTM model with conditional sampling.
    """
    start_time = time.time()

    effective_batch_size = max(batch_size, sample_size)
    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    batch_seed_texts = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
    batch_active = np.ones(effective_batch_size, dtype=bool)

    net, initial_marking, final_marking = petri_net

    concept_name = "concept:name"

    valid_tokens = [
        index for index, word in index_word.items()
        if f"=={concept_name}==" in word
    ]
    
    token_labels = {
        index: index_word[index].split("==")[0]
        for index in valid_tokens
    }

    concept_index = {
        label: index
        for index, label in token_labels.items()
    }
    #print("valid_tokens", valid_tokens)
    #print("token_labels", token_labels)
    label_successors_map = generate_adjacency_dict(petri_net)

    # Step 1: Collect invalid keys and values
    invalid_concepts = set()

    for key, values in label_successors_map.items():
        if key is not None and key not in concept_index:
            invalid_concepts.add(key)
        for val in values:
            if val not in concept_index:
                invalid_concepts.add(val)

    # Mark invalid transitions as silent
    for transition in net.transitions:
        if transition.label in invalid_concepts:
            print(f"Marking transition '{transition.label}' as silent")
            transition.label = None
    label_successors_map = generate_adjacency_dict([net,initial_marking,final_marking])

    # Step 3: Map string → index
    label_to_token_ids = {
        key: [concept_index[val] for val in values]
        for key, values in label_successors_map.items()
    }

    #print("label_successors_map: ", label_successors_map)
    #print("label_to_token_ids ",label_to_token_ids)
    

    # Progress tracking variables
    total_sequences = effective_batch_size
    completed_sequences = 0
    last_percentage = 0

    def update_progress():
        nonlocal completed_sequences, last_percentage
        completed_sequences += 1
        current_percentage = int((completed_sequences / total_sequences) * 100)
        if current_percentage > last_percentage:
            progress_bar = "█" * (current_percentage // 2) + "░" * (50 - (current_percentage // 2))
            print(f"\rProgress: |{progress_bar}| {current_percentage}% ", end="", flush=True)
            last_percentage = current_percentage

    last_token_per_index = [None for _ in range(len(batch_seed_texts))]

    while np.any(batch_active):
        token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
        padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")
        model.reset_states()
        predictions = model.predict(padded_token_lists, verbose=0)

        for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
            if not active:
                continue

            latest_concept_name = None
            synth_row = []

            for prediction_output, column in zip(predictions, column_list):
                prediction_output = prediction_output[i]
                prediction_output = prediction_output / np.sum(prediction_output)

                if latest_concept_name is None:
                    valid_tokens = [
                        index for index, word in index_word.items()
                        if f"=={column}==" in word
                    ]
                    valid_tokens = label_to_token_ids[last_token_per_index[i]]
                else:
                    valid_tokens = [
                        index for index, word in index_word.items()
                        if f"{latest_concept_name}=={column}==" in word
                    ]

                if len(valid_tokens) == 0:
                    batch_active[i] = False
                    update_progress()
                    break

                filtered_probabilities = [prediction_output[token] for token in valid_tokens]
                filtered_probabilities = np.array(filtered_probabilities) / np.sum(filtered_probabilities)

                next_word_index = np.random.choice(valid_tokens, p=filtered_probabilities)
                next_word = index_word.get(next_word_index, END_TOKEN)

                if column == "concept:name":
                    latest_concept_name = next_word.split("==")[0]
                    last_token_per_index[i] = latest_concept_name
                    if latest_concept_name == "END" or next_word == END_TOKEN:
                        batch_active[i] = False
                        update_progress()
                        break

                synth_row.append(next_word)

            if batch_active[i]:
                seq.extend(synth_row)
                if len(seq) >= (max_sequence_len * 2):
                    batch_active[i] = False
                    update_progress()

    synthetic_event_log_sentences.extend(batch_seed_texts)
    K.clear_session()

    # Clean event prefixes and exclude overly long sequences
    clean_synthetic_event_log_sentences = [
        clean_sequence(sentence, round(max_sequence_len * 1.5))
        for sentence in synthetic_event_log_sentences
        if len(sentence) < round(max_sequence_len * 1.5)
    ]

    # Randomly sample the required number of sequences
    if len(clean_synthetic_event_log_sentences) > sample_size:
        clean_synthetic_event_log_sentences = random.sample(clean_synthetic_event_log_sentences, sample_size)

    print(f"\nGenerated {len(clean_synthetic_event_log_sentences)} sequences")
    print("Time taken to generate synthetic event log sentences: ", time.time() - start_time)

    return clean_synthetic_event_log_sentences


def sample_batch_2(
        sample_size: int,
        tokenizer,
        max_sequence_len: int,
        model,
        batch_size: int,
        num_cols: int,
        column_list: list[str],
        petri_net: Optional[Tuple[PetriNet, Marking, Marking]] = None
) -> list[list[str]]:
    """
    Generate synthetic event log sentences using a trained DP-BiLSTM model with conditional sampling.
    """
    start_time = time.time()

    effective_batch_size = max(batch_size, sample_size)
    synthetic_event_log_sentences = []
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    batch_seed_texts = [[START_TOKEN] * num_cols for _ in range(effective_batch_size)]
    batch_active = np.ones(effective_batch_size, dtype=bool)

    concept_name = "concept:name"

    valid_tokens = [
        index for index, word in index_word.items()
        if f"=={concept_name}==" in word
    ]
    
    token_labels = {
        index: index_word[index].split("==")[0]
        for index in valid_tokens
    }

    concept_index = {
        label: index
        for index, label in token_labels.items()
    }

    net, initial_marking, final_marking = petri_net

    visible_transitions, silent_transition_map = analyze_transitions_with_end_labels(net, final_marking)

    # Mark invalid transitions as silent
    for transition in net.transitions:
        if transition.label is not None and transition.label not in concept_index:
            print(f"Marking transition '{transition.label}' as silent")
            transition.label = None

    visible_transitions, silent_transition_map = analyze_transitions_with_end_labels(net, final_marking)

    filtered_transitions = []
    for t in visible_transitions:
        if t.label in concept_index:
            filtered_transitions.append(t)
        else:
            print(f"Transition '{t.label}' not found. Prob set to 0")

    visible_transitions = filtered_transitions
    

    marking = initial_marking.copy()

    last_marking_per_index = [marking for _ in range(len(batch_seed_texts))]
    
    # Progress tracking variables
    total_sequences = effective_batch_size
    completed_sequences = 0
    last_percentage = 0

    def update_progress():
        nonlocal completed_sequences, last_percentage
        completed_sequences += 1
        current_percentage = int((completed_sequences / total_sequences) * 100)
        if current_percentage > last_percentage:
            progress_bar = "█" * (current_percentage // 2) + "░" * (50 - (current_percentage // 2))
            print(f"\rProgress: |{progress_bar}| {current_percentage}% ", end="", flush=True)
            last_percentage = current_percentage

    def get_filtered_probabilities(valid_tokens, prediction_output):
        filtered_probabilities = []
        for token in valid_tokens:
            if isinstance(token, list):
                prob_sum = sum(prediction_output[t] for t in token)
                filtered_probabilities.append(prob_sum)
            else:
                filtered_probabilities.append(prediction_output[token])
        return filtered_probabilities

    last_token_per_index = [None for _ in range(len(batch_seed_texts))]

    def get_enabled_tokens_and_transitions(
        net: PetriNet,
        current_marking: Marking,
        visible_transitions: List[PetriNet.Transition],
        silent_transition_map: Dict[PetriNet.Transition, Set[str]],
        concept_index: Dict[str, int]
    ) -> Tuple[
        List[int],                    # valid_tokens
        List[PetriNet.Transition]     # currently_enabled_transitions
    ]:
        """
        Returns the valid concept indexes and currently enabled transitions.

        Parameters
        ----------
        net : PetriNet
        current_marking : Marking
        visible_transitions : list of visible transitions
        silent_transition_map : dict mapping silent transitions to visible labels
        concept_index : dict mapping labels to concept indexes

        Returns
        -------
        Tuple[List[int], List[PetriNet.Transition]]
            valid_tokens and currently_enabled_transitions
        """
        enabled_transitions = get_enabled_transitions(net, current_marking)

        enabled_visible, enabled_silent = split_enabled_transitions(
            enabled_transitions,
            visible_transitions,
            silent_transition_map
        )

        enabled_mapped_visible = [
            concept_index[t.label]
            for t in enabled_visible
            if t.label in concept_index
        ]
        
        enabled_silent_reachable_labels = [
            [concept_index[label] for label in silent_transition_map.get(silent_t, set()) if label in concept_index]
            for silent_t in enabled_silent
        ]

        valid_tokens = enabled_mapped_visible + enabled_silent_reachable_labels
        currently_enabled_transitions = enabled_visible + enabled_silent

        return valid_tokens, currently_enabled_transitions

    while np.any(batch_active):
        token_lists = [tokenizer.texts_to_sequences([seq])[0] for seq in batch_seed_texts]
        padded_token_lists = pad_sequences(token_lists, maxlen=max_sequence_len, padding="pre")
        model.reset_states()
        predictions = model.predict(padded_token_lists, verbose=0)

        for i, (active, seq) in enumerate(zip(batch_active, batch_seed_texts)):
            if not active:
                continue

            latest_concept_name = None
            synth_row = []

            for prediction_output, column in zip(predictions, column_list):
                prediction_output = prediction_output[i]
                prediction_output = prediction_output / np.sum(prediction_output)

                if latest_concept_name is None:
                    valid_tokens = [
                        index for index, word in index_word.items()
                        if f"=={column}==" in word
                    ]

                    valid_tokens, currently_enabled_transitions = get_enabled_tokens_and_transitions(
                        net,
                        last_marking_per_index[i],
                        visible_transitions,
                        silent_transition_map,
                        concept_index
                    )


                else:
                    valid_tokens = [
                        index for index, word in index_word.items()
                        if f"{latest_concept_name}=={column}==" in word
                    ]

                if len(valid_tokens) == 0:
                    batch_active[i] = False
                    update_progress()
                    break
                
                if latest_concept_name is None:
                    while True:
                        filtered_probabilities = get_filtered_probabilities(valid_tokens, prediction_output)
                        filtered_probabilities = np.array(filtered_probabilities) / np.sum(filtered_probabilities)

                        selected_idx = np.random.choice(len(valid_tokens), p=filtered_probabilities)

                        next_word_index = valid_tokens[selected_idx]

                        if not isinstance(next_word_index, list):
                            last_marking_per_index[i] = execute_transition(currently_enabled_transitions[selected_idx],net,last_marking_per_index[i])
                            break
                        else:
                            last_marking_per_index[i] = execute_transition(currently_enabled_transitions[selected_idx], net, last_marking_per_index[i])
                            
                        if last_marking_per_index[i] == final_marking:
                            if isinstance(next_word_index, list):
                                # if END is a single list
                                next_word_index = next_word_index[0]
                            break
                        
                        valid_tokens, currently_enabled_transitions = get_enabled_tokens_and_transitions(
                            net,
                            last_marking_per_index[i],
                            visible_transitions,
                            silent_transition_map,
                            concept_index
                        )

                    next_word = index_word.get(next_word_index, END_TOKEN)

                else:
                    filtered_probabilities = [prediction_output[token] for token in valid_tokens]
                    filtered_probabilities = np.array(filtered_probabilities) / np.sum(filtered_probabilities)
                    
                    next_word_index = np.random.choice(valid_tokens, p=filtered_probabilities)
                    next_word = index_word.get(next_word_index, END_TOKEN)

                if column == "concept:name":
                    latest_concept_name = next_word.split("==")[0]
                    last_token_per_index[i] = latest_concept_name



                    last_marking_per_index[i]
                    if latest_concept_name == "END" or next_word == END_TOKEN:
                        batch_active[i] = False
                        update_progress()
                        break

                synth_row.append(next_word)

            if batch_active[i]:
                seq.extend(synth_row)
                if len(seq) >= (max_sequence_len * 2):
                    batch_active[i] = False
                    update_progress()

    synthetic_event_log_sentences.extend(batch_seed_texts)
    K.clear_session()

    # Clean event prefixes and exclude overly long sequences
    clean_synthetic_event_log_sentences = [
        clean_sequence(sentence, round(max_sequence_len * 1.5))
        for sentence in synthetic_event_log_sentences
        if len(sentence) < round(max_sequence_len * 1.5)
    ]

    # Randomly sample the required number of sequences
    if len(clean_synthetic_event_log_sentences) > sample_size:
        clean_synthetic_event_log_sentences = random.sample(clean_synthetic_event_log_sentences, sample_size)

    print(f"\nGenerated {len(clean_synthetic_event_log_sentences)} sequences")
    print("Time taken to generate synthetic event log sentences: ", time.time() - start_time)

    return clean_synthetic_event_log_sentences

