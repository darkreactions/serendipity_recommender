import numpy as np


def euclidean(inputs: list) -> list:
    """Calculate pair-wise serendipity using euclidean distance form

    Args:
        inputs (list):      List representation of one pair of reactions.
                                * Format: [pair name, reaction 1 data,
                                reaction 2 data]

    Return:
        List of pair name and the serendipity value in Euclidean distance form.
    """

    # Separate data input
    pair_id, reaction_1, reaction_2 = inputs

    # For euclidean distance, find the diff vector
    diff = reaction_1 - reaction_2

    # Calculate L-2 norm
    dist = np.linalg.norm(diff, 2)

    # Normalize the result to range [0, 1] with tanh
    return [pair_id, np.tanh(dist / len(diff))]


def cosine_similarity(inputs: list) -> list:
    """Calculate pair-wise serendipity using cosine similarity complement form

    Args:
        inputs (list):      List representation of one pair of reactions.
                                * Format: [pair name, reaction 1 data,
                                reaction 2 data]

    Return:
        List of pair name and the serendipity value in cosine similarity form.
    """

    # Separate data input
    pair_id, reaction_1, reaction_2 = inputs

    # Find cosine similarity
    cos_sim = np.dot(reaction_1, reaction_2) / (np.linalg.norm(
        reaction_1) * np.linalg.norm(reaction_2))

    # Normalize the complement of cosine similarity to range [0, 1]
    return [pair_id, 0.5 * (1 - cos_sim)]
