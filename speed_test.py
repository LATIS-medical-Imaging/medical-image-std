import timeit
from itertools import permutations

import numpy as np
from math import log, e


def entropy_numpy(labels, base=None):
    """Computes entropy of label distribution for each row in a 2D array and returns their average."""
    labels = np.array(labels)

    def entropy_1d(row, base):
        n_labels = len(row)

        if n_labels <= 1:
            return 0

        value, counts = np.unique(row, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.0

        # Compute entropy
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)

        return ent

    # Calculate entropy for each row
    row_entropies = [entropy_1d(row, base) for row in labels]

    # Return the average entropy
    return np.mean(row_entropies)


def entropy_main(input):
    """
    This function calculates Shannon Entropy of an image.
    For more information about the Entropy this link:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Parameters:
        input: 2d ndarray to process.

    Returns:
        entropy: float rounded to 4 decimal places

    Notes:
        The logarithm used is the bit logarithm (base-2).

    Examples:
        >>> import numpy as np
        >>> a = np.random.randint(0, 4095, (512,512))
        >>> ent = entropy_main(a)
        >>> ent
        11.9883
    """
    input = np.array(input)
    # Flatten the input to a 1D array for histogram calculation
    flat_input = input.flatten()
    histogram, _ = np.histogram(
        flat_input,
        bins=np.arange(flat_input.min(), flat_input.max() + 2) - 0.5,
    )

    # Calculate probabilities
    probabilities = histogram / flat_input.size

    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return np.around(entropy, decimals=4)


if __name__ == "__main__":
    digits = '9876543210'
    perms = permutations(digits, 9)

    max_number = 0
    for perm in perms:
        number = int(''.join(perm))
        odd_sum = sum(int(perm[i]) for i in range(0, 9, 2))
        even_sum = sum(int(perm[i]) for i in range(1, 9, 2))
        if (odd_sum - even_sum) % 11 == 0:
            max_number = max(max_number, number)
    print(max_number)
#     repeat_number = 1000000
#     print("start A")
#     a = timeit.repeat(
#         stmt="""entropy_numpy(labels)""",
#         setup="""labels=[
#     [1, 1, 2, 2, 2],
#     [1, 2, 3, 3, 3],
#     [1, 1, 1, 1, 1]
# ];from __main__ import entropy_numpy""",
#         repeat=3,
#         number=repeat_number,
#     )
#     print("end A")
#     b = timeit.repeat(
#         stmt="""entropy_main(input)""",
#         setup="""input = [
#     [1, 1, 2, 2, 2],
#     [1, 2, 3, 3, 3],
#     [1, 1, 1, 1, 1]
# ];from __main__ import entropy_main""",
#         repeat=3,
#         number=repeat_number,
#     )
#     for approach, timeit_results in zip(["numpy/math", "main/numpy"], [a, b]):
#         print(
#             "Method: {}, Avg.: {:.6f}".format(approach, np.array(timeit_results).mean())
#         )
