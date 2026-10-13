"""Balanced scene-size schedules shared by collection and expert benchmarks."""

from collections import Counter
from itertools import product

import numpy as np


def add_count_arguments(parser, *, mixed=True):
    parser.add_argument("--num-cubes", "--num-objects", dest="num_cubes",
                        type=int, choices=(1, 2, 3, 4), default=3,
                        help="Fixed object count; Stack requires at least two (4 is legacy)")
    parser.add_argument("--num-receptacles", type=int, choices=(1, 2, 3, 4),
                        help="Fixed Place receptacle count (default: 3)")
    if mixed:
        parser.add_argument("--object-counts", nargs="+", type=int, choices=(1, 2, 3),
                            help="Balance successful demos / benchmark attempts across these counts")
        parser.add_argument("--receptacle-counts", nargs="+", type=int, choices=(1, 2, 3),
                            help="Place only: balance the Cartesian object/receptacle count pairs")


def count_options(task, args):
    objects = tuple(getattr(args, "object_counts", None) or [args.num_cubes])
    supplied_receptacles = getattr(args, "receptacle_counts", None)
    fixed_receptacles = getattr(args, "num_receptacles", None)
    if len(set(objects)) != len(objects):
        raise ValueError("Object counts cannot contain duplicates")
    minimum = 2 if task == "stack" else 1
    if any(count not in range(minimum, 5) for count in objects):
        raise ValueError(f"{task} requires object counts between {minimum} and 4")
    if task != "place":
        if supplied_receptacles is not None or fixed_receptacles is not None:
            raise ValueError("Receptacle count options are only valid for Place")
        return objects, (None,)
    receptacles = tuple(supplied_receptacles or [fixed_receptacles or 3])
    if len(set(receptacles)) != len(receptacles):
        raise ValueError("Receptacle counts cannot contain duplicates")
    if any(count not in range(1, 5) for count in receptacles):
        raise ValueError("Place requires receptacle counts between 1 and 4")
    return objects, receptacles


def balanced_count_plan(episodes, object_counts, receptacle_counts=(None,), seed=0):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    combinations = list(product(object_counts, receptacle_counts))
    if not combinations:
        raise ValueError("No scene count combinations were requested")
    if episodes < len(combinations):
        raise ValueError("episodes must cover every requested count combination at least once")
    rng = np.random.default_rng(seed)
    quotient, remainder = divmod(episodes, len(combinations))
    extras = set(rng.permutation(len(combinations))[:remainder].tolist())
    plan = [configuration for index, configuration in enumerate(combinations)
            for _ in range(quotient + int(index in extras))]
    rng.shuffle(plan)
    return plan


def count_label(configuration):
    objects, receptacles = configuration
    label = f"objects={objects}"
    if receptacles is not None:
        label += f",receptacles={receptacles}"
    return label


def count_distribution(configurations):
    return dict(sorted(Counter(count_label(value) for value in configurations).items()))


def metadata_counts(metadata):
    return (len(metadata.get("objects", [])),
            len(metadata.get("receptacles", [])) if metadata.get("task") == "place" else None)
