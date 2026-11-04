"""Shared collection and expert-gate options for fixed-order stacking."""

import numpy as np

from environments.stack_env import StackEnv


def add_stack_arguments(parser):
    parser.add_argument('--stack-order', nargs='+', choices=tuple(StackEnv.COLOR_PALETTE),
                        help='Stack only: fixed colors from bottom to top; omitted keeps random order')
    parser.add_argument('--stack-workspace-half-size', type=float,
                        help='Stack only: randomize X and Y within +/- this distance in meters')


def stack_environment_kwargs(order=None, workspace=None):
    settings = {}
    if order is not None:
        if not 2 <= len(order) <= 4 or len(set(order)) != len(order):
            raise ValueError('Stack order requires 2–4 distinct colors, bottom to top')
        if not set(order) <= set(StackEnv.COLOR_PALETTE):
            raise ValueError('Unknown stack color')
        settings['fixed_stack_order'] = list(order)
    if workspace is not None:
        if not np.isfinite(workspace) or not .02 < workspace <= .35:
            raise ValueError('Stack workspace half-size must be finite and in (0.02, 0.35] meters')
        settings.update(x_range=(-workspace, workspace), y_range=(-workspace, workspace))
    return settings


def validate_stack_arguments(args, object_counts):
    order, workspace = args.stack_order, args.stack_workspace_half_size
    if (order is not None or workspace is not None) and args.task != 'stack':
        raise ValueError('--stack-order and --stack-workspace-half-size require --task stack')
    settings = stack_environment_kwargs(order, workspace)
    if order is not None and any(n != len(order) for n in object_counts):
        raise ValueError('--stack-order length must match every requested object count')
    return settings
