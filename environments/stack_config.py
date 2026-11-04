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


def validate_fixed_stack_prepared(prepared, num_objects=None):
    """Check that a language-free policy has exactly one unambiguous stack target."""
    if prepared.get('tasks') != ['stack']:
        raise ValueError('Expected a Stack-only dataset')
    collection = prepared.get('environment_configs', {}).get('stack', {})
    order = collection.get('environment_kwargs', {}).get('fixed_stack_order')
    if order is None:
        raise ValueError('Stack policies require a fixed --stack-order; random language targets are unsupported')
    stack_environment_kwargs(order)
    if any(int(count[0]) != len(order) or int(count[1]) != 0
           for count in prepared.get('scene_counts', [])) or not prepared.get('scene_counts'):
        raise ValueError('Stack scene counts must match the fixed color order')
    if num_objects is not None and num_objects != len(order):
        raise ValueError('Evaluation cube count must match the fixed Stack order')
    if collection.get('stack_success_version') != 2:
        raise ValueError('Stack requires the recorded physical success protocol (version 2)')
    return list(order)
