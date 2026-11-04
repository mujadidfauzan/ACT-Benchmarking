"""Fixed semantic targets and physical stack success, without simulator startup."""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import torch

from environments.stack_env import StackEnv
from environments.stack_config import stack_environment_kwargs, validate_stack_arguments, validate_fixed_stack_prepared
from experts.stack_expert import StackExpert
from experts.primitives.release import release_object
from scripts.record_demonstrations import parse_args, build_task
from scripts.evaluate_visual_bc import evaluate_episode, build_environment


BASE = 'robosuite.environments.manipulation.manipulation_env.ManipulationEnv'


def fixed_env():
    with patch(f'{BASE}.__init__', return_value=None):
        env = StackEnv(num_cubes=3, fixed_stack_order=['red', 'green', 'blue'])
    env.rng = np.random.default_rng(7)
    env._sample_task_metadata()
    return env


def tower():
    env = fixed_env()
    positions = {'red': np.array([0., 0., .82]), 'green': np.array([0., 0., .86]),
                 'blue': np.array([0., 0., .90])}
    env.get_object_position = lambda color: positions[color]
    env.cubes = [SimpleNamespace(name=name, joints=[name]) for name in env.object_ids]
    env.object_body_ids = {name: i for i, name in enumerate(env.object_ids)}
    env.robots = [SimpleNamespace(arms=['right'], gripper={'right': 'gripper'},
                                  eef_site_id={'right': 0})]
    env.sim = SimpleNamespace(data=SimpleNamespace(site_xpos=np.array([[0, 0, 1.1]]),
        body_xmat=np.stack([np.eye(3).flatten()] * 3),
        get_joint_qvel=lambda joint: np.zeros(6)))
    env.check_contact = lambda first, second: first != 'gripper'
    env._last_gripper_action = -1.
    return env, positions


class StackProtocolTests(unittest.TestCase):
    def test_fixed_order_randomizes_ids_and_keeps_instruction_consistent(self):
        env = fixed_env()
        assignments = set()
        for _ in range(10):
            env._sample_task_metadata()
            metadata = env.get_task_metadata()
            self.assertEqual(metadata['stack_order'], ['red', 'green', 'blue'])
            self.assertEqual(metadata['fixed_stack_order'], metadata['stack_order'])
            self.assertEqual(set(env.object_id_to_color.values()), {'red', 'green', 'blue'})
            assignments.add(tuple(env.object_id_to_color.values()))
        self.assertGreater(len(assignments), 1)
        with patch(f'{BASE}.__init__', return_value=None):
            for order in (['red', 'red', 'blue'], ['red', 'green'], ['red', 'green', 'bad']):
                with self.assertRaises(ValueError):
                    StackEnv(num_cubes=3, fixed_stack_order=order)

    def test_stack_cli_scope_counts_and_builder(self):
        argv = ['record', '--task', 'stack', '--num-objects', '3', '--stack-order',
                'red', 'green', 'blue', '--stack-workspace-half-size', '0.16']
        with patch('sys.argv', argv):
            args = parse_args()
        settings = validate_stack_arguments(args, [3])
        self.assertEqual(settings['x_range'], (-.16, .16))
        with self.assertRaises(ValueError):
            validate_stack_arguments(args, [2, 3])
        args.task = 'place'
        with self.assertRaises(ValueError):
            validate_stack_arguments(args, [3])
        args.task = 'stack'
        with patch('scripts.record_demonstrations.StackEnv') as constructor, patch(
                'scripts.record_demonstrations.PandaController'), patch(
                'scripts.record_demonstrations.GraspExpert'), patch('scripts.record_demonstrations.StackExpert'):
            build_task(args, 3)
        self.assertEqual(constructor.call_args.kwargs['fixed_stack_order'], ['red', 'green', 'blue'])
        self.assertEqual(constructor.call_args.kwargs['y_range'], (-.16, .16))
        for size in (float('nan'), -.1, .4):
            with self.assertRaises(ValueError):
                stack_environment_kwargs(workspace=size)

    def test_geometry_alone_cannot_satisfy_success(self):
        env, positions = tower()
        self.assertTrue(env.stack_status()['success'])
        self.assertEqual(env.stack_status()['stacked_cubes'], 3)
        self.assertFalse(env._check_success())  # Instantaneous tower needs hold time.
        env._stack_hold_steps = 20
        self.assertTrue(env._check_success())
        env._last_gripper_action = 1
        self.assertFalse(env._check_success())
        env._last_gripper_action = -1
        env.check_contact = lambda first, second: False
        self.assertFalse(env.stack_status()['success'])  # Aligned but unsupported.
        env.check_contact = lambda first, second: True
        self.assertFalse(env.stack_status()['success'])  # Gripper still touching.
        env.check_contact = lambda first, second: first != 'gripper'
        env.sim.data.get_joint_qvel = lambda joint: np.array([.1, 0, 0, 0, 0, 0])
        self.assertFalse(env.stack_status()['success'])
        env.sim.data.get_joint_qvel = lambda joint: np.zeros(6)
        positions['blue'][0] = .04
        self.assertEqual(env.stack_status()['stacked_cubes'], 2)
        self.assertFalse(env._check_success())

    def test_hold_counter_is_consecutive_and_resets(self):
        env, _ = tower()
        with patch(f'{BASE}._post_action', return_value=(0., False, {})):
            for i in range(19):
                reward, _, _ = env._post_action([0] * 6 + [-1])
                self.assertEqual(reward, 0.)
            reward, _, info = env._post_action([0] * 6 + [-1])
            self.assertEqual(reward, 1.)
            self.assertEqual(info['stack']['stable_success_steps'], 20)
            env._post_action([0] * 6 + [1])
            self.assertEqual(env._stack_hold_steps, 0)

    def test_expert_rejects_transient_tower_and_accepts_stable_prefix(self):
        env, _ = tower()
        expert = StackExpert(SimpleNamespace(env=env), None)
        expert._stable_steps = 19
        self.assertFalse(expert.verify_stack({}, ['red', 'green'])['success'])
        expert._stable_steps = 20
        self.assertTrue(expert.verify_stack({}, ['red', 'green'])['success'])
        self.assertFalse(expert.verify_stack({})['success'])
        env._stack_hold_steps = 20
        self.assertTrue(expert.verify_stack({})['success'])

    def test_adaptive_release_waits_for_consecutive_ready_and_rejects_timeout(self):
        env = SimpleNamespace(step=Mock(return_value=({}, 0., False, {})))
        controller = SimpleNamespace(env=env, GRIPPER_OPEN=-1., set_phase=Mock(),
            create_action=Mock(return_value=np.zeros(7)),
            get_eef_position=Mock(return_value=np.array([0., 0., 1.])),
            move_to_pose=Mock(return_value=({}, True)), open_gripper=Mock(return_value={}))
        # A brief contact-free interval is interrupted; require three new ready steps.
        predicate = Mock(side_effect=[False, True, True, False, True, True, True])
        result = release_object({}, controller, [0, 0, 0, 1], open_steps=50,
                                release_ready=predicate)
        self.assertTrue(result.success)
        self.assertEqual(env.step.call_count, 7)
        controller.open_gripper.assert_not_called()
        controller.move_to_pose.assert_called_once()
        env.step.reset_mock()
        controller.move_to_pose.reset_mock()
        result = release_object({}, controller, [0, 0, 0, 1], open_steps=8,
                                release_ready=lambda obs: False)
        self.assertFalse(result.success)
        self.assertEqual(result.stage, 'release')
        self.assertEqual(env.step.call_count, 8)
        controller.move_to_pose.assert_not_called()
        # Shared primitive retains the fixed opening behavior for non-Stack callers.
        release_object({}, controller, [0, 0, 0, 1])
        controller.open_gripper.assert_called_once_with({}, steps=50, render=False)

    def test_release_requires_open_fingers_and_no_cube_contact(self):
        env, _ = tower()
        controller = SimpleNamespace(env=env, gripper_qpos_key='robot0_gripper_qpos')
        expert = StackExpert(controller, None)
        self.assertFalse(expert._release_ready({'robot0_gripper_qpos': [.01, -.01]}))
        obs = {'robot0_gripper_qpos': [.04, -.04]}
        self.assertTrue(expert._release_ready(obs))
        env.check_contact = lambda gripper, cube: True
        self.assertFalse(expert._release_ready(obs))

    def test_settle_stops_after_stable_hold_and_keeps_maximum_timeout(self):
        env, _ = tower()
        env.step = Mock(return_value=({}, 0., False, {}))
        controller = SimpleNamespace(env=env, GRIPPER_OPEN=-1., set_phase=Mock(),
                                     create_action=Mock(return_value=np.zeros(7)))
        expert = StackExpert(controller, None)
        status = [True] * 9 + [False] + [True] * 20
        env.stack_status = Mock(side_effect=[{'success': x} for x in status])
        expert._settle({}, False, ['red', 'green'])
        self.assertEqual(env.step.call_count, 30)
        self.assertEqual(expert._stable_steps, 20)
        env.step.reset_mock()
        env.stack_status = Mock(return_value={'success': False})
        expert._settle({}, False, ['red', 'green'])
        self.assertEqual(env.step.call_count, 60)
        self.assertEqual(expert._stable_steps, 0)

    def test_fixed_stack_checkpoint_contract_and_environment_restore(self):
        prepared = dict(tasks=['stack'], fixed_target_color=None, scene_counts=[[3, 0]],
            camera_sizes=[[128, 128]], environment_configs={'stack': dict(stack_success_version=2,
                environment_kwargs=dict(fixed_stack_order=['red', 'green', 'blue'],
                                        x_range=[-.16, .16], y_range=[-.16, .16]))})
        self.assertEqual(validate_fixed_stack_prepared(prepared), ['red', 'green', 'blue'])
        for mutate in ('random_order', 'wrong_count', 'old_criterion'):
            bad = copy.deepcopy(prepared)
            if mutate == 'random_order':
                bad['environment_configs']['stack']['environment_kwargs'].pop('fixed_stack_order')
            elif mutate == 'wrong_count':
                bad['scene_counts'] = [[2, 0]]
            else:
                bad['environment_configs']['stack']['stack_success_version'] = 1
            with self.assertRaises(ValueError):
                validate_fixed_stack_prepared(bad)
        with self.assertRaises(ValueError):
            validate_fixed_stack_prepared(prepared, 2)
        model = SimpleNamespace(camera_names=('agentview',))
        with patch('scripts.record_demonstrations.build_task', return_value=('env', None)) as factory:
            self.assertEqual(build_environment(321, 3, False, model, prepared, 1000), 'env')
        args = factory.call_args.args[0]
        self.assertEqual(args.max_steps, 1000)
        self.assertEqual(args.environment_kwargs['fixed_stack_order'], ['red', 'green', 'blue'])
        with patch('scripts.record_demonstrations.StackEnv') as constructor, patch(
                'scripts.record_demonstrations.PandaController'), patch(
                'scripts.record_demonstrations.GraspExpert'), patch('scripts.record_demonstrations.StackExpert'):
            build_task(args, 3)
        self.assertEqual(constructor.call_args.kwargs['horizon'], 1000)
        self.assertEqual(constructor.call_args.kwargs['fixed_stack_order'], ['red', 'green', 'blue'])

    def test_stack_rollout_single_hold_partial_progress_and_logs(self):
        # Exercise task dispatch and temporal bookkeeping, independently of physics.
        class Env:
            object_ids = ['cube_0', 'cube_1', 'cube_2']
            action_spec = (-np.ones(7), np.ones(7))
            full_at = 3
            def reset(self):
                self.t = 0
                return self.obs()
            def obs(self):
                return dict(agentview_image=np.zeros((32, 32, 3), dtype=np.uint8),
                            robot0_eef_pos=np.array([0, 0, 1.2]), robot0_gripper_qpos=[.04, -.04])
            def step(self, action):
                self.t += 1
                return self.obs(), 0., False, {}
            def get_task_metadata(self):
                return dict(task='stack', stack_order=['red', 'green', 'blue'])
            def resolve_object_id(self, color):
                return self.object_ids[['red', 'green', 'blue'].index(color)]
            def get_object_position(self, identifier):
                index = self.object_ids.index(identifier)
                return np.array([0, 0, .82 + (.04 if index == 1 and self.t else 0)
                                 + (.08 if index == 2 and self.t >= self.full_at else 0)])
            def stack_status(self, order=None):
                partial = order is not None and len(order) == 2
                ready = self.t >= (1 if partial else self.full_at)
                return dict(success=ready, stacked_cubes=3 if self.t >= self.full_at else 2,
                            released=True, gripper_open=True, stable=True, upright=True)
        model = SimpleNamespace(camera_names=('agentview',), temporal=True, uses_previous_gripper=False,
            step=lambda **kw: {'action': torch.tensor([[0., 0., 0., 0., 0., 0., -1.]])})
        normalizer = SimpleNamespace(proprio_keys=('obs__robot0_eef_pos',),
            normalize_proprio=lambda x: x, denormalize_action=lambda x: x)
        env = Env()
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'rollout.jsonl'
            result = evaluate_episode(env, model, normalizer, 32, torch.device('cpu'),
                                      30, 20, False, 0, None, 20., None, log)
            self.assertTrue(result['success'])
            self.assertEqual(result['steps'], 22)  # ready at 3 + 19, not a double hold.
            self.assertTrue(result['two_cube_stack_achieved'])
            rows = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual(set(rows[-1]['object_positions_after']), set(env.object_ids))
            self.assertIn('stack', rows[-1])
        env.full_at = 1000
        result = evaluate_episode(env, model, normalizer, 32, torch.device('cpu'),
                                  30, 20, False, 0, None, 20., None)
        self.assertFalse(result['success'])
        self.assertTrue(result['two_cube_stack_achieved'])
        self.assertEqual(result['failure_reason'], 'incomplete_stack')


if __name__ == '__main__':
    unittest.main()
