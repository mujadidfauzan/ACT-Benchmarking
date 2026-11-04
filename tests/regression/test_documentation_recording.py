"""High-res documentation must not replace policy pixels or alter actions."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts import record_policy_evaluation as recording


class DocumentationRecordingTests(unittest.TestCase):
    def test_controller_scaling_clipping_and_forecast_origin(self):
        controller=SimpleNamespace(input_type="delta",input_ref_frame="world",
            input_min=np.full(6,-1.),input_max=np.ones(6),
            output_min=np.array([-.05]*3+[-.5]*3),output_max=np.array([.05]*3+[.5]*3))
        actions=np.array([[1,0,-1,0,0,0,-1],[2,.5,0,0,0,0,1.]])
        original=actions.copy()
        points=recording.integrated_command_path(controller,actions,[0,0,1],(-np.ones(7),np.ones(7)))
        np.testing.assert_allclose(points,[[0,0,1],[.05,0,.95],[.1,.025,.95]])
        np.testing.assert_array_equal(actions,original)
        controller.input_ref_frame="base"
        with self.assertRaisesRegex(ValueError,"world-frame"):
            recording.integrated_command_path(controller,actions,[0,0,1],(-np.ones(7),np.ones(7)))

    def test_projection_rejects_behind_camera_and_does_not_join_offscreen_segments(self):
        points=np.array([[0,0,1],[2,0,1],[0,0,-1],[.5,.5,1]])
        pixels,visible=recording.project_points(points,np.eye(4),2,2)
        np.testing.assert_array_equal(visible,[True,False,False,True])
        np.testing.assert_allclose(pixels[-1],[.5,.5])
        frame=np.zeros((100,100,3),np.uint8)
        recording.draw_path(frame,np.array([[10,10,1],[20,20,1]]),np.eye(4),predicted=True)
        self.assertTrue(frame.any())

    def test_policy_capture_uses_existing_output_without_second_inference(self):
        action=torch.zeros(1,7)
        output=dict(action=action,hidden=object())
        model=SimpleNamespace(step=lambda *a,**k:output)
        captured=[]
        proxy=recording.RecordingPolicy(model,SimpleNamespace(capture=lambda v:captured.append(v)))
        actual=proxy.step(image=torch.zeros(1,3,128,128),proprio=torch.zeros(1,16))
        self.assertIs(actual,output)
        self.assertIs(captured[0],output)

    def test_native_size_video_and_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'native.mp4'
            args=SimpleNamespace(width=320,height=180,fps=20.,encoder='opencv',crf=18)
            writer=recording.Video(path,args)
            frame=np.zeros((180,320,3),np.uint8);frame[:90,:,0]=255;frame[90:,:,2]=255
            for _ in range(3):writer.write(frame)
            writer.close()
            video=cv2.VideoCapture(str(path));ok,bgr=video.read()
            self.assertTrue(ok)
            self.assertEqual(bgr.shape,(180,320,3))
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),3)
            self.assertGreater(bgr[20,:,2].mean(),200)  # RGB red is BGR channel 2.
            self.assertGreater(bgr[150,:,0].mean(),200)
            video.release()

    def test_selected_episode_replays_preceding_rollouts_and_closes_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'video'
            flags=['record','--checkpoint','chosen.pt','--output-dir',str(output),'--seed','2000',
                   '--episode','2','--max-steps','3','--width','320','--height','180','--device','cpu']
            model=SimpleNamespace(policy_type='act_lerobot',camera_names=('agentview','robot0_eye_in_hand'),
                                  observation_image_size=128,execution_steps=1,temporal_ensemble=False)
            checkpoint=dict(updates=123,data_config=dict(scene_counts=[[1,1]],fixed_target_color='red'))
            env=SimpleNamespace(control_freq=20,close=lambda:None)
            fake=SimpleNamespace(capture=lambda **k:None,close=lambda:None,
                                 videos={'agentview':(SimpleNamespace(encoder='fake'),)})
            result=dict(success=True,failure_reason=None,steps=3,initial_positions={})
            with patch('sys.argv',flags),patch.object(recording,'load_policy',return_value=(model,None,checkpoint,0)), \
                    patch.object(recording,'build_environment',return_value=env), \
                    patch.object(recording,'Recorder',return_value=fake), \
                    patch.object(recording,'evaluate_episode',return_value=result) as evaluator:
                recording.main()
            self.assertEqual(evaluator.call_count,3)
            self.assertIs(evaluator.call_args_list[0].args[0],env)
            self.assertIs(evaluator.call_args_list[1].args[0],env)
            self.assertIsInstance(evaluator.call_args_list[2].args[0],recording.RecordingEnvironment)
            self.assertEqual(evaluator.call_args_list[2].args[3],128)
            self.assertIsNone(evaluator.call_args_list[2].args[12])
            self.assertEqual(list(output.iterdir()),[])

    def test_recorder_outputs_only_videos_and_annotations_only_for_act(self):
        class FakeVideo:
            def __init__(self,path,args):
                self.path=path
                path.write_bytes(b'video placeholder')
                self.frames=0
            def write(self,frame):
                self.frames+=1
            def close(self):
                pass

        for kind,annotated in [('act_lerobot',True),('visual_bc',False),('robomimic_bc_rnn',False)]:
            with self.subTest(policy=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                args=SimpleNamespace(output_dir=root,cameras=list(recording.CAMERAS),width=320,height=180,
                    trajectory_overlay=True,trajectory_points=32,seed=2000,episode=1)
                rendered=[]
                def render(**kwargs):
                    rendered.append(kwargs['camera_name'])
                    return np.zeros((180,320,3),np.uint8)
                env=SimpleNamespace(control_freq=20,sim=SimpleNamespace(render=render))
                model=SimpleNamespace(policy_type=kind)
                norm=SimpleNamespace(denormalize_action=lambda x:x)
                with patch.object(recording,'Video',FakeVideo), \
                        patch('robosuite.utils.camera_utils.get_camera_transform_matrix',return_value=np.eye(4)):
                    recorder=recording.Recorder(env,model,norm,args)
                    recorder.observation={'robot0_eef_pos':np.zeros(3)}
                    recorder.capture(dict(action=torch.zeros(1,7),hidden=None))
                    recorder.capture(final=dict(success=True,failure_reason=None))
                    recorder.close()
                expected={f'{camera}.mp4' for camera in recording.CAMERAS}
                if annotated:expected|={f'{camera}_annotated.mp4' for camera in recording.CAMERAS}
                self.assertEqual({p.name for p in root.iterdir()},expected)
                self.assertIn('frontview',rendered)


if __name__=='__main__':
    unittest.main(verbosity=2)
