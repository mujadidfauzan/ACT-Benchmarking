"""Read-only checks of native observations, target timing, and full action chunks."""

import numpy as np
import torch
from torch.utils.data import default_collate


def audit_inputs(model, normalizer, dataset, device, sample_count=8):
    indices = np.linspace(0, len(dataset)-1, min(sample_count, len(dataset)), dtype=int).tolist()
    batch = {k: v.to(device) for k, v in default_collate([dataset[i] for i in indices]).items()}
    target, padding = batch.pop("action"), batch.pop("is_pad")
    raw_proprio = batch["proprio"].clone()
    samples = []
    for i, index in enumerate(indices):
        ep, start = dataset._sample_index[index]
        entry = dataset.episodes[ep]
        valid = min(model.chunk_size, int(entry["steps"])-start)
        with dataset._episode(ep) as stored:
            expected = torch.as_tensor(stored["actions"][start:start+valid], device=device)
            torch.testing.assert_close(target[i, :valid], expected, rtol=0, atol=0)
            state = torch.as_tensor(np.concatenate([stored[k][start] for k in normalizer.proprio_keys]),
                                    device=device, dtype=torch.float32)
            torch.testing.assert_close(raw_proprio[i], state, rtol=0, atol=0)
            for name in model.camera_names:
                key = "image" if name == "agentview" else "eye_in_hand_image"
                expected_image = torch.as_tensor(stored[f"obs__{name}_image"][start].transpose(2,0,1).copy(), device=device)
                torch.testing.assert_close(batch[key][i], expected_image, rtol=0, atol=0)
        if padding[i].sum() != model.chunk_size-valid:
            raise ValueError("Action padding does not match episode end")
        samples.append(dict(path=entry["path"], start=start, valid_actions=valid,
                            source_observation=start, target_start=start, target_stop=start+valid))
    normalized_target = normalizer.normalize_action(target)
    torch.testing.assert_close(normalizer.denormalize_action(normalized_target), target, rtol=1e-5, atol=1e-6)
    batch["proprio"] = normalizer.normalize_proprio(raw_proprio)
    mode = model.training
    try:
        model.eval()
        with torch.inference_mode():
            prediction = model(**batch)["action"]
            raw_prediction = normalizer.denormalize_action(prediction)
            shuffled = {**batch, "proprio": batch["proprio"].flip(0)}
            state_change = (model(**shuffled)["action"]-prediction).abs().mean().item()
            changed = {}
            for name in model.camera_names:
                key = "image" if name == "agentview" else "eye_in_hand_image"
                changed[name] = (model(**{**batch, key: batch[key].flip(0)})["action"]-prediction).abs().mean().item()
    finally:
        model.train(mode)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite full-chunk prediction during input audit")
    std = prediction[:, 0].std(0, unbiased=False)
    report = dict(samples=samples, image_source="current_RGB_only", observation_history=1,
                  proprio_keys=list(normalizer.proprio_keys), action_order=["dx","dy","dz","drx","dry","drz","gripper"],
                  normalization_roundtrip_pass=True, source_alignment_pass=True,
                  images={name: dict(dtype=str(batch["image" if name=="agentview" else "eye_in_hand_image"].dtype),
                                     shape=list(batch["image" if name=="agentview" else "eye_in_hand_image"].shape)) for name in model.camera_names},
                  camera_shuffle_mean_abs_delta=changed, proprio_shuffle_mean_abs_delta=state_change,
                  first_action_std_across_samples=std.cpu().tolist(),
                  chunk_std_across_offsets=prediction.std(1, unbiased=False).mean(0).cpu().tolist(),
                  raw_action_range=[raw_prediction.amin((0,1)).cpu().tolist(), raw_prediction.amax((0,1)).cpu().tolist()],
                  near_constant_first_action=bool(std.max() < 1e-6),
                  shuffle_note="Sensitivity diagnostic; not a success metric or proof of causal task understanding")
    arrays = dict(predicted_chunk_normalized=prediction.cpu().numpy(), predicted_chunk_raw=raw_prediction.cpu().numpy(),
                  target_chunk_raw=target.cpu().numpy(), target_chunk_normalized=normalized_target.cpu().numpy(),
                  is_pad=padding.cpu().numpy(), proprio_raw=raw_proprio.cpu().numpy(),
                  proprio_normalized=batch["proprio"].cpu().numpy(), sample_indices=np.array(indices))
    for name in model.camera_names:
        key = "image" if name == "agentview" else "eye_in_hand_image"
        arrays[f"image_{name}"] = batch[key].cpu().numpy()
    return report, arrays
