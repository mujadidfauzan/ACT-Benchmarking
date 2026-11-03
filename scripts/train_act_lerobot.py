"""Train the pinned LeRobot ACT baseline on native HDF5; no conversion required."""

from models.act_lerobot import LeRobotACTPolicy
from scripts import train_act

DEFAULTS = {**train_act.DEFAULTS, "learning_rate": 1e-5, "backbone_learning_rate": 1e-5,
            "decoder_layers": 1}


def parse_args():
    return train_act.parse_args(DEFAULTS)


def main():
    return train_act.main(policy_class=LeRobotACTPolicy, default_overrides=DEFAULTS,
                          evaluation_module="scripts.evaluate_act_lerobot", require_pretrained=True)


if __name__ == "__main__":
    main()
