Seaquest - QR-DQN Experiments

This project contains several QR-DQN experiments for the Atari game Seaquest.

Training

All experiments are run using the same command structure:

python3 utils/train_custom.py --env SeaquestNoFrameskip-v4 --algo qrdqn --conf experiments/<CONFIG>.yml --seed <SEED>

Example (baseline): python3 utils/train_custom.py --env SeaquestNoFrameskip-v4 --algo qrdqn --conf experiments/seaquest_baseline.yml --seed 123

Each experiment was trained with 20 different seeds (e.g. 123, 234, 345, ...).

Configuration Files

Available configuration files:

seaquest_baseline.yml

seaquest_reward.yml

seaquest_actions_only_move_with_fire.yml

seaquest_rewardandaction.yml

To switch between experiments, only change the --conf argument.

Report

The full report and analysis can be found here: seminar/report/seaquestQRDQN.ipynb
