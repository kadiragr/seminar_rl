# Seaquest - QR-DQN Experiments

This project contains several QR-DQN experiments for the Atari game Seaquest.

## Training

All experiments are run using the same command structure:

```bash
python3 utils/train_custom.py --env SeaquestNoFrameskip-v4 --algo qrdqn --conf experiments/<CONFIG>.yml --seed <SEED>

Example (baseline):

python3 utils/train_custom.py --env SeaquestNoFrameskip-v4 --algo qrdqn --conf experiments/seaquest_baseline.yml --seed 123
