#!/bin/bash

# for imbalance in $(seq 0 0.2 1.0); do
#   for run in {1..5}; do
#     # Update the imbalance value in the config.yaml file
#     sed -i "s/imbalance: .*/imbalance: $imbalance/" config.yaml
    
#     # Run the Python script
#     python main.py --phase train --cfg config.yaml --devices 4
#   done
# done

# for run in {1..5}; do
#   # Run the Python script
#   python main.py --phase train --cfg config.yaml --devices 4
# done

for version_dir in lightning_logs/0.0/version_*; do
    for checkpoint in "${version_dir}/checkpoints/"epoch*.ckpt; do
        python main.py --phase train --cfg config.yaml --devices 4 --checkpoint $checkpoint
    done
done