#!/usr/bin/env bash

source setup_for_standard.bash

python exec_standard.py \
    --config_manager 'config/manager/trust_dataset_collection_v1.py' \
    --config_world 'config/world/default_world.py' \
    --seed 5 \
    --duration 10 \
    --remove_data
