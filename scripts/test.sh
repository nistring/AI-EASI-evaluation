# for dir in 0.0_both; do # 0.0_deterministic 0.0_no_ordinal_reg; do
#     for version_dir in lightning_logs/${dir}/version_*; do
#         for checkpoint in "${version_dir}/checkpoints/"*.ckpt; do
#         if [ -f "$checkpoint" ]; then
#             for dataset in "19_int" "19_ext"; do
#             python main.py --phase test --checkpoint "$checkpoint" --devices 1 --test-dataset "$dataset" --cfg "$version_dir/config.yaml"
#             done
#         fi
#         done
#     done
# done

for test_dataset in "19_int" "19_ext"; do
    for version_dir in lightning_logs/0.0_whole/version_*; do
        for checkpoint in "${version_dir}/checkpoints/"epoch*.ckpt; do
        if [ -f "$checkpoint" ]; then
            python main.py --phase test --checkpoint "$checkpoint" --devices 0 --test-dataset "$test_dataset" --cfg "$version_dir/config.yaml"
        fi
        done
    done
done