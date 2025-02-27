# python main.py --phase predict --checkpoint lightning_logs/0.0/version_0/checkpoints/epoch=195-step=4704.ckpt --devices 0 --test-dataset roi_predict
python main.py --phase predict --checkpoint lightning_logs/weight2/version_9/checkpoints/epoch=133-step=6298.ckpt --devices 2 --test-dataset wb_predict
