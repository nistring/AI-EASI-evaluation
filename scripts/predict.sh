# python main.py --phase predict --checkpoint lightning_logs/ROI_model/checkpoints/epoch=87-step=4136.ckpt --devices 2 --test-dataset roi_predict
python main.py --phase predict --checkpoint lightning_logs/whole_body_8_1/checkpoints/epoch=49-step=6200.ckpt --devices 2 --test-dataset wb_predict
