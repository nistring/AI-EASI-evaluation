python main.py --mc_n 1024 --phase test --multi --checkpoint weights/multi/checkpoints/epoch=1324-step=63600.ckpt --batch_size 128 --devices 0
python main.py --mc_n 512 --phase test --multi --checkpoint weights/multi/checkpoints/epoch=1324-step=63600.ckpt --batch_size 128 --devices 0
python main.py --mc_n 256 --phase test --multi --checkpoint weights/multi/checkpoints/epoch=1324-step=63600.ckpt --batch_size 128 --devices 0
python main.py --mc_n 128 --phase test --multi --checkpoint weights/multi/checkpoints/epoch=1324-step=63600.ckpt --batch_size 128 --devices 0

python main.py --mc_n 1024 --phase test --checkpoint weights/HPU/checkpoints/epoch=1271-step=61056.ckpt --batch_size 128 --devices 1
python main.py --mc_n 512 --phase test --checkpoint weights/HPU/checkpoints/epoch=1271-step=61056.ckpt --batch_size 128 --devices 1
python main.py --mc_n 256 --phase test --checkpoint weights/HPU/checkpoints/epoch=1271-step=61056.ckpt --batch_size 128 --devices 1
python main.py --mc_n 128 --phase test --checkpoint weights/HPU/checkpoints/epoch=1271-step=61056.ckpt --batch_size 128 --devices 1