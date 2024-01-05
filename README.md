# AI-EASI-evaluation

This work is based on a [pytorch implementation](https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch) of [hierarchical probabilistic unet](https://arxiv.org/abs/1905.13077v1)

# Docker
```
docker pull nistring/ai_easi_evaluation:latest
sudo docker run -p 8501:8501 -it --gpus "device=0" -v $(pwd) nistring/ai_easi_evaluation streamlit run app.py
```
Open `http://localhost:8501/`
