# probablistic-skin-lesion-segmentation

This work is based on a [pytorch implementation](https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch) of [hierarchical probabilistic unet](https://arxiv.org/abs/1905.13077v1)

## Data directory
```
data
├── predict
└── train
    ├── images
    |   ├── first
    |   ├── second
    |   ...
    └── labels
        ├── labeller 1
        |   ├── first
        |   ├── second
        |   ...
        ├── labeller 2
        ...
```

## Scripts
Train model.

`bash scripts/train.sh`

Make a report on test dataset.

`bash scripts/test.sh`

Generate predicted skin lesion segmentation.

`bash scripts/predict.sh`

### To do
Mixed precision
Pruning
Tensorrt