import torch
step=192
# window
window = torch.ones(256)
window[: 256 - step] = torch.arange(256 - step) / (256 - step)
window = torch.tile(window.unsqueeze(1), (1, 256))
window = (
    (window * torch.rot90(window) * torch.rot90(window, 2) * torch.rot90(window, 3))
)

pred = torch.zeros(448, 448)
pred[:256, :256] += window
pred[:256, -256:] += window
pred[-256:, :256] += window
pred[-256:, -256:] += window
pred = pred[64:-64, 64:-64]

print(pred.mean())
print(pred.std())
