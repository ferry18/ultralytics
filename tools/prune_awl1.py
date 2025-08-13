import argparse
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import model_info


def l1_norm_prune(model: nn.Module, ratio: float = 0.3):
    """Prune lowest-L1-norm filters per Conv2d layer (excluding Detect)."""
    m = deepcopy(model).cpu()
    conv_layers = []
    for name, module in m.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            conv_layers.append((name, module))
    for name, conv in conv_layers:
        w = conv.weight.data
        oc = w.shape[0]
        keep = max(1, int(oc * (1 - ratio)))
        scores = w.abs().view(oc, -1).sum(dim=1)
        _, idx = torch.topk(scores, k=keep, largest=True)
        idx, _ = torch.sort(idx)
        conv.weight = nn.Parameter(w[idx].contiguous())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias.data[idx].contiguous())
        conv.out_channels = keep
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, help='model yaml or pt')
    ap.add_argument('--weights', type=str, default=None)
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--out', type=str, default='pruned.pt')
    args = ap.parse_args()

    model = DetectionModel(args.model)
    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    print('Original model:')
    model_info(model)

    pruned = l1_norm_prune(model, ratio=args.ratio)
    print('Pruned model:')
    model_info(pruned)

    torch.save({'model': pruned}, args.out)
    print('Saved to', args.out)


if __name__ == '__main__':
    main()