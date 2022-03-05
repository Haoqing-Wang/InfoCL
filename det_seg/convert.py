import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    obj = torch.load(input, map_location="cpu")

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder."):
            continue
        old_k = k
        k = k.replace("encoder.", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t+1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("ds_conv1", "shortcut")
        k = k.replace("ds_bn1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "contrastive_learning", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)