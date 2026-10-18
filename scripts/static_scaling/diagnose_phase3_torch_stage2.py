"""Save PyTorch internals of stage2.block0 for the Phase 3 Flux diagnostic."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_torch as mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    mt.configure_backend(num_threads=1)
    device = mt.resolve_device(args.device)
    model = mt.make_model().to(device)
    mt.load_bundle(model, args.bundle)
    ref = np.load(args.reference, allow_pickle=False)
    x = torch.from_numpy(np.ascontiguousarray(ref["act.stage1.block1"])).to(device)
    blk = model.stage2.block0
    with torch.no_grad():
        conv1 = blk.conv1(x)
        gn1 = blk.gn1(conv1)
        r1 = F.relu(gn1)
        conv2 = blk.conv2(r1)
        gn2 = blk.gn2(conv2)
        proj = blk.proj(x)
        pgn = blk.proj_gn(proj)
        total = gn2 + pgn
        out = F.relu(total)
    arrays = {
        "conv1": conv1, "gn1": gn1, "relu1": r1, "conv2": conv2,
        "gn2": gn2, "proj": proj, "pgn": pgn, "total": total, "out": out,
    }
    np.savez(args.out, **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
    for k, v in arrays.items():
        print("TORCH_STAGE2 %-6s shape=%s max_abs=%.9g mean_abs=%.9g" %
              (k, tuple(v.shape), float(v.abs().max()), float(v.abs().mean())))
    print("TORCH_STAGE2_SAVED", args.out)


if __name__ == "__main__":
    main()
