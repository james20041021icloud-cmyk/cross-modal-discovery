"""
Step 2a: Train the visual Soft-VQ-VAE on frames extracted from the video.
Same model as the AFHQ / CIFAR runs. Saves the trained checkpoint.
"""
import argparse, math, json, time, os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from vqvae_model import SoftVQVAE


class FrameDataset(Dataset):
    def __init__(self, root, size=128):
        self.paths = sorted(Path(root).glob("frame_*.jpg"))
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def cosine_temp(step, total, t0=1.0, t1=0.1):
    p = step / max(total - 1, 1)
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--size",       type=int, default=128)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--steps",      type=int, default=2000)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--codebook",   type=int, default=512)
    ap.add_argument("--hidden",     type=int, default=128)
    ap.add_argument("--latent",     type=int, default=64)
    ap.add_argument("--log_every",  type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out_dir); (out/"samples").mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    ds = FrameDataset(args.frames_dir, size=args.size)
    print(f"Frames: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)

    model = SoftVQVAE(in_channels=3, hidden=args.hidden,
                      latent_dim=args.latent,
                      num_embeddings=args.codebook).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    log, t0, it = [], time.time(), iter(loader)
    print(f"Training visual VQ-VAE for {args.steps} steps ...")
    for step in range(args.steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader); x = next(it)
        x = x.to(device)
        temp = cosine_temp(step, args.steps)
        recon, vq_loss, ppl, _ = model(x, temp)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.log_every == 0:
            e = dict(step=step, loss=round(loss.item(),5),
                     recon=round(recon_loss.item(),5),
                     vq=round(vq_loss.item(),5),
                     ppl=round(ppl.item(),1))
            log.append(e)
            print(f"[{step:4d}] loss={e['loss']:.4f}  recon={e['recon']:.4f}  "
                  f"ppl={e['ppl']:5.1f}/{args.codebook}  "
                  f"{time.time()-t0:.0f}s")

        if step % args.save_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                n = min(8, x.shape[0])
                grid = torch.cat([x[:n], recon[:n].float()])
                save_image(grid * 0.5 + 0.5,
                           out/f"samples/recon_{step:04d}.png", nrow=n)
            model.train()

    torch.save({"step": args.steps, "model": model.state_dict(),
                "args": vars(args)}, out/"visual_vqvae.pt")
    with open(out/"train_log.json", "w") as f: json.dump(log, f, indent=2)
    print(f"\n✅ Saved model to {out/'visual_vqvae.pt'}")


if __name__ == "__main__":
    main()
