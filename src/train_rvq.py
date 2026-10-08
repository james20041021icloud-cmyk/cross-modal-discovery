"""
Unified training entrypoint for RVQ VQ-VAE on images or audio.
"""
import argparse, math, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import soundfile as sf

from vqvae_rvq_models import ImageRVQVAE, AudioRVQVAE


SR, CLIP = 16000, 16000

# ── datasets ──────────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, root, size=128):
        self.paths = sorted(Path(root).glob("frame_*.jpg"))
        self.tf = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class AudioWindows(Dataset):
    def __init__(self, wav_path, n_virtual=4000, clip_len=CLIP):
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1: wav = wav.mean(1)
        assert sr == SR
        self.wav = wav.astype(np.float32)
        peak = np.abs(self.wav).max()
        if peak > 0: self.wav /= peak
        self.wav *= 0.95
        self.clip_len = clip_len
        self.n = n_virtual
    def __len__(self): return self.n
    def __getitem__(self, _):
        if len(self.wav) >= self.clip_len:
            s = random.randint(0, len(self.wav) - self.clip_len)
            x = self.wav[s:s+self.clip_len]
        else:
            x = np.pad(self.wav, (0, self.clip_len - len(self.wav)))
        return torch.from_numpy(x[None])


def cosine_temp(step, total, t0=1.0, t1=0.1):
    p = step / max(total - 1, 1)
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["image", "audio"], required=True)
    ap.add_argument("--data_path", required=True,
                    help="frames directory (image) or wav file (audio)")
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--num_levels",     type=int, default=2)
    ap.add_argument("--codes_per_level", type=int, default=256)
    ap.add_argument("--hidden",   type=int, default=128)
    ap.add_argument("--latent",   type=int, default=64)
    ap.add_argument("--size",     type=int, default=128)      # images only
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--steps",    type=int, default=3000)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--log_every",  type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--n_virtual",  type=int, default=4000)   # audio only
    args = ap.parse_args()

    out = Path(args.out_dir); (out/"samples").mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Modality: {args.modality}   levels={args.num_levels}   "
          f"K/level={args.codes_per_level}   total_codes={args.num_levels*args.codes_per_level}")

    if args.modality == "image":
        ds = FrameDataset(args.data_path, size=args.size)
        print(f"Frames: {len(ds)}")
        model = ImageRVQVAE(3, args.hidden, args.latent,
                            args.num_levels, args.codes_per_level).to(device)
    else:
        ds = AudioWindows(args.data_path, n_virtual=args.n_virtual)
        print(f"Audio virtual clips/epoch: {len(ds)}")
        model = AudioRVQVAE(1, args.hidden, args.latent,
                            args.num_levels, args.codes_per_level).to(device)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    log, t0, it = [], time.time(), iter(loader)
    print(f"Training for {args.steps} steps ...")
    for step in range(args.steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader); x = next(it)
        x = x.to(device)
        temp = cosine_temp(step, args.steps)
        recon, vq_loss, ppls, _ = model(x, temp)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.log_every == 0:
            ppl_str = "/".join(f"{p.item():.0f}" for p in ppls)
            e = dict(step=step, loss=round(loss.item(),5),
                     recon=round(recon_loss.item(),5),
                     vq=round(vq_loss.item(),5),
                     ppl=ppl_str)
            log.append(e)
            print(f"[{step:4d}] loss={e['loss']:.4f}  recon={e['recon']:.4f}  "
                  f"vq={e['vq']:.4f}  ppl={e['ppl']}/{args.codes_per_level}  "
                  f"{time.time()-t0:.0f}s")

        if step % args.save_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                if args.modality == "image":
                    n = min(8, x.shape[0])
                    grid = torch.cat([x[:n], recon[:n].float()])
                    save_image(grid * 0.5 + 0.5,
                               out/f"samples/recon_{step:04d}.png", nrow=n)
                else:
                    sf.write(out/f"samples/orig_{step:04d}.wav",
                             x[0,0].float().cpu().numpy(), SR)
                    sf.write(out/f"samples/recon_{step:04d}.wav",
                             recon[0,0].float().cpu().numpy(), SR)
            model.train()

    torch.save({"step": args.steps, "model": model.state_dict(),
                "args": vars(args)},
               out/f"{args.modality}_rvq.pt")
    with open(out/"train_log.json", "w") as f: json.dump(log, f, indent=2)
    print(f"\n✅ Saved: {out/(args.modality+'_rvq.pt')}")


if __name__ == "__main__":
    main()
