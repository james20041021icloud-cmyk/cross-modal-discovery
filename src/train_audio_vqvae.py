"""
Step 2b: Train the audio Soft-VQ-VAE on 1-second windows of the video's audio.
"""
import argparse, math, json, time
from pathlib import Path
import random

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np

from vqvae_audio import AudioSoftVQVAE


SAMPLE_RATE = 16000
CLIP_LEN    = 16000  # 1 second


class AudioWindows(Dataset):
    """Random 1-second crops from a single long WAV, with a fixed number
    of virtual samples per epoch."""
    def __init__(self, wav_path, n_virtual=None, clip_len=CLIP_LEN):
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
        self.wav = wav.astype(np.float32)
        # normalise once
        peak = np.abs(self.wav).max()
        if peak > 0: self.wav /= peak
        self.wav *= 0.95
        self.clip_len = clip_len
        self.n = n_virtual if n_virtual else max(1, (len(self.wav)-clip_len)//clip_len)
        print(f"Audio: {len(self.wav)/SAMPLE_RATE:.1f} s, {self.n} virtual clips/epoch")

    def __len__(self): return self.n

    def __getitem__(self, _):
        if len(self.wav) >= self.clip_len:
            s = random.randint(0, len(self.wav) - self.clip_len)
            x = self.wav[s:s+self.clip_len]
        else:
            x = np.pad(self.wav, (0, self.clip_len - len(self.wav)))
        return torch.from_numpy(x[None])   # (1, T)


def cosine_temp(step, total, t0=1.0, t1=0.1):
    p = step / max(total - 1, 1)
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--steps",      type=int, default=2000)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--codebook",   type=int, default=256)
    ap.add_argument("--hidden",     type=int, default=128)
    ap.add_argument("--latent",     type=int, default=64)
    ap.add_argument("--log_every",  type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--n_virtual",  type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out_dir); (out/"samples").mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    ds = AudioWindows(args.audio_path, n_virtual=args.n_virtual)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)

    model = AudioSoftVQVAE(in_channels=1, hidden=args.hidden,
                           latent_dim=args.latent,
                           num_embeddings=args.codebook).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    log, t0, it = [], time.time(), iter(loader)
    print(f"Training audio VQ-VAE for {args.steps} steps ...")
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
            print(f"[{step:4d}] loss={e['loss']:.5f}  recon={e['recon']:.5f}  "
                  f"ppl={e['ppl']:5.1f}/{args.codebook}  "
                  f"{time.time()-t0:.0f}s")

        if step % args.save_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                sf.write(out/f"samples/orig_{step:04d}.wav",
                         x[0,0].float().cpu().numpy(), SAMPLE_RATE)
                sf.write(out/f"samples/recon_{step:04d}.wav",
                         recon[0,0].float().cpu().numpy(), SAMPLE_RATE)
            model.train()

    torch.save({"step": args.steps, "model": model.state_dict(),
                "args": vars(args)}, out/"audio_vqvae.pt")
    with open(out/"train_log.json", "w") as f: json.dump(log, f, indent=2)
    print(f"\n✅ Saved model to {out/'audio_vqvae.pt'}")


if __name__ == "__main__":
    main()
