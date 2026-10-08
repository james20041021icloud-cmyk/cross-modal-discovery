"""
Soft-VQ-VAE on LJSpeech raw waveforms.
~13k single-speaker English clips, cut to 1-second 16 kHz chunks.
"""
import os, math, json, time, argparse, glob, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import soundfile as sf

from vqvae_audio import AudioSoftVQVAE


SAMPLE_RATE = 16000          # downsampled from native 22050
CLIP_LEN    = 16000          # 1 second


class LJSpeechClips(Dataset):
    """Yields random 1-second 16 kHz mono waveform chunks from LJSpeech wavs."""
    def __init__(self, root, sample_rate=SAMPLE_RATE, clip_len=CLIP_LEN, limit=None):
        self.paths = sorted(glob.glob(os.path.join(root, 'wavs', '*.wav')))
        if limit:
            self.paths = self.paths[:limit]
        self.sample_rate = sample_rate
        self.clip_len = clip_len
        self.resamplers = {}
        print(f"LJSpeech: {len(self.paths)} source clips")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.paths[idx])         # (C, T)
        wav = wav.mean(0, keepdim=True)                    # mono
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self.resamplers[sr](wav)
        # Random 1-second crop (or pad if too short)
        if wav.shape[-1] >= self.clip_len:
            start = random.randint(0, wav.shape[-1] - self.clip_len)
            wav = wav[:, start:start + self.clip_len]
        else:
            wav = F.pad(wav, (0, self.clip_len - wav.shape[-1]))
        # peak-normalise to [-0.95, 0.95] (close to Tanh range)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
        return wav                                          # (1, 16000)


def cosine_temp(step, total, t_start=1.0, t_end=0.1):
    p = step / max(total - 1, 1)
    return t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      default='data/LJSpeech-1.1')
    ap.add_argument('--out',       default='runs/ljspeech')
    ap.add_argument('--batch',     type=int, default=64)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--steps',     type=int, default=5000)
    ap.add_argument('--codebook',  type=int, default=512)
    ap.add_argument('--latent',    type=int, default=64)
    ap.add_argument('--hidden',    type=int, default=128)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--save_every',type=int, default=500)
    ap.add_argument('--workers',   type=int, default=4)
    args = ap.parse_args()

    os.makedirs(f'{args.out}/samples', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")
    print(f"Args: {vars(args)}\n")

    dataset = LJSpeechClips(args.data)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = AudioSoftVQVAE(
        in_channels=1, hidden=args.hidden, latent_dim=args.latent,
        num_embeddings=args.codebook,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Audio Soft-VQ-VAE: {n_params:.1f}M parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scaler = GradScaler()

    log, t0, loader_iter = [], time.time(), iter(loader)
    print(f"Training for {args.steps} steps (bf16 mixed-precision)...\n")

    for step in range(args.steps):
        model.train()
        try:
            x = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x = next(loader_iter)
        x = x.to(device, non_blocking=True)               # (B, 1, 16000)

        temp = cosine_temp(step, args.steps)
        with autocast(dtype=torch.bfloat16):
            recon, vq_loss, perplexity, _ = model(x, temp)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + vq_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            entry = dict(step=step,
                         loss=round(loss.item(), 5),
                         recon=round(recon_loss.item(), 5),
                         vq=round(vq_loss.item(), 5),
                         ppl=round(perplexity.item(), 1),
                         temp=round(temp, 3),
                         elapsed=round(elapsed, 1))
            log.append(entry)
            print(f"[{step:4d}/{args.steps}] loss={entry['loss']:.5f}  "
                  f"recon={entry['recon']:.5f}  vq={entry['vq']:.5f}  "
                  f"ppl={entry['ppl']:6.1f}/{args.codebook}  "
                  f"temp={entry['temp']:.3f}  {elapsed:.0f}s", flush=True)
            with open(f'{args.out}/log.json', 'w') as f:
                json.dump(log, f, indent=2)

        if step % args.save_every == 0 or step == args.steps - 1:
            torch.save({'step': step, 'model': model.state_dict(),
                        'args': vars(args)},
                       f'{args.out}/ckpt_{step:05d}.pt')
            model.eval()
            with torch.no_grad():
                # Save first 3 input vs reconstruction pairs as wavs
                n = min(3, x.shape[0])
                for i in range(n):
                    sf.write(f'{args.out}/samples/orig_{step:05d}_{i}.wav',
                             x[i, 0].float().cpu().numpy(), SAMPLE_RATE)
                    sf.write(f'{args.out}/samples/recon_{step:05d}_{i}.wav',
                             recon[i, 0].float().cpu().numpy(), SAMPLE_RATE)
            print(f"  -> ckpt + sample wavs saved (step {step})", flush=True)

    final = log[-1]
    print(f"\n{'='*60}")
    print(f"LJSpeech audio Soft-VQ-VAE training complete!")
    print(f"  Final loss:      {final['loss']:.5f}")
    print(f"  Reconstruction:  {final['recon']:.5f}")
    print(f"  VQ loss:         {final['vq']:.5f}")
    print(f"  Codebook usage:  {final['ppl']:.0f}/{args.codebook}")
    print(f"  Wall time:       {final['elapsed']:.0f}s")
    print(f"  Output dir:      {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
