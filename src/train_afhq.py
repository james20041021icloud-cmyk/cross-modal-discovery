"""
Soft-VQ-VAE on AFHQ (Animal Faces HQ) — Quest A100 GPU.

Trains on ~14.6k animal face images at 256×256.
Same model as the local M1 run, just rebatched for the A100.
"""
import os, math, json, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torchvision.utils import save_image

from vqvae_model import SoftVQVAE


def cosine_temp(step, total, t_start=1.0, t_end=0.1):
    p = step / max(total - 1, 1)
    return t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      default='data/afhq/train')
    ap.add_argument('--out',       default='runs/afhq')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--batch',     type=int, default=32)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--steps',     type=int, default=5000)
    ap.add_argument('--hidden',    type=int, default=128)
    ap.add_argument('--latent',    type=int, default=64)
    ap.add_argument('--codebook',  type=int, default=1024)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--save_every',type=int, default=500)
    ap.add_argument('--workers',   type=int, default=4)
    args = ap.parse_args()

    os.makedirs(f'{args.out}/samples', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")
    print(f"Args: {vars(args)}\n")

    # ── Data ───────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(args.data, transform=tf)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes "
          f"({dataset.classes})")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = SoftVQVAE(
        in_channels=3, hidden=args.hidden, latent_dim=args.latent,
        num_embeddings=args.codebook,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scaler = GradScaler()                              # mixed-precision

    log, t0, loader_iter = [], time.time(), iter(loader)
    print(f"Training for {args.steps} steps (bf16 mixed-precision)...\n")

    for step in range(args.steps):
        model.train()
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, _ = next(loader_iter)
        x = x.to(device, non_blocking=True)

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
            print(f"[{step:4d}/{args.steps}] loss={entry['loss']:.4f}  "
                  f"recon={entry['recon']:.4f}  vq={entry['vq']:.4f}  "
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
                n = min(8, x.shape[0])
                grid = torch.cat([x[:n], recon[:n].float()])
                save_image(grid * 0.5 + 0.5,
                           f'{args.out}/samples/recon_{step:05d}.png', nrow=n)
            print(f"  -> ckpt + samples saved (step {step})", flush=True)

    final = log[-1]
    print(f"\n{'='*60}")
    print(f"AFHQ training complete!")
    print(f"  Final loss:      {final['loss']:.4f}")
    print(f"  Reconstruction:  {final['recon']:.4f}")
    print(f"  VQ loss:         {final['vq']:.4f}")
    print(f"  Codebook usage:  {final['ppl']:.0f}/{args.codebook}")
    print(f"  Wall time:       {final['elapsed']:.0f}s")
    print(f"  Output dir:      {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
