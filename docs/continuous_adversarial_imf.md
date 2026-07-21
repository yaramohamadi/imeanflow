# Continuous adversarial improved MeanFlow

This port post-trains a plain `imfDiT_*` improved MeanFlow checkpoint with the
finite-interval adversarial objective from the CA-iMF draft.

## Implemented objective

For a strict interval `r < t`, the training path is

```text
x_t       = (1 - t) x + t epsilon
x_r       = (1 - r) x + r epsilon
xhat_r    = x_t - (t - r) u_theta(x_t, r, t)
a_real    = (D(x_t, t) - D(x_r, r)) / (t - r)
a_fake    = (D(x_t, t) - D(xhat_r, r)) / (t - r)
```

The discriminator uses least-squares targets `a_real=1`, `a_fake=-1` plus the
potential-centering penalty. The generator uses

```text
lambda_imf * L_iMF + lambda_adv * (a_fake - 1)^2
                       + lambda_ot * mean(u_theta^2).
```

The discriminator has the generator's patch embedder, condition embedders,
shared transformer, and u-branch transformer. Every matching tensor is copied
from the loaded generator. Only its RMS normalization and scalar output head
are new. Matching CAFM, the initialized discriminator backbone is trainable by
default; set `caimf.freeze_discriminator_backbone=True` for a head-only
ablation.

The first 10,000 batches update only the discriminator. Training then repeats
16 discriminator batches followed by one generator batch. The MF-only baseline
skips discriminator creation and updates the generator every batch.

## Required data and checkpoint

The configured Caltech latent layout is:

```text
/home/ens/Zdehghani/datasets/caltech-101_processed_latents/train/*.pt
```

`IMF_CHECKPOINT` must be an iMF checkpoint, not the original DiT/JiT/SiT
checkpoint. Both requested entry modes use the same launcher:

- `ENTRY_MODE=target_ft`: point `IMF_CHECKPOINT` at the iMF checkpoint already
  fine-tuned on Caltech, then apply CA-iMF post-training.
- `ENTRY_MODE=imagenet_joint`: point `IMF_CHECKPOINT` at the ImageNet iMF
  checkpoint. Shape-compatible weights are loaded, target class parameters are
  initialized for Caltech, and target adaptation happens jointly with CA-iMF.

Loading is parameter-only: both optimizers and all step counters are reset. By
default the online generator parameters are loaded because checkpoints produced
with `use_ema=False` can contain a stale EMA tree. Set
`--config.caimf.load_generator_ema=True` only for a checkpoint whose EMA was
actually maintained.

## Launch the four experiments

Use both Taylor A100s by leaving `CUDA_VISIBLE_DEVICES=0,1` (the launcher
default). For example:

```bash
cd /home/ens/Zdehghani/imeanflow

IMF_CHECKPOINT=/absolute/path/to/caltech_imf_checkpoint \
ENTRY_MODE=target_ft EXPERIMENT=1 \
bash scripts/train_caltech_caimf.sh
```

`EXPERIMENT` selects:

1. `lambda_imf=0.01`, `lambda_adv=1`
2. `lambda_imf=1`, `lambda_adv=1`
3. `lambda_imf=1`, `lambda_adv=0.01`
4. `lambda_imf=1`, `lambda_adv=0`, no discriminator updates

For the joint target-adaptation run:

```bash
IMF_CHECKPOINT=/absolute/path/to/imagenet_imf_checkpoint \
ENTRY_MODE=imagenet_joint EXPERIMENT=2 \
bash scripts/train_caltech_caimf.sh
```

Extra config overrides can be appended to either command, for example:

```bash
bash scripts/train_caltech_caimf.sh \
  --config.caimf.lambda_ot=0.001 \
  --config.caimf.max_posttrain_batches=50000
```

Saved checkpoints retain top-level `params` and `ema_params`, so the existing
`main.py` evaluation/sampling path can load them. The CA trainer itself focuses
on post-training and checkpointing; run the existing evaluation command on each
saved work directory to compare FID/FD-DINO at the desired NFE.

## Dedicated Caltech-finetuned checkpoint workflow

For the sequential experiment--first ordinary iMF fine-tuning on Caltech, then
CA-iMF post-training--use the dedicated launcher. It requires an explicit target
checkpoint and cannot fall back to the ImageNet iMF checkpoint:

```bash
cd /home/ens/Zdehghani/imeanflow
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CALTECH_IMF_CHECKPOINT=/absolute/path/to/caltech_imf_run/best_fid \
CAIMF_EXPERIMENT=1 \
bash scripts/train_caltech_finetuned_caimf.sh target_exp1
```

The source checkpoint initializes G. The optimizer and post-training counters
are reset. D is a distinct network initialized by copying the target-finetuned
G backbone; its new scalar head is initialized separately, and both the D
backbone and head are trainable. The default target schedule is 5,000 D-only
batches followed by 4D:1G through batch 155,000, giving 30,000 G updates.

Experiments 1--4 reproduce the CA-iMF loss-weight ablations. Evaluation reports
FID, IS, and FD-DINO for 1-NFE and 4-NFE sampling, and `best_fid` is selected by
the configured 4-NFE sampler.

Set `CAIMF_EXPERIMENT=5` for adversarial-only CA-iMF with
`lambda_imf=0` and `lambda_adv=1`.
