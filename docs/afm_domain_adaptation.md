# AFM post-training after target iMF fine-tuning

This is a separate discrete adversarial-flow post-training path. It does not
alter the original `main.py` iMF fine-tuning path or the existing CA-iMF path.

## Data flow

The Caltech loader returns cached SD-VAE posterior means and standard
deviations. The AFM trainer samples these posteriors and applies the repository's
four-channel normalization, so AFM operates on NHWC `32x32x4` normalized
latents. For each batch it samples a target latent `x0`, Gaussian `x1`, and
`0 <= r < t <= 1`, then constructs

```text
x_t      = (1-t) x0 + t x1
x_r      = (1-r) x0 + r x1
u        = iMF(x_t, r, t, y)
x_r_fake = x_t - (t-r) u
```

The implementation calls `iMeanFlow.afm_u_fn`, which maps the repository's
actual iMF-DiT signature to

```text
net.predict_u_only(x_t, t, t-r, omega, t_min, t_max, y)
```

and skips the auxiliary v branch and JVP.

## Losses

The endpoint discriminator has only this public interface:

```text
D(endpoint, endpoint_time, target_label) -> one scalar per sample
```

The losses are

```text
L_D_adv = mean(softplus(d_fake - d_real))
L_R1/R2 = mean(max(t-r, eps) * (D(x)-D(x+fd_noise))^2 / fd_epsilon^2)
L_cp    = mean((d_real+d_fake)^2)
L_D     = L_D_adv + lambda_gp*(L_R1+L_R2) + lambda_cp*L_cp

L_G_adv = mean(softplus(d_real-d_fake))
L_ot    = mean(||x_r_fake-x_t||^2 / (data_dimension*max(t-r, eps)))
L_G     = lambda_adv*L_G_adv + lambda_ot*L_ot
          + lambda_imf*L_iMF + lambda_anchor(step)*L_anchor
```

With `lambda_imf=0`, `L_iMF` and its JVP are not called or traced.

## Principal experiment

Wait for the independent Caltech iMF fine-tuning job to finish, then point the
launcher to its selected `best_fid` checkpoint:

```bash
cd /home/ens/Zdehghani/imeanflow
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TARGET_IMF_CHECKPOINT=/absolute/path/to/caltech_imf_run/best_fid \
AFM_ABLATION=B \
bash scripts/train_caltech_afm_posttrain.sh afm_only
```

## Ablations

```bash
# A: target iMF objective only
TARGET_IMF_CHECKPOINT=/path/to/best_fid AFM_ABLATION=A \
  bash scripts/train_caltech_afm_posttrain.sh target_imf

# B: principal endpoint AFM adversarial-only objective
TARGET_IMF_CHECKPOINT=/path/to/best_fid AFM_ABLATION=B \
  bash scripts/train_caltech_afm_posttrain.sh afm_only

# C: iMF + endpoint AFM
TARGET_IMF_CHECKPOINT=/path/to/best_fid AFM_ABLATION=C \
  bash scripts/train_caltech_afm_posttrain.sh imf_afm

# D: endpoint AFM plus a linearly decaying checkpoint anchor
TARGET_IMF_CHECKPOINT=/path/to/best_fid AFM_ABLATION=D LAMBDA_ANCHOR=0.1 \
  bash scripts/train_caltech_afm_posttrain.sh afm_anchor
```

Any AFM config entry can be overridden after the run label, for example:

```bash
TARGET_IMF_CHECKPOINT=/path/to/best_fid AFM_ABLATION=B \
bash scripts/train_caltech_afm_posttrain.sh afm_gp_ablation \
  --config.afm.lambda_gp=0.1 \
  --config.afm.fd_epsilon=0.01 \
  --config.afm.gp_batch_fraction=0.25 \
  --config.afm.discriminator_trainable_blocks=8 \
  --config.afm.use_discriminator_augmentation=True \
  --config.afm.discriminator_augmentation_probability=0.2
```

## Checkpoints

Full checkpoints contain generator, EMA generator, discriminator, both Optax
optimizer/schedule states, batch/epoch/G/D/image counters, optional anchor
parameters, and the serialized JAX RNG key. `afm_metadata.json` beside every
checkpoint records the target class mapping, AFM configuration, source
checkpoint, data root, and latent representation. `generator_only/` is directly
loadable by the existing inference/evaluation pipeline.

Resume without resetting warm-up or counters with:

```bash
TARGET_IMF_CHECKPOINT=/path/to/original/best_fid AFM_ABLATION=B \
bash scripts/train_caltech_afm_posttrain.sh resumed \
  --config.afm.resume_from=/path/to/afm/run
```

