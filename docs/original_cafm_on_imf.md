# Original CAFM on a fine-tuned improved MeanFlow checkpoint

This pipeline is separate from both existing adversarial implementations:

- `caimf.py` uses finite intervals and endpoint potentials.
- `afm.py` uses discrete generated endpoints.
- `cafm_imf.py` uses the original CAFM infinitesimal JVP objective.

For a real latent `x_0`, noise `z`, and uniformly sampled time `t`, it forms

```text
x_t = (1 - t) x_0 + t z
v_real = z - x_0
v_fake = v_theta(x_t, t)
```

The fine-tuned iMF checkpoint supplies `v_fake` through its trained auxiliary
instantaneous-velocity head. The average-velocity `u(x_t,r,t)` is deliberately
not used in the CAFM loss.

The discriminator is a generator-initialized iMF-DiT potential
`D(x_t,t,y)`. Its unused guidance/interval conditioning inputs are fixed to
constants. Real and fake logits are directional derivatives:

```text
logit_real = JVP[D](x_t, t; v_real, 1)
logit_fake = JVP[D](x_t, t; v_fake, 1)
```

The objectives match the official CAFM post-training setup:

```text
L_D = (logit_real - 1)^2 + (logit_fake + 1)^2
      + lambda_cp D(x_t,t)^2
L_G = lambda_adv (logit_fake - 1)^2
      + lambda_ot mean(v_fake^2)
```

There is no iMF regression term in original CAFM, so `lambda_imf` must remain
zero. The supplied config uses the paper's post-training defaults:

- generator/discriminator learning rate: `1e-5`
- Adam betas: `(0.0, 0.95)`
- weight decay: `0`
- centering weight: `0.001`
- OT weight: `0`
- discriminator warm-up: `10,000` batches
- update schedule after warm-up: `16D:1G`
- EMA decay: `0.99`

Metric sampling also uses the instantaneous velocity with Euler integration.
This is important: using the ordinary iMF sampler would instead evaluate its
finite-interval average-velocity head and would not be a clean CAFM result.
The sampling model retains the full auxiliary-v refinement branch.

Example:

```bash
cd /home/ens/Zdehghani/imeanflow
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
CALTECH_IMF_CHECKPOINT=/path/to/caltech/imf/checkpoint \
USE_WANDB=False \
bash scripts/train_caltech_finetuned_cafm_imf.sh run1
```

Additional configuration overrides can be placed after `run1`.
