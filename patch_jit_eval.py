#!/usr/bin/env python3
"""Add eval_only support to train_imf_jit.py (JiT DMF MeFT).

Mirrors train.py:just_evaluate but for the pixel-space JiT path:
- lightweight EvalState restore (no optimizer), use_ema=False for these runs
- loops requested NFE (force_metric_num_steps / metric_num_steps), scalar CFG
- calls image_metric_evaluator with ema_only=use_ema (NOT `not use_ema`; these
  runs have no ema_params, so ema_only must be False)
- writes the same eval_only CSV schema the DiT/SiT/plain-JiT rows use
Also routes main_imf_jit.py to just_evaluate when config.eval_only.
Idempotent: bails if the marker is already present.
"""
import io, sys, os

REPO = "/opt/dlami/nvme/meanflow/imeanflow"
TRAIN = os.path.join(REPO, "train_imf_jit.py")
MAIN = os.path.join(REPO, "main_imf_jit.py")
MARKER = "def just_evaluate("

src = io.open(TRAIN, encoding="utf-8").read()
if MARKER in src:
    print("train_imf_jit.py: already patched (just_evaluate present); skipping.")
    sys.exit(0)

# --- backup ---
io.open(TRAIN + ".bak_jiteval", "w", encoding="utf-8").write(src)
io.open(MAIN + ".bak_jiteval", "w", encoding="utf-8").write(io.open(MAIN, encoding="utf-8").read())

# 1) import restore_eval_checkpoint alongside the existing ckpt_util imports
old_imp = (
    "from utils.ckpt_util import (\n"
    "    restore_checkpoint,\n"
    "    restore_partial_checkpoint,\n"
    "    save_best_checkpoint,\n"
    "    save_checkpoint,\n"
    ")"
)
new_imp = (
    "from utils.ckpt_util import (\n"
    "    restore_checkpoint,\n"
    "    restore_eval_checkpoint,\n"
    "    restore_partial_checkpoint,\n"
    "    save_best_checkpoint,\n"
    "    save_checkpoint,\n"
    ")"
)
assert old_imp in src, "ckpt_util import block not found as expected"
src = src.replace(old_imp, new_imp, 1)

# 2) replace the eval_only guard so eval routing is explicit in the driver too
old_guard = (
    "    writer = Writer(config, workdir)\n"
    "    if config.eval_only:\n"
    "        raise ValueError(\"eval_only is not supported by train_imf_jit.\")\n"
)
new_guard = (
    "    writer = Writer(config, workdir)\n"
    "    if config.eval_only:\n"
    "        # eval_only is handled by just_evaluate(); main_imf_jit routes there.\n"
    "        return just_evaluate(config, workdir)\n"
)
assert old_guard in src, "eval_only guard not found as expected"
src = src.replace(old_guard, new_guard, 1)

# 3) append helpers + just_evaluate at end of file
EVAL_CODE = '''

########################################################
#                 Evaluation (eval_only)               #
########################################################


def _get_eval_sampling_configs(config):
    """Scalar CFG only (JiT DMF is not guidance-controllable): (omega, t_min, t_max)."""
    sampling = config.sampling
    omega = sampling.get("omega", None)
    t_min = sampling.get("t_min", None)
    t_max = sampling.get("t_max", None)
    if omega is not None and t_min is not None and t_max is not None:
        return [(float(omega), float(t_min), float(t_max))]
    raise ValueError(
        "eval_only requires sampling.omega + sampling.t_min + sampling.t_max in the config."
    )


def _get_metric_num_steps(config):
    """NFE list to evaluate: force_metric_num_steps overrides; else metric_num_steps;
    else the training sampling.num_steps. Primary step is always included first."""
    forced = str(config.training.get("force_metric_num_steps", "") or "").strip()
    if forced:
        steps = [int(s) for s in forced.replace(",", " ").split()]
    else:
        configured = config.training.get("metric_num_steps", ())
        steps = [int(s) for s in configured] if configured else [int(config.sampling.num_steps)]
    primary = int(config.sampling.num_steps)
    ordered = []
    for s in [primary] + steps:
        if s < 1:
            raise ValueError("Metric sampling steps must be >= 1.")
        if s not in ordered:
            ordered.append(s)
    return tuple(ordered)


def _primary_metric_mode(use_ema):
    return "ema" if use_ema else "online"


def just_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Post-hoc multi-NFE FID/FDD/IS eval for a saved JiT DMF checkpoint.

    Mirrors train.py:just_evaluate on the pixel-space JiT path. Restores a
    lightweight EvalState from config.load_from (a best_fid/checkpoint_* dir),
    then evaluates every NFE in _get_metric_num_steps and writes eval_only rows.
    """
    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert config.load_from != "", "config.load_from must be specified for just_evaluate"

    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)

    image_size = int(config.dataset.image_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    # These runs train with use_ema=False -> evaluate online params.
    use_ema = config.training.get("use_ema", True)
    metric_mode = _primary_metric_mode(use_ema)

    model = _build_model(config, eval_mode=True)

    state = restore_eval_checkpoint(config.load_from, use_ema=use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    pixel_manager = PixelImageManager(
        sample_device_bsz,
        decode_num_local_devices=sample_local_device_count,
    )

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(99),
                device_batch_size=sample_device_bsz,
                config=config,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    image_metric_evaluator = get_image_metric_evaluator(config, writer, pixel_manager)
    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        n: build_p_sample_step(n) for n in metric_num_steps
    }

    best_fid = float("inf")
    best_config = None
    best_fd_dino = float("inf")
    best_fd_dino_config = None
    csv_rows = []
    for num_steps, p_sample_step in p_metric_sample_steps.items():
        for omega, t_min, t_max in _get_eval_sampling_configs(config):
            kwargs = jax_utils.replicate(
                {"omega": omega, "t_min": t_min, "t_max": t_max},
                devices=sample_devices,
            )
            result = image_metric_evaluator(
                state,
                p_sample_step,
                step,
                ema_only=use_ema,
                metric_suffix=f"steps_{num_steps}",
                **kwargs,
            )
            fid = float(result["fid"])
            is_score = float(result["is"])
            fd_dino = result.get("fd_dino", None)
            row = dict(sampling_num_steps=num_steps, omega=omega, t_min=t_min,
                       t_max=t_max, fid=fid, is_score=is_score, fd_dino=fd_dino)
            csv_rows.append(row)
            cfg_key = (num_steps, omega, t_min, t_max)
            if fid < best_fid:
                best_fid = fid
                best_config = cfg_key
            if fd_dino is not None and fd_dino < best_fd_dino:
                best_fd_dino = fd_dino
                best_fd_dino_config = cfg_key
            log_for_0("eval_only NFE=%d omega=%.2f -> FID=%.4f IS=%.4f FDD=%s",
                      num_steps, omega, fid, is_score,
                      "None" if fd_dino is None else f"{float(fd_dino):.4f}")

    for row in csv_rows:
        cfg_key = (row["sampling_num_steps"], row["omega"], row["t_min"], row["t_max"])
        _write_eval_metrics_csv(
            workdir,
            eval_phase="eval_only",
            metric_mode=metric_mode,
            training_step=step,
            sampling_num_steps=row["sampling_num_steps"],
            omega=row["omega"],
            t_min=row["t_min"],
            t_max=row["t_max"],
            fid=float(row["fid"]),
            inception_score=float(row["is_score"]),
            fd_dino="" if row["fd_dino"] is None else float(row["fd_dino"]),
            is_best_fid=int(cfg_key == best_config),
            is_best_fd_dino=int(best_fd_dino_config is not None and cfg_key == best_fd_dino_config),
            checkpoint_path=os.path.abspath(config.load_from),
        )

    log_for_0("eval_only DONE. best FID=%.4f at %s", best_fid, str(best_config))
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
'''

# CSV writer helper (_write_eval_metrics_csv) already exists in train_imf_jit.py.
src = src.rstrip("\n") + "\n" + EVAL_CODE
io.open(TRAIN, "w", encoding="utf-8").write(src)
print("train_imf_jit.py: patched (import + guard + just_evaluate).")

# 4) main_imf_jit.py routing (train_and_evaluate now returns just_evaluate result,
#    so no change strictly needed, but make it explicit for clarity/safety).
msrc = io.open(MAIN, encoding="utf-8").read()
old_call = "        train_imf_jit.train_and_evaluate(FLAGS.config, FLAGS.workdir)"
new_call = (
    "        if FLAGS.config.get(\"eval_only\", False):\n"
    "            train_imf_jit.just_evaluate(FLAGS.config, FLAGS.workdir)\n"
    "        else:\n"
    "            train_imf_jit.train_and_evaluate(FLAGS.config, FLAGS.workdir)"
)
if new_call in msrc:
    print("main_imf_jit.py: already routes eval_only; skipping.")
elif old_call in msrc:
    msrc = msrc.replace(old_call, new_call, 1)
    io.open(MAIN, "w", encoding="utf-8").write(msrc)
    print("main_imf_jit.py: patched (explicit eval_only routing).")
else:
    print("WARNING: main_imf_jit.py train_and_evaluate call not found verbatim; "
          "guard-return in train_imf_jit still handles eval_only.")
print("DONE.")
