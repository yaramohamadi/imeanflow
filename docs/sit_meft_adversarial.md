# AFM and CA-iMF on SiT-DMF MeFT checkpoints

The SiT MeFT checkpoints are under `files/SiT` (the directory `files/SiTand`
does not exist). Use the `best_fid/checkpoint_*` directory from the desired
MeFT run as `load_from`; the adversarial optimizer and discriminator states
are initialized afresh.

The adapters are `main_afm_sit_meft.py` and `main_caimf_sit_meft.py`. They
reuse the existing AFM/CA-iMF training loops, use the SiT-DMF average
velocity convention, and initialize the scalar discriminator from the SiT
generator backbone.

For example, on the Caltech MeFT checkpoint:

```bash
cd /home/ens/Zdehghani/imeanflow
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_sit_meft_adversarial.sh afm caltech101 \
  /home/ens/Zdehghani/imeanflow/files/SiT/caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w/best_fid/checkpoint_15000 \
  /home/ens/Zdehghani/imeanflow/files/logs/afm_sit_meft/caltech_afm

CUDA_VISIBLE_DEVICES=2,3 \
bash scripts/run_sit_meft_adversarial.sh caimf caltech101 \
  /home/ens/Zdehghani/imeanflow/files/SiT/caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w/best_fid/checkpoint_15000 \
  /home/ens/Zdehghani/imeanflow/files/logs/caimf_sit_meft/caltech_caimf
```

The launcher maps `caltech101`, `artbench10`, `cub200`, `food101`, and
`stanfordcars` to their latent roots and metric caches. Set `DATA_ROOT` if
the datasets are not in `/home/ens/Zdehghani/datasets`.
