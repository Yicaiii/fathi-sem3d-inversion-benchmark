# Reusable Mini Iteration Handoff

The mini workflow is iteration-generic. The Python modules are reused for
every transition; iteration numbers and paths are supplied by JSON.

## Validate accepted iter_001 as the next parent

Preparation only:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --config configs/fathi_mini_e2e_iter001_to_iter002.json \
  --stage next-forward-prepare
```

Heavy strict-DUDX forward:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --config configs/fathi_mini_e2e_iter001_to_iter002.json \
  --stage next-forward \
  --np 12
```

Read-only status:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --config configs/fathi_mini_e2e_iter001_to_iter002.json \
  --stage next-forward-status
```

Expected result:

```text
RESULT = PASS_MINI_ITER001_TO_ITER002_FORWARD_HANDOFF
```

This proves that the accepted iter_001 material model can directly seed the
next 3,600-control-point DUDX forward. No iteration-specific Python copy is
required.

## Presentation sentence

> The iteration number is not hard-coded in the algorithm. The accepted output
> of one transition becomes the configured parent input of the next transition.
> The same Python modules are reused; only the JSON iteration context changes.
