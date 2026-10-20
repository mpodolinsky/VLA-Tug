# data

Local data, ignored by git. One folder per dataset (`libero10_groot`, `libero10_pi05`, `liberox_groot`):

```
<dataset>/
  manifest.csv   one row per rollout, as published with the Hub dataset named in fp/config.py
  splits/        LIBERO-X only: the held-out task lists (leave_task_out, leave_signature_out, leave_skill_out/<skill>/test.txt)
```

`manifest.csv` and `splits/` are copied unchanged from the Hub datasets.
