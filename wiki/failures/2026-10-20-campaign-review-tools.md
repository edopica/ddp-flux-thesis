# Campaign review tool failures — 2026-10-20

The initial remote checksum wrapper used `subprocess.run(..., text=True, capture_output=True)` through the login-node `python3`.
That Python is version 3.6. It rejected `text` with `TypeError` before the checksum command ran.
The replacement used `universal_newlines=True`, `stdout=subprocess.PIPE`, and `stderr=subprocess.PIPE`.
The remote checksum command then returned 0 for all 307 entries.

A source read used the nonexistent path `source/scripts/static_scaling/full_dataset_timing.jl` in the downloaded evidence.
The source snapshot uses the flat path `source/full_dataset_timing.jl`.
The corrected read succeeded.

Neither failure changed the campaign artifacts or ran training.
Separately, scheduler history records failed step `61493236.9` inside the completed campaign allocation.
Its cause remains unresolved. The claim review records this limitation.
