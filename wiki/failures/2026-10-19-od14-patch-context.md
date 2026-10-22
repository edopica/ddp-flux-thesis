# OD-14 documentation patch context mismatch

Date: 2026-10-19.

The first `apply_patch` call failed because its expected TOML line did not match the existing text.
The patch was atomic. A subsequent read found no new decision file.
The corrected patch used stable section headers and succeeded.
No benchmark command failed, and no campaign artifacts changed.
