# Command Failure: SSH to deathstar

**Date**: 2026-07-09

**Command**:
```bash
./scripts/sync_code.sh deathstar
```

**Output**:
```text
Syncing to deathstar:~/projects...
ssh_askpass: exec(/usr/lib/ssh/ssh-askpass): No such file or directory
Permission denied, please try again.
ssh_askpass: exec(/usr/lib/ssh/ssh-askpass): No such file or directory
Permission denied, please try again.
ssh_askpass: exec(/usr/lib/ssh/ssh-askpass): No such file or directory
picazio@deathstar-hpc.sm.unibocconi.it: Permission denied (publickey,password).
```

**Context**:
Attempted to synchronize the project codebase to the `deathstar` remote node to configure its environment, but failed because the node requires a password or a registered public key that is currently not available in this environment.

**Resolution/Next Step**:
User must provide the necessary SSH credentials (either by running `ssh-copy-id` or providing a password) before the environment can be fully configured.
