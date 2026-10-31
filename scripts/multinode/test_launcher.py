"""Intercept Julia: CPU make targets must invoke exactly one parent."""
import os, pathlib, subprocess, tempfile
with tempfile.TemporaryDirectory() as tmp:
    stub = pathlib.Path(tmp)/'julia'
    log = pathlib.Path(tmp)/'calls'
    stub.write_text('#!/bin/sh\nprintf "%s|%s|%s|%s\\n" "$JULIA_MPI_TEST_NPROCS" "$FLUX_TEST_DISTRIBUTED_NCCL" "$FLUX_TEST_DISTRIBUTED_TIMEOUT" "$*" >> "$CALL_LOG"\n')
    stub.chmod(0o755)
    for target, n in [('c8-mpi',2),('c8-mpi-4',4)]:
        log.write_text('')
        subprocess.run(['make',target,f'JULIA={stub}','FLUX_REPO_PATH=../flux-multinode'],check=True,env={**os.environ,'CALL_LOG':str(log)})
        lines=log.read_text().splitlines()
        assert len(lines)==1
        assert lines[0]==f'{n}|false|900|--startup-file=no --project=../flux-multinode/test ../flux-multinode/test/ext_distributed/runtests.jl'
print('LAUNCHER_STUB PASS')
