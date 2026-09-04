JULIA ?= julia
FLUX_REPO_PATH ?= ../Flux.jl
MPIEXECJL ?= $(HOME)/.julia/bin/mpiexecjl

# Default to PMI2 for SLURM environments because Julia's MPICH_jll uses PMI2
SLURM_MPI_TYPE ?= pmi2

.PHONY: env precompile precompile-flux-test precompile-all env-gpu install-mpiexec check smoke-cpu audit status reference reference-verify health-check launch-2 launch-4 sync-model verify-sync verify-data verify-gradients verify-conditional example-ddp c8-mpi c8-mpi-4 c8-nccl c8-all

# NOTE on precompilation (Julia 1.12, HPC): never let julia precompile the
# project envs on the HPC login node. Julia caches bake the host CPU feature
# set: login (graniterapids) compiles are unusable on compute nodes
# (icelake-server), and gnode caches cannot load on login either. Precompile
# must run on a compute node: `bash scripts/remote/precompile.sh hpc`.
# `env` only resolves dependencies and never precompiles (auto-precompile off).
env:
	FLUX_REPO_PATH="$(FLUX_REPO_PATH)" JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA) --project=. -e 'using Pkg; Pkg.develop(path=ENV["FLUX_REPO_PATH"]); Pkg.add(["MPI","Optimisers","Zygote","Functors","MLUtils","Adapt","BenchmarkTools","Revise","JLD2"]); Pkg.instantiate()'

precompile:
	@if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 1 -c 8 $(JULIA) --startup-file=no --project=. -e 'using Pkg; Pkg.precompile()'; \
	elif [ "$$(hostname)" = "slnode01" ]; then echo "ERROR: on the HPC login node. Precompile would target the wrong CPU. Use: bash scripts/remote/precompile.sh hpc"; exit 1; \
	else $(JULIA) --startup-file=no --project=. -e 'using Pkg; Pkg.precompile()'; fi

precompile-flux-test:
	@if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 1 -c 8 $(JULIA) --startup-file=no --project=$(FLUX_REPO_PATH)/test -e 'using Pkg; Pkg.precompile()'; \
	elif [ "$$(hostname)" = "slnode01" ]; then echo "ERROR: on the HPC login node. Precompile would target the wrong CPU. Use: bash scripts/remote/precompile.sh hpc"; exit 1; \
	else $(JULIA) --startup-file=no --project=$(FLUX_REPO_PATH)/test -e 'using Pkg; Pkg.precompile()'; fi

precompile-all: precompile precompile-flux-test

env-gpu:
	$(JULIA) --project=. scripts/setup/setup_gpu.jl

install-mpiexec:
	$(JULIA) --project=. -e 'using MPI; MPI.install_mpiexecjl(force=true)'

check:
	$(JULIA) --project=. scripts/setup/check_env.jl

smoke-cpu:
	$(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/checks/smoke_mpi.jl

audit:
	bash scripts/setup/audit_distributed.sh "$(FLUX_REPO_PATH)"

status:
	git status --short
	git -C "$(FLUX_REPO_PATH)" status --short
	git -C "$(FLUX_REPO_PATH)" branch --show-current
	git -C "$(FLUX_REPO_PATH)" rev-parse HEAD

reference:
	@mkdir -p artifacts/baselines
	$(JULIA) --project=. scripts/reference/save.jl

reference-verify:
	$(JULIA) --project=. scripts/reference/verify.jl

launch-2:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/launch/skeleton.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/launch/skeleton.jl; fi

launch-4:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 4 $(JULIA) --project=. scripts/launch/skeleton.jl; else $(MPIEXECJL) --project=. -n 4 $(JULIA) scripts/launch/skeleton.jl; fi

health-check:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/checks/health_check.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/checks/health_check.jl; fi

sync-model:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/sync/broadcast.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/sync/broadcast.jl; fi

verify-sync:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/sync/verify.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/sync/verify.jl; fi

verify-data:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 4 $(JULIA) --project=. scripts/data/verify_sharding.jl; else $(MPIEXECJL) --project=. -n 4 $(JULIA) scripts/data/verify_sharding.jl; fi
verify-gradients:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 4 $(JULIA) --project=. scripts/sync/verify_gradients.jl; else $(MPIEXECJL) --project=. -n 4 $(JULIA) scripts/sync/verify_gradients.jl; fi

verify-conditional:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 3 $(JULIA) --project=. scripts/sync/verify_conditional.jl; else $(MPIEXECJL) --project=. -n 3 $(JULIA) scripts/sync/verify_conditional.jl; fi

example-ddp:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/examples/train_ddp.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/examples/train_ddp.jl; fi

c8-mpi:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl mpi; else $(MPIEXECJL) --project=$(FLUX_REPO_PATH)/test -n 2 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl mpi; fi

c8-mpi-4:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 4 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl mpi; else $(MPIEXECJL) --project=$(FLUX_REPO_PATH)/test -n 4 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl mpi; fi

c8-nccl:
	if [ -n "$$SLURM_JOB_ID" ]; then FLUX_TEST_DISTRIBUTED_NCCL=true srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl nccl; else FLUX_TEST_DISTRIBUTED_NCCL=true $(MPIEXECJL) --project=$(FLUX_REPO_PATH)/test -n 2 $(JULIA) --project=$(FLUX_REPO_PATH)/test $(FLUX_REPO_PATH)/test/ext_distributed/runtests.jl nccl; fi

c8-all: c8-mpi c8-mpi-4
	@echo "Running c8-mpi and c8-mpi-4 completed."
	@if command -v nvidia-smi > /dev/null 2>&1; then \
		$(MAKE) c8-nccl; \
	else \
		echo "SKIP: No GPU (nvidia-smi not found). NCCL suite not run."; \
	fi

