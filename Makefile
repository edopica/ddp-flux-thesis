JULIA ?= julia
FLUX_REPO_PATH ?= ../Flux.jl
MPIEXECJL ?= $(HOME)/.julia/bin/mpiexecjl

# Default to PMI2 for SLURM environments because Julia's MPICH_jll uses PMI2
SLURM_MPI_TYPE ?= pmi2

.PHONY: env precompile env-gpu install-mpiexec check smoke-cpu audit status reference reference-verify health-check launch-2 launch-4 sync-model verify-sync verify-data verify-gradients verify-conditional example-ddp

env:
	FLUX_REPO_PATH="$(FLUX_REPO_PATH)" $(JULIA) --project=. -e 'using Pkg; Pkg.develop(path=ENV["FLUX_REPO_PATH"]); Pkg.add(["MPI","Optimisers","Zygote","Functors","MLUtils","Adapt","BenchmarkTools","Revise","JLD2"]); Pkg.instantiate(); Pkg.precompile()'

precompile:
	$(JULIA) --project=. -t auto -e 'using Pkg; Pkg.precompile()'

env-gpu:
	$(JULIA) --project=. scripts/setup/setup_gpu.jl

install-mpiexec:
	$(JULIA) --project=. -e 'using MPI; MPI.install_mpiexecjl()'

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

