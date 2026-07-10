JULIA ?= julia
FLUX_REPO_PATH ?= ../Flux.jl
MPIEXECJL ?= $(HOME)/.julia/bin/mpiexecjl

# Default to PMI2 for SLURM environments because Julia's MPICH_jll uses PMI2
SLURM_MPI_TYPE ?= pmi2

.PHONY: env env-gpu install-mpiexec check smoke-cpu audit status reference reference-verify health-check launch-2 launch-4

env:
	FLUX_REPO_PATH="$(FLUX_REPO_PATH)" $(JULIA) --project=. -e 'using Pkg; Pkg.develop(path=ENV["FLUX_REPO_PATH"]); Pkg.add(["MPI","Optimisers","Zygote","Functors","MLUtils","Adapt","BenchmarkTools","Revise"]); Pkg.instantiate(); Pkg.precompile()'

env-gpu:
	$(JULIA) --project=. scripts/setup_env_gpu.jl

install-mpiexec:
	$(JULIA) --project=. -e 'using MPI; MPI.install_mpiexecjl()'

check:
	$(JULIA) --project=. scripts/check_env.jl

smoke-cpu:
	$(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/smoke_mpi_cpu.jl

audit:
	bash scripts/audit_flux_distributed.sh "$(FLUX_REPO_PATH)"

status:
	git status --short
	git -C "$(FLUX_REPO_PATH)" status --short
	git -C "$(FLUX_REPO_PATH)" branch --show-current
	git -C "$(FLUX_REPO_PATH)" rev-parse HEAD

reference:
	@mkdir -p artifacts/baselines
	$(JULIA) --project=. scripts/save_reference.jl

reference-verify:
	$(JULIA) --project=. scripts/verify_reference.jl

launch-2:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/launch_skeleton.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/launch_skeleton.jl; fi

launch-4:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 4 $(JULIA) --project=. scripts/launch_skeleton.jl; else $(MPIEXECJL) --project=. -n 4 $(JULIA) scripts/launch_skeleton.jl; fi

health-check:
	if [ -n "$$SLURM_JOB_ID" ]; then srun --mpi=$(SLURM_MPI_TYPE) -n 2 $(JULIA) --project=. scripts/mpi_health_check.jl; else $(MPIEXECJL) --project=. -n 2 $(JULIA) scripts/mpi_health_check.jl; fi
