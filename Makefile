JULIA ?= julia
FLUX_REPO_PATH ?= ../Flux.jl
MPIEXECJL ?= mpiexecjl

.PHONY: env env-gpu install-mpiexec check smoke-cpu audit status reference reference-verify

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
