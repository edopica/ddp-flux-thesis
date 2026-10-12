testdir = "/home/kurapica/Projects/ddp_flux/ddp_flux/test/ext_distributed"
files = readdir(testdir)

for f in files
    if endswith(f, "_distributedtest.jl")
        path = joinpath(testdir, f)
        content = read(path, String)
        
        # We need to hoist "import MPI" etc to top level.
        # So we can just put "import MPI" at the top regardless of backend,
        # since Pkg.test installs MPI anyway for these tests.
        
        # Replace the `const` declarations which aren't allowed in functions.
        content = replace(content, "const backend_type" => "backend_type")
        content = replace(content, "const dev" => "dev")
        content = replace(content, "const aType" => "aType")
        
        # Also remove `import MPI` and `import MPI, NCCL, CUDA` from inside if blocks,
        # and instead prepend them to the file along with `include("helper.jl")` and wrapping.
        content = replace(content, "import MPI, NCCL, CUDA" => "")
        content = replace(content, "import MPI" => "")
        
        lines = split(content, "\n")
        using_lines = String[]
        rest_lines = String[]
        
        in_usings = true
        for line in lines
            if in_usings && (startswith(line, "using ") || startswith(line, "import "))
                push!(using_lines, line)
            elseif in_usings && strip(line) == ""
                # skip empty lines at top
            else
                in_usings = false
                push!(rest_lines, line)
            end
        end
        
        # Build new file
        header = join(using_lines, "\n") * "\n\n" *
                 "import MPI\n" *
                 "using Pkg\n" *
                 "if get(ENV, \"FLUX_TEST_DISTRIBUTED_NCCL\", \"false\") == \"true\"\n" *
                 "    import NCCL, CUDA\n" *
                 "end\n" *
                 "include(\"helper.jl\")\n\n" *
                 "run_with_enforced_exit() do\n"
                 
        body = join(map(l -> "    " * l, rest_lines), "\n")
        
        new_content = header * body * "\nend\n"
        
        write(path, new_content)
        println("Transformed ", f)
    end
end
