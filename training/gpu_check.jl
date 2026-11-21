# Can training use the GPU? Prints versions and PASS/FAIL for the three kinds of GPU work the trainer does.
#   docker compose -f docker/docker-compose.yml run --rm train julia training/gpu_check.jl
using CUDA, Flux
using Flux.Optimisers

println("Julia ", VERSION, ", CUDA.jl ", pkgversion(CUDA), ", Flux ", pkgversion(Flux))
try
  CUDA.versioninfo()
catch e
  println("versioninfo failed: ", first(sprint(showerror, e), 300))
end
println("CUDA.functional(): ", CUDA.functional())

function check(name, f)
  try
    println("PASS  ", name, "  (", f(), ")")
    return true
  catch e
    println("FAIL  ", name, "  ", first(sprint(showerror, e), 400))
    return false
  end
end

ok = CUDA.functional()
ok &= check("cuBLAS matrix multiply", () -> sum(CUDA.rand(Float32, 256, 256) * CUDA.rand(Float32, 256, 256)))
ok &= check("Julia-compiled GPU kernel (broadcast)", () -> sum(sin.(CUDA.rand(Float32, 10_000)) .+ 1f0))
ok &= check("Flux training step (explicit API, like the trainer)", () -> begin
  m = Chain(Dense(18, 7, sigmoid), Dense(7, 13, sigmoid), Dense(13, 3), Dense(3, 1)) |> gpu
  x, y = CUDA.rand(Float32, 18, 1000), CUDA.rand(Float32, 1, 1000)
  st = Optimisers.setup(Optimisers.AdaGrad(), m)
  gs = Flux.gradient(mm -> Flux.mse(mm(x), y), m)
  st, m = Optimisers.update!(st, m, gs[1])
  Flux.mse(m(x), y)
end)
println(ok ? "\nRESULT: GPU OK" : "\nRESULT: GPU NOT usable - training will fall back to CPU")
