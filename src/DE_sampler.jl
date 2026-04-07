

"""
    Model Structure

Initiates a structure of class Model

# Arguments
- T
- N
- D
- νT
- bufferTime
- batchTime
- learning_rate
- Z
- U
- Basis
- States
- Λ::Function: Library of potential functions
- Λnames::Vector{String}: Names of components of library of potential functions

"""
mutable struct Model

  # model
  # S, T, N, D, νS, νT, bufferSpace, bufferTime, batchSpace, batchTime, learning_rate, Z, U, Basis, States

  # dimensions
  T::Int
  L::Int
  N::Int
  D::Int

  # data parameters
  νT
  buffer
  batch_size
  TimeStep
  learning_rate
  inner_inds

  # data
  V::Array{Float64}
  H
  X

  # basis functions
  Basis::DerivativeClass

  # function information
  Λ::Function
  ΛTimeNames::Vector{String}

  # SSVS parameters
  v0::Float64
  v1::Float64

  function_names
  response

  # elastic net regularization scale (one value per component/row of A)
  scale_el::Vector{Float64}

  # optional fixed gamma mask: N×D matrix where -1 = free, 0 = fixed to 0, 1 = fixed to 1
  gamma_mask::Union{Nothing, Matrix{Int}}

end

Base.show(io::IO, model::Model) =
  print(io, "Model\n",
    " ├─── data dimensions: ", [model.T, model.L], '\n',
    " ├─── process dimensions: ", [model.T, model.N], '\n',
    " ├─── response: ", model.response, '\n',
    " ├─── library: ", model.function_names, '\n',
    " ├─── covariates: ", model.X === nothing ? false : size(model.X, 1), '\n',
    " ├─── temporal domain: ", model.TimeStep, '\n',
    " ├─── number of temporal basis functions: ", model.νT, '\n',
    " ├─── SSVS parameters: v0 = " * string(model.v0) * ", v1 = " * string(model.v1), '\n',
    " ├─── learning rate: ", model.learning_rate, '\n',
    " ├─── elastic net scale: ", model.scale_el, '\n',
    " ├─── time buffer: ", model.buffer, '\n',
    " ├─── time batch size: ", model.batch_size, '\n',
    " └─── gamma mask: ", model.gamma_mask === nothing ? "none" : "$(sum(model.gamma_mask .!= -1)) fixed entries")
#

"""
    Pars Structure

Initiates a structure of class Pars

# Arguments
- A: basis coefficients
- M: PDE coefficients
- ΣZ : measurement variance/covariance matrix
- ΣU: process variance/covariance matrix
- gamma: ssvs latent parameters
- sample_inds: current subsampled indicies for SGD
- F: current F
- Fprime: current Fprime
- a_R
- A_R
- nu_R
- a_Q
- A_Q
- nu_Q
- v0
- v1
- Sigma_M

"""
mutable struct Pars

  # parameters
  # A, M, ΣY, ΣU, gamma, F, a_R, A_R, nu_R, a_Q, A_Q, nu_Q, v0, v1, Sigma_M

  # estimated parameters
  A::Array{Float64}
  M::Array{Float64}
  ΣV::Array{Float64}
  ΣU::Array{Float64}
  gamma::Array{Int}

  # current value of function
  sample_inds::Vector{Int}
  F
  Fprime
  State::StateClass

  # hyperpriors
  a_V
  A_V
  nu_V::Int

  a_U
  A_U
  nu_U::Int

  # SSVS parameters
  ΣM::Array{Float64}
  π

end

"""
    Posterior Struct

Initiates a structure of class Posterior to hold posterior samples

# Arguments

"""
mutable struct Posterior

  M
  gamma
  ΣV
  ΣU
  A
  π

end


"""
    make_H(M, N, one_inds)

Used to construct the mapping matrix H for missing data

"""
function make_H(L, N, one_inds)
  H_tmp = zeros(L, N)
  H_tmp[diagind(H_tmp)] .= one_inds
  return H_tmp
end

"""
    create_pars()

Constructs the parameter and model classes.
"""
function create_pars(Y::Array{Float64,2},
  TimeStep::Vector,
  nbasis::Int,
  buffer::Int,
  batch_size::Int,
  learning_rate::Float64,
  v0::Float64,
  v1::Float64,
  Λ::Function,
  ΛNames::Vector{String};
  degree = 4, orderTime = 1, latent_dim = size(Y, 1), covariates = nothing,
  scale_el = fill(0.01, latent_dim), gamma_mask = nothing, π_init = fill(0.5, latent_dim))

  L = size(Y, 1)
  T = size(Y, 2)
  N = latent_dim # need to update

  V = Y

  inner_inds = (buffer):(T-buffer-1)

  if covariates === nothing
    X = nothing
  else
    X = reduce(vcat, [reshape(covs[:, :, i], 1, :) for i in 1:size(covs, 3)])
  end

  # get missing values - no missing here
  H = ones(L, T)
  H = [make_H(L, N, H[:, i]) for i in 1:(T)]

  # basis functions
  Basis = derivatives_dictionary(orderTime, nbasis, TimeStep, degree = degree)

  Φ = Basis.TimeDerivative[Basis.TimeNames[1]]
  A = copy((inv(Φ'*Φ) * Φ' * V')')

  State = construct_state(A, Basis)

  # parameters
  samp_inds = sample(inner_inds, batch_size, replace = false)
  F = create_Λ(Λ, A, Basis, ΛNames, X)
  D = size(F, 2)
  # dF = create_∇Λ(Λ, A, Basis, samp_inds, ΛNames, D, nbasis, N, X)
  dF = create_∇Λ(Λ, A, Basis, samp_inds, ΛNames, X)


  # D = size(F, 2)
  M = zeros(N, D)
  ΣV = Diagonal(ones(N))
  ΣU = Diagonal(ones(N))

  # hyperpriors
  a_V = ones(N)
  A_V = fill(1e6, N)
  nu_V = 2

  a_U = ones(N)
  A_U = fill(1e6, N)
  nu_U = 2

  # SSVS parameters
  gamma = ones(Int, N, D)
  if gamma_mask !== nothing
    for idx in eachindex(gamma_mask)
      if gamma_mask[idx] != -1
        gamma[idx] = gamma_mask[idx]
      end
    end
  end
  ΣM = Diagonal([(vec(gamma)[i] == 1 ? v1 : v0) for i in 1:(D*N)])
  π = Vector{Float64}(π_init)


  Φtmp = [Basis.TimeDerivative[ΛNames[i]] for i in 1:length(ΛNames)]
  if X === nothing
    function_names = replace((@code_string Λ(A, Φtmp)).string[findfirst("return", (@code_string Λ(A, Φtmp)).string)[end]+2:end], "\n    " => " ", "\n" => "", "end" => "", "[" => "", "]" => "", "." => "")
  else
    function_names = replace((@code_string Λ(A, Φtmp, X)).string[findfirst("return", (@code_string Λ(A, Φtmp, X)).string)[end]+2:end], "\n    " => " ", "\n" => "", "end" => "", "[" => "", "]" => "", "." => "")
  end

  response = State.StateNames[end]

  # store model and parameter values
  model = Model(T, L, N, D, nbasis, buffer, batch_size, TimeStep, learning_rate, inner_inds, V, H, X, Basis, Λ, ΛNames, v0, v1, function_names, response, Vector{Float64}(scale_el), gamma_mask)
  pars = Pars(A, M, ΣV, ΣU, gamma, samp_inds, F, dF, State, a_V, A_V, nu_V, a_U, A_U, nu_U, ΣM, π)

  return pars, model

end # fully observed data

function create_pars(Y::Array{Union{Missing,Float64},2},
  TimeStep::Vector,
  nbasis::Int,
  buffer::Int,
  batch_size::Int,
  learning_rate::Float64,
  v0::Float64,
  v1::Float64,
  Λ::Function,
  ΛNames::Vector{String};
  degree = 4, orderTime = 1, latent_dim = size(Y, 1), covariates = nothing,
  scale_el = fill(0.01, latent_dim), gamma_mask = nothing, π_init = fill(0.5, latent_dim))

  L = size(Y, 1)
  T = size(Y, 2)
  N = latent_dim # need to update

  V = collect(Missings.replace(Y, 0)) # convert to Float64

  inner_inds = (buffer):(T-buffer-1)

  if covariates === nothing
    X = nothing
  else
    X = reduce(vcat, [reshape(covs[:, :, i], 1, :) for i in 1:size(covs, 3)])
  end

  # get missing values - no missing here
  # na_inds = getindex.(findall(ismissing, Y), 2)
  na_inds = hcat(getindex.(findall(ismissing, Y), 1), getindex.(findall(ismissing, Y), 2))
  H = ones(L, T)
  if (size(na_inds)[1] != 0)
    H[na_inds[:,1], na_inds[:,2]] .= 0
  end
  H = [make_H(L, N, H[:, i]) for i in 1:(T)]

  # construct data for inital guess, not saved for sampled
  VM = copy(Y)
  tmp_ind = hcat(getindex.(findall(ismissing, VM), 1), getindex.(findall(ismissing, VM), 2))
  if tmp_ind[1,2] == 1
    tmp_ind[1,2] = 3
  end
  tmp_ind[:,2] = tmp_ind[:,2] .- 1

  VM[na_inds[:,1], na_inds[:,2]] = VM[tmp_ind[:,1], tmp_ind[:,2]]
  still_missing = hcat(getindex.(findall(ismissing, VM), 1), getindex.(findall(ismissing, VM),2))
  while size(still_missing, 1) > 1
    tmp_ind = copy(still_missing)
    tmp_ind[:,2] = tmp_ind[:,2] .- 1
    VM[still_missing[:,1], still_missing[:,2]] = VM[tmp_ind[:,1], tmp_ind[:,2]]
    still_missing = hcat(getindex.(findall(ismissing, VM), 1), getindex.(findall(ismissing, VM),2))
  end

  VM = convert(Matrix{Float64}, VM) # not saved



  # basis functions
  Basis = derivatives_dictionary(orderTime, nbasis, TimeStep, degree = degree)

  Φ = Basis.TimeDerivative[Basis.TimeNames[1]]
  # A = copy((inv(Φ'*Φ) * Φ' * V')')
  A = copy((inv(Φ'*Φ) * Φ' * VM')')

  State = construct_state(A, Basis)

  # parameters
  samp_inds = sample(inner_inds, batch_size, replace = false)
  F = create_Λ(Λ, A, Basis, ΛNames, X)
  dF = create_∇Λ(Λ, A, Basis, samp_inds, ΛNames, X)

  D = size(F, 2)
  M = zeros(N, D)
  ΣV = Diagonal(ones(N))
  ΣU = Diagonal(ones(N))

  # hyperpriors
  a_V = ones(N)
  A_V = fill(1e6, N)
  nu_V = 2

  a_U = ones(N)
  A_U = fill(1e6, N)
  nu_U = 2

  # SSVS parameters
  gamma = ones(Int, N, D)
  if gamma_mask !== nothing
    for idx in eachindex(gamma_mask)
      if gamma_mask[idx] != -1
        gamma[idx] = gamma_mask[idx]
      end
    end
  end
  ΣM = Diagonal([(vec(gamma)[i] == 1 ? v1 : v0) for i in 1:(D*N)])
  π = Vector{Float64}(π_init)


  Φtmp = [Basis.TimeDerivative[ΛNames[i]] for i in 1:length(ΛNames)]
  if X === nothing
    function_names = replace((@code_string Λ(A, Φtmp)).string[findfirst("return", (@code_string Λ(A, Φtmp)).string)[end]+2:end], "\n    " => " ", "\n" => "", "end" => "", "[" => "", "]" => "", "." => "")
  else
    function_names = replace((@code_string Λ(A, Φtmp, X)).string[findfirst("return", (@code_string Λ(A, Φtmp, X)).string)[end]+2:end], "\n    " => " ", "\n" => "", "end" => "", "[" => "", "]" => "", "." => "")
  end

  response = State.StateNames[end]

  # store model and parameter values
  model = Model(T, L, N, D, nbasis, buffer, batch_size, TimeStep, learning_rate, inner_inds, V, H, X, Basis, Λ, ΛNames, v0, v1, function_names, response, Vector{Float64}(scale_el), gamma_mask)
  pars = Pars(A, M, ΣV, ΣU, gamma, samp_inds, F, dF, State, a_V, A_V, nu_V, a_U, A_U, nu_U, ΣM, π)

  return pars, model

end # missing data




"""
    update_M!(pars)

Used within sgmcmc() function.
Gibbs step for M.
"""
function update_M!(pars, model)

  V = (pars.F[model.inner_inds, :]' ⊗ I(model.N)) * (Diagonal(ones(length(model.inner_inds))) ⊗ inv(pars.ΣU)) * transpose(pars.F[model.inner_inds, :]' ⊗ I(model.N)) + inv(pars.ΣM)
  a = (pars.F[model.inner_inds, :]' ⊗ I(model.N)) * (Diagonal(ones(length(model.inner_inds))) ⊗ inv(pars.ΣU)) * vec(pars.State.State[model.response][:,model.inner_inds])
 
  C = cholesky(Hermitian(V))
  b = rand(Normal(0.0, 1.0), model.N * model.D)

  pars.M = reshape(C.U \ ((C.L \ a) + b), model.N, :)

  return pars
end



"""
    update_gamma!(pars)

Used within sgmcmc() function.
Gibbs step for latent SSVS variables.
"""
function update_gamma!(pars, model)

  p1 = pdf.(Normal(0, sqrt(model.v1)), pars.M) .* pars.π
  p0 = pdf.(Normal(0, sqrt(model.v0)), pars.M) .* (1 .- pars.π)
  p = p1 ./ (p1 + p0)
  p = replace!(p, NaN => 0)

  pars.gamma = rand.(Binomial.(1, p))

  # apply fixed gamma mask: override sampled values for any masked entries
  if model.gamma_mask !== nothing
    for idx in eachindex(model.gamma_mask)
      if model.gamma_mask[idx] != -1
        pars.gamma[idx] = model.gamma_mask[idx]
      end
    end
  end

  pars.ΣM = Diagonal([(vec(pars.gamma)[i] == 1 ? model.v1 : model.v0) for i in 1:(model.D*model.N)])
  # for i in 1:model.N
  #   pars.π[i] = rand(Beta(1 + sum(pars.gamma[i, :]), 1 + sum(1 .- pars.gamma[i, :])))
  # end

  return pars
end


"""
  update_ΣV!(pars)

Used within sgmcmc() function.
Gibbs step for measurement error.
"""
function update_ΣV!(pars, model)

  sse_tmp = [model.V[:, j] - model.H[j] * pars.State.State[pars.State.StateNames[1]][:, j] for j in model.inner_inds]
  sse_tmp = reduce(hcat, sse_tmp)
  for n in 1:model.N
    a_hat = (length(model.inner_inds) + pars.nu_V) / 2
    b_hat = (0.5*(sse_tmp[n, :]'*sse_tmp[n, :]) + pars.nu_V[1]/pars.a_V[n])[1]
    pars.ΣV[n,n] = rand(InverseGamma(a_hat, b_hat))
  end

  for n in 1:model.N
    a_hat = (pars.nu_V + 1) / 2
    b_hat = (pars.nu_V / pars.ΣV[n,n]) + (1 / pars.A_V[n]^2)
    pars.a_V[n] = rand(InverseGamma(a_hat, b_hat))
  end

  return pars
end


"""
  update_ΣU!(pars)

Used within sgmcmc() function.
Gibbs step for process error.
"""
function update_ΣU!(pars, model)

  Uprime = pars.State.State[model.response][:,model.inner_inds]
  
  nu_hat = pars.nu_U + model.N + length(model.inner_inds) - 1
  Phi_hat = 2 * pars.nu_U * Diagonal(1 ./ pars.a_U) + (Uprime - pars.M * pars.F[model.inner_inds,:]') * transpose(Uprime - pars.M * pars.F[model.inner_inds,:]')
  pars.ΣU = rand(InverseWishart(nu_hat, cholesky(Hermitian(Phi_hat))))

  a_hat = (pars.nu_U + model.N) / 2
  for n in 1:model.N
    b_hat = (pars.nu_U / pars.ΣU[n, n]) + (1 / pars.A_U[n]^2)
    pars.a_U[n] = rand(InverseGamma(a_hat, b_hat))
  end

  return pars
end



"""
  ΔL(z, H, ψ, gψ, ϕ, ϕ_t, Θ, ΣZinv, ΣUinv, A, M, fcurr, fprime)

Used within DEtection() function.
Calculates the gradient of the log likelihood.
"""
function ΔL(v, H, ϕ, ϕ_t, ΣVinv, ΣUinv, A, M, fcurr, fprime)

  ΔL = -H' * ΣVinv * v * ϕ' +
        H' * ΣVinv * H * A * ϕ * ϕ' +
        ΣUinv  * A * ϕ_t * ϕ_t' -
        ΣUinv * M * fcurr * ϕ_t' -
        reduce(vcat, [ϕ_t' * A'  * ΣUinv * M * fprime[i] - (fcurr' * M' * ΣUinv * M) * fprime[i] for i in 1:size(fprime,1)])
        # reduce(vcat, [ϕ_t' * A'  * ΣUinv * M * fprime[:,:,i] - (fcurr' * M' * ΣUinv * M) * fprime[:,:,i] for i in 1:size(fprime,3)])

  return ΔL

end


"""
    update_A!(pars)

Updates 𝐀 with the elastic net prior (Li 2010)

"""
function update_A!(pars, model)

  samp_inds = sample(model.inner_inds, model.batch_size, replace = false)

  v = model.V[:, samp_inds] # if an error occurs with 1D vs 2D space, it is here
  h = model.H[samp_inds]
  ϕ = model.Basis.TimeDerivative[model.Basis.TimeNames[1]][samp_inds, :]
  ϕ_t = model.Basis.TimeDerivative[model.Basis.TimeNames[end]][samp_inds, :]
  ΣVinv = inv(pars.ΣV)
  ΣUinv = inv(pars.ΣU)
  A = pars.A
  M = pars.M

  fcurr = pars.F[samp_inds,:]
  # fprime = create_∇Λ(model.Λ, A, model.Basis, samp_inds, model.ΛTimeNames, model.D, model.νT, model.N, model.X)
  fprime = create_∇Λ(model.Λ, A, model.Basis, samp_inds, model.ΛTimeNames, model.X)


  # dq = [ΔL(v[:, i], h[i], ϕ[i, :], ϕ_t[i, :], ΣVinv, ΣUinv, A, M, fcurr[i,:], fprime[i,:][1]) for i in 1:length(samp_inds)]
  dq = [ΔL(v[:, i], h[i], ϕ[i, :], ϕ_t[i, :], ΣVinv, ΣUinv, A, M, fcurr[i,:], fprime[i,:]) for i in 1:length(samp_inds)]


  sel = reshape(model.scale_el, :, 1)
  dq = mean(dq) + (1 / (model.T)) * (sel .* sign.(A) + 2 .* sel .* A)


  # pars.A = A - model.learning_rate * dq
  pars.A = A - dq .* model.learning_rate

  # save updated state information
  pars.sample_inds = samp_inds
  pars.State = construct_state(pars.A, model.Basis)
  pars.F = create_Λ(model.Λ, pars.A, model.Basis, model.ΛTimeNames, model.X)
  pars.Fprime = fprime

  return pars
end


"""
    DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, Λnames)

DEtection sampler function. Can accept missing values in the input data argument.
Returns the model, parameters, and posterior values.
The model are the model settings.
The parameters are the final value of the parameters in from the sampler.
The posterior are the saved posterior values.

Consider the function ``U_t = M(U, U^2, U^3, ...)``.
``DEtection_sampler()`` is used to determine ``M`` given a library of potential values, `Λ(U)`, to search over. 
Within the function, derivatives are denoted as ``ΔU_t, ΔU_tt, ...``, so a potential function could be

```jldoctest 
ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  X = u[1,:]
  Y = u[2,:]
  Z = u[3,:]
  
  return [X, Y, Z, X .* Y, X .* Z, Y .* Z, X.^2, Y.^2, Z.^2, X.^2 .* Y, X .* Y.^2, X.^2 .* Z, X .* Z.^2, Y.^2 .* Z, Y .* Z.^2, X.^3, Y.^3, Z.^3, X .* Y .* Z]

end
```

The function must take in ``A`` and ``Φ``, where ``Φ`` is a vector of the basis functions.

To make the function identify the correct partial derivatives, the arguments that are passed into `Λ`, `Λnames`, are required where.
For the example above, `Λnames = ["Phi"]` because the function `Λ` does not take any higher order derivatives.
If the function accepted higher order derivatives,  `Λnames = ["Phi", "Phi_t", ...]`.


# Required Arguments (in order)
- `Y`: Input data. Needs to be Array{Float64, 2} or Array{Union{Missing, Float64}, 2} where the dimensions are Components by Time
- `TimeStep`: of type Vector()
- `nbasis::Int`: Number of temporal basis functions
- `buffer::Int`: Buffer on each side of the data to negate edge effects
- `batch_size::Int`: size of the minibatch
- `learning_rate::Float64`: Learning rate for the stochastic gradient descent with constant learning rate
- `v0::Float64`: SSVS exclusion variance
- `v1::Float64`: SSVS inclusion variance
- `Λ::Function`: Library to search over
- `Λnames::Vector{String}`: Names of derivatives of library (see example)

# Optional Arguments
- `degree` = 4: Degree of the B-spline. Must be at least one order higher than the highest order partial derivative.
- `orderTime` = 1: Order of the highest order temporal derivative (default U_t)
- `latent_dim` = size(Y, 1): Dimension of the latent space. Default is same as data dimension.
- `covariates` = nothing: Additional covariates.
- `nits` = 2000: Number of samples for the Gibbs sampler.
- `burnin` = nits / 2: Number of samples to discard as burnin (default is half of `nits`).
- `learning_rate_end` = learning_rate: End learning rate (default is same as initial learning rate).


# Examples

```jldoctest 
Y = Array{Float64}(rand, 3, 1000) # component (3) by time (1000)
TimeStep = Vector(range(0.01, end_time, step = 0.01))
batch_size = 50
buffer = 20
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e1

Λnames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  X = u[1,:]
  Y = u[2,:]
  Z = u[3,:]
  
  return [X, Y, Z, X .* Y, X .* Z, Y .* Z, X.^2, Y.^2, Z.^2, X.^2 .* Y, X .* Y.^2, X.^2 .* Z, X .* Z.^2, Y.^2 .* Z, Y .* Z.^2, X.^3, Y.^3, Z.^3, X .* Y .* Z]

end

# not run
# model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 2000)
``` 

# Example Higher Order Library

```jldoctest 
ΛNames = ["Phi", "Phi_t"]

function Λ(A, Φ)
    
  ϕ = Φ[1] # corresponds to no derivative
  ϕt = Φ[2] # corresponds to first derivative
  x = (A * ϕ')'
  xt = (A * ϕt')'
  
  return [x, sin.(x), cos.(x), x ./ sin.(x), x ./ cos.(x), xt, xt.^2, xt .* sin.(x), xt .* cos.(x), sin.(xt), cos.(xt)]
  
end
``` 
"""
function DEtection_sampler(Y,
  TimeStep::Vector,
  nbasis::Int,
  buffer::Int,
  batch_size::Int,
  learning_rate::Float64,
  v0::Float64,
  v1::Float64,
  Λ::Function,
  ΛNames::Vector{String};
  degree = 4, orderTime = 1, latent_dim = size(Y, 1), covariates = nothing, nits = 2000, burnin = nits / 2, learning_rate_end = learning_rate,
  scale_el = fill(0.01, latent_dim), gamma_mask = nothing, π_init = fill(0.5, latent_dim))

  pars, model = create_pars(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, degree = degree, orderTime = orderTime, latent_dim = latent_dim, covariates = covariates, scale_el = scale_el, gamma_mask = gamma_mask, π_init = π_init)

  keep_samps = Int(nits - burnin)
  n_change = log10(learning_rate / learning_rate_end)
  if n_change == 0
    change_times = 0.0
  else
    change_times = floor.(collect(range(1, stop=burnin, length=Int((n_change + 1)))))[2:end]
  end

  M_post = Array{Float64}(undef, model.N, model.D, keep_samps)
  gamma_post = Array{Float64}(undef, model.N, model.D, keep_samps)
  ΣV_post = Array{Float64}(undef, model.N, model.N, keep_samps)
  ΣU_post = Array{Float64}(undef, model.N, model.N, keep_samps)
  A_post = Array{Float64}(undef, model.N, size(pars.A)[2], keep_samps)
  π_post = Array{Float64}(undef, model.N, keep_samps)


  @showprogress 1 "Burnin..." for i in 1:burnin

    pars = update_M!(pars, model)
    pars = update_gamma!(pars, model)
    pars = update_ΣV!(pars, model)
    pars = update_ΣU!(pars, model)
    pars = update_A!(pars, model)

    if i in change_times
      raise_pow = findall(change_times .== i)[1]
      model.learning_rate = round(learning_rate * 10.0^(-raise_pow), digits = raise_pow+6)
    end

  end

  @showprogress 1 "Sampling..." for i in 1:keep_samps

    pars = update_M!(pars, model)
    pars = update_gamma!(pars, model)
    pars = update_ΣV!(pars, model)
    pars = update_ΣU!(pars, model)
    pars = update_A!(pars, model)

    M_post[:, :, i] = pars.M
    gamma_post[:, :, i] = pars.gamma
    ΣV_post[:, :, i] = pars.ΣV
    ΣU_post[:, :, i] = pars.ΣU
    A_post[:, :, i] = pars.A
    π_post[:,i] = pars.π

  end

  posterior = Posterior(M_post, gamma_post, ΣV_post, ΣU_post, A_post, π_post)

  return model, pars, posterior

end # 1 spatial dimension DEtection
