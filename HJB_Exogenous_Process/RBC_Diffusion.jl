#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an RBC model with a Diffusion process

	Translated Julia code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm

        Updated to Julia 1.0.0
==============================================================================#

using Distributions, Plots, LinearAlgebra, SparseArrays

γ= 2.0 #gamma parameter for CRRA utility
ρ = 0.05 #the discount rate
α = 0.3 # the curvature of the production function (cobb-douglas)
δ = 0.05 # the depreciation rate



# Z our state variable follows this process
	#= for this process:
	 		dlog(z) = -θ⋅log(z)dt + σ^2⋅dw
		and
			log(z)∼N(0,var) where var = σ^2/(2⋅θ) =#
var = 0.07
μ_z = exp(var/2)
corr = 0.9
θ = -log(corr)
σ_sq = 2*θ*var

#=============================================================================
 	k our capital follows a process that depends on z,

	using the regular formula for capital accumulation
	we would have:
		(1+ρ)k_{t+1} = k_{t}⋅f'(k_{t}) + (1-δ)k_{t}
	where:
		f(k_{t}) = z⋅k^{α} so f'(k_{t}) = (α)⋅z⋅k^{α-1}
	so in steady state where k_{t+1} = k_{t}
		(1+ρ)k = α⋅z⋅k^{α} + (1-δ)k
		k = [(α⋅z)/(ρ+δ)]^{1/(1-α)}

=============================================================================#
#K_starting point, for mean of z process

k_st = ((α⋅μ_z)/(ρ+δ))^(1/(1-α))

# create the grid for k
H = 15 #number of points on grid
k_min = 0.3*k_st # min value
k_max = 3*k_st # max value
k = LinRange(k_min, k_max, H)
dk = (k_max-k_min)/(H-1)

# create the grid for z
J = 45
z_min = μ_z*0.8
z_max = μ_z*1.2
z = LinRange(z_min, z_max, J)
dz = (z_max-z_min)/(J-1)
dz_sq = dz^2


# Check the pdf to make sure our grid isn't cutting off the tails of
	# our distribution
y = pdf.(LogNormal(0, var), z)
plot(z,y, grid=false,
		xlabel="z", ylabel="Probability",
		legend=false, color="purple", title="PDF of z")
png("PDF_Z")

#create matrices for k and z
z= convert(Array, z)'
kk = k*ones(1,J)
zz = ones(H,1)*z

# use Hto's lemma to find the drift and variance of our optimization equation

μ = (-θ*log.(z).+σ_sq/2).*z # the drift from Hto's lemma
Σ_sq = σ_sq.*z.^2 #the variance from Hto's lemma

max_it = 100
ε = 0.1^(6)
Δ = 1000

# set up all of these empty matrices
Vaf_1, Vab_1, Vzf_1, c_1 = [zeros(H,J) for i in 1:6]

#==============================================================================

    Now we are going to construct a matrix summarizing the evolution of V_z

    This comes from the following discretized Bellman equation:

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + v_z⋅μ(z)
                                   + 1/2(v_{zz})σ^2(z)

                                   or

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + ((v_{i,j+1}-v_{i,j})/Δz)μ(z)
                            + 1/2((v_{i,j+1}-2v_{i,j}+v_{ij-1})/Δz^2)σ^2(z)

    Assume forward difference because of boundary conditions

==============================================================================#

 yy = (-Σ_sq/dz_sq - μ/dz)
 χ = Σ_sq/(2*dz_sq)
 ζ = μ/dz + Σ_sq/(2*dz_sq)


 # Define the diagonals of this matrix
 updiag = zeros(H,1)
 	for j = 1:J
		global updiag =[updiag; repeat([ζ[j]], H, 1)]
	end
 updiag =(updiag[:])


 centerdiag=repeat([χ[1]+yy[1]],H,1)
	for j = 2:J-1
		global centerdiag = [centerdiag; repeat([yy[j]], H, 1)]
	end
 centerdiag=[centerdiag; repeat([yy[J]+ζ[J]], H, 1)]
 centerdiag = centerdiag[:]

lowdiag = repeat([χ[2]], H, 1)
	for j=3:J
		global lowdiag = [lowdiag; repeat([χ[j]],H,1)]
	end
lowdiag=lowdiag[:]

# spdiags in Matlab allows for automatic trimming/adding of zeros
    # spdiagm does not do this

B_switch = sparse(Diagonal(centerdiag))+ [zeros(H,H*J);  sparse(Diagonal(lowdiag)) zeros(H*(J-1), H)]+ sparse(Diagonal(updiag))[(H+1):end,1:(H*J)] # trim off rows of zeros


# Now it's time to solve the model, first put in a guess for the value function
v0 = (zz.*kk.^α).^(1-γ)/(1-γ)/ρ
v_1=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

for n = 1:maxit
    V=v_1

    #Now set up the forward difference

    Vaf_1[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vaf_1[H,:] = (z.*k_max.^α .- δ.*k_max).^(-γ) # imposes a constraint

    #backward difference
    Vab_1[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vab_1[1,:] = (z.*k_min.^α .- δ.*k_min).^(-γ)

    #H_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf_1.^(-1/γ)
    sf = zz .* kk.^α - δ.*kk - cf

    # consumption and saving backwards difference

    cb = Vab_1.^(-1.0/γ)
    sb = zz .* kk.^α - δ.*kk - cb
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = zz.*kk.^α - δ.*kk
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    Hf = sf.>0 # positive drift will ⇒ forward difference
    Hb = sb.<0 # negative drift ⇒ backward difference
    H0=(1.0.-Hf-Hb) # at steady state

    Va_upwind = Vaf_1.*Hf + Vab_1.*Hb + Va0.*H0 # need to include SS term

    global c_1 = Va_upwind.^(-1/γ)
    u = (c_1.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min.(sb, 0)/dk
    #println(X)
    Y = -max.(sf, 0)/dk + min.(sb, 0)/dk
    Z = max.(sf, 0)/dk

    updiag_k = 0
       for j = 1:J
           updiag_k =[updiag_k; Z[1:H-1,j]; 0]
       end

    centerdiag_k=reshape(Y, H*J,1)
    centerdiag_k=(centerdiag_k[:])

   lowdiag_k = X[2:H, 1]
       for j = 2:J
           lowdiag_k = [lowdiag_k; 0; X[2:H,j]]
       end

   # Upated using julia 1.0.0
  AA = sparse(Diagonal(centerdiag_k))+ [zeros(1, H*J); sparse(Diagonal(lowdiag_k)) zeros(H*J-1,1)] + sparse(Diagonal(updiag_k))[2:end, 1:(H*J)] # trim first element

  A = AA + B_switch
  B = (1/Δ + ρ)*sparse(1.0I, H*J, H*J) - A

  u_stacked= reshape(u, H*J, 1)
  V_stacked = reshape(V,H*J, 1)

  b = u_stacked + (V_stacked./Δ)

  V_stacked = B\b

  V = reshape(V_stacked, H, J)

  V_change = V-v_1

  global v_1= V

  # need push function to add to an already existing array
  push!(dist, findmax(abs.(V_change))[1])
  if dist[n].< ε
      println("Value Function Converged Iteration=")
      println(n)
      break
  end

end

# calculate the savings for kk
ss_1 = zz.*kk.^α - δ.*kk - c_1

# Plot the savings vs. k
plot(k, ss_1, grid=false,
		xlabel="k", ylabel="s(k,z)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Savings Policies")
plot!(k, zeros(H,1))

png("OptimalSavings")

plot(k, v_1, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max),
		legend=false, title="")
png("Value_function_vs_k")

z = LinRange(z_min, z_max, J)
plot(z, v_1', grid=false,
		xlabel="z", ylabel="V(z)",
		xlims=(z_min,z_max),
		legend=false, title="")
png("Value_function_vs_z")


#===========================================================================#


#  Now the agents misspecify θ and σ^2
θ_g = .005
σ_sq_g = 2.0

# They will update this based on a draw from a binomial

Dist = Bernoulli(.5)

T = 1000 #periods of time for updating

μ_g = (-θ_g*log.(z)+σ_sq_g/2).*z # the drift from Ito's lemma
Σ_sq_g = σ_sq_g.*z.^2 #the variance from Ito's lemma

max_it = 100
ε = 0.1^(6)
Δ = 1000

# set up all of these empty matrices
Vaf_2, Vab_2,c_2 = [zeros(H,J) for i in 1:6]

#==============================================================================

    Now we are going to construct a matrix summarizing the evolution of V_z

==============================================================================#

 yy_g = (-Σ_sq_g/dz_sq - μ_g/dz)
 χ_g = Σ_sq_g/(2*dz_sq)
 ζ_g = μ_g/dz + Σ_sq_g/(2*dz_sq)


 # Define the diagonals of this matrix
 updiag_g = zeros(H,1)
 	for j = 1:J
		global updiag_g =[updiag_g; repmat([ζ_g[j]], H, 1)]
	end
 updiag_g =(updiag_g[:])


 centerdiag_g=repmat([χ_g[1]+yy_g[1]],H,1)
	for j = 2:J-1
		global centerdiag_g = [centerdiag_g; repmat([yy_g[j]], H, 1)]
	end
 centerdiag_g=[centerdiag_g; repmat([yy_g[J]+ζ_g[J]], H, 1)]
 centerdiag_g = centerdiag_g[:]

lowdiag_g = repmat([χ_g[2]], H, 1)
	for j=3:J
		global lowdiag_g = [lowdiag_g; repmat([χ_g[j]],H,1)]
	end
lowdiag_g=lowdiag_g[:]

# spdiags in Matlab allows for automatic trimming/adding of zeros
    # spdiagm does not do this

B_switch_g = sparse(Diagonal(centerdiag_g))+ [zeros(H,H*J);  sparse(Diagonal(lowdiag_g)) zeros(H*(J-1), H)]+ sparse(Diagonal(updiag_g))[(H+1):end,1:(H*J)] # trim off rows of zeros


# Now it's time to solve the model, first put in a guess for the value function
v0 = (zz.*kk.^α).^(1-γ)/(1-γ)/ρ
v_2=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

V_tot = []
for t = 1:T
        indicator = rand(Dist)
    for n = 1:maxit
    V=v_2

    #Now set up the forward difference

    Vaf_2[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vaf_2[H,:] = (z.*k_max.^α .- δ.*k_max).^(-γ) # imposes a constraint

    #backward difference
    Vab_2[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vab_2[1,:] = (z.*k_min.^α .- δ.*k_min).^(-γ)

    #H_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf_2.^(-1/γ)
    sf = zz .* kk.^α - δ.*kk - cf

    # consumption and saving backwards difference

    cb = Vab_2.^(-1.0/γ)
    sb = zz .* kk.^α - δ.*kk - cb
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = zz.*kk.^α - δ.*kk
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    Hf = sf.>0 # positive drift will ⇒ forward difference
    Hb = sb.<0 # negative drift ⇒ backward difference
    H0=(1.0.-Hf-Hb) # at steady state

    Va_upwind = Vaf_2.*Hf + Vab_2.*Hb + Va0.*H0 # need to include SS term

    global c_2 = Va_upwind.^(-1/γ)
    u = (c_2.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min.(sb, 0)/dk
    #println(X)
    Y = -max.(sf, 0)/dk + min.(sb, 0)/dk
    Z = max.(sf, 0)/dk

    updiag_k = 0
       for j = 1:J
           updiag_k =[updiag_k; Z[1:H-1,j]; 0]
       end
    updiag_k =(updiag_k[:])

    centerdiag_k=reshape(Y, H*J, 1)
    centerdiag_k = (centerdiag_k[:]) # for tuples

    lowdiag_k = X[2:H, 1]
       for j = 2:J
           lowdiag_k = [lowdiag_k; 0; X[2:H,j]]
       end
    lowdiag_k=(lowdiag_k)

    # spdiags in Matlab allows for automatic trimming/adding of zeros
       # spdiagm does not do this
    AA = sparse(Diagonal(centerdiag_k))+ [zeros(1, H*J); sparse(Diagonal(lowdiag_k)) zeros(H*J-1,1)] + sparse(Diagonal(updiag_k))[2:end, 1:(H*J)] # trim first element

    A = AA + B_switch_g
    B = (1/Δ + ρ)*sparse(I,H*J,H*J) - A

    u_stacked= reshape(u, H*J, 1)
    V_stacked = reshape(V,H*J, 1)

    b = u_stacked + (V_stacked./Δ)

    V_stacked = B\b

    V = reshape(V_stacked, H, J)

    V_change = V-v_2

    global v_2= V

    # need push function to add to an already existing array
    push!(dist, findmax(abs.(V_change))[1])
    if dist[n].< ε
      println("Value Function Converged Iteration=")
      println(n)
      break
    end
    end

    if indicator == 1
      B_switch_g = B_switch_g + .01(B_switch - B_switch_g)
    end

    push!(V_tot, v_2[:,20])
end

# calculate the savings for kk
ss_2 = zz.*kk.^α - δ.*kk - c_2

# Plot the savings vs. k
plot(k, ss_2, grid=false,
		xlabel="k", ylabel="s(k,z)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Savings Policies")
plot!(k, zeros(H,1))
png("OptimalSavings_2")

plot(k, v_2, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max),
		legend=false, title="Second Value Function over k")
png("Value_function_vs_k_2")

z = LinRange(z_min, z_max, J)
plot(z, v_2', grid=false,
		xlabel="z", ylabel="V(z)",
		xlims=(z_min,z_max),
		legend=false, title="Second Value Function over z")
png("Value_function_vs_z_2")


plot(k, V_tot, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max),
		legend=false, title="Second Value Median values of z over k")
png("Value_function_vs_k_2_median")


plot(k, v_1[:,20], grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max), title="Value Function over k for Median z",
        legend=:bottomright, label="Correct Specification")
plot!(k,V_tot[500], label="Misspecfication after 500 periods")
plot!(k,V_tot[1000], label="Misspecfication after 1000 periods", line=:dot)
plot!(k,V_tot[1], label="Misspecfication", line=:dash, color=:black)
png("Value_functions_median_Z")
