#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an RBC model with a Diffusion process

	Orignially based on Matlab code from Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm

        Updated to Julia 1.0.0
==============================================================================#

using Distributions, Plots, SparseArrays, LinearAlgebra, DifferentialEquations

using Random

Random.seed!(123)

include("B_Switch.jl")

γ= 2.0 #gamma parameter for CRRA utilit
ρ = 0.05 #the discount rate
α = 0.3 # the curvature of the production function (cobb-douglas)
δ = 0.05 # the depreciation rate



# Create a continuous time OrnsteinUhlenbeckProcess
var = 0.07
T = 10001 # Forecasting periods/ length of process
T_obs= 100# set the number of inital observations
global dt_inv = 5 # Does this dt value matter?
global dt= 1/dt_inv

OU_process=zeros((T+T_obs)*dt_inv)
OU_process[1]=exp(var/2)

# Z our state variable follows a stochastic process with the given parameters
corr = 0.9
θ = -log(corr)
σ = 1.0
ε_OU = rand(Normal(0,1), (T+T_obs)*dt_inv).*sqrt(dt)


for i in 1:((T+T_obs)*dt_inv-1)
	OU_process[i+1]=(1-θ*dt)*OU_process[i].+σ*ε_OU[i]
end

cd("/home/chandler/Desktop/Simple_Exogenous_Rule/Updating_const_gain/")
# Now plot the process
plot(OU_process[:], grid=false,
		xlabel="Time", ylabel="z",
		legend=false, color="red", title="\$ \\textrm{Ornstein-Uhlenbeck Process for } \\log(z_t) \$")
png("OU_Process")

# Step up grid space
var = mean(OU_process)
μ_z = exp(var/2)

#=============================================================================
 	k our capital follows a process that depends on z,

	using the regular formula for capital accumulation
	we would have:
		(1+ρ)k_{t+1} = k_{t}⋅f'(k_{t}) + (1-δ)k_{t}
	where:
		f(k_{t}) = z⋅k^{α} so f'(k_{t}) = (α)⋅z⋅k^{α-1}
	so in steady state where k_{t+1} = k_{t}
		(1+ρ)k = α⋅z⋅k^{α} + (1-δ)k
		k = [(α⋅z)/(ρ+δ)]^{1/(1-α)

=============================================================================#
#K_starting point, for mean of z process

k_st = ((α*exp(μ_z))/(ρ+δ))^(1/(1-α))

# create the grid for k
H = 100 #number of points on grid
k_min = 0.3*k_st # min value
k_max = 9.0*k_st # max value
k = LinRange(k_min, k_max, H)
dk = (k_max-k_min)/(H-1)

# create the grid for z
J = 40
z_min = μ_z*0.8
z_max = μ_z.*1.2
z = LinRange(z_min, z_max, J)
dz = (z_max-z_min)/(J-1)
dz_sq = dz^2


# Check the pdf to make sure our grid isn't cutting off the tails of
	#= our distribution
y = pdf.(LogNormal(0, var), z)
plot(z,y, grid=false,
		xlabel="z", ylabel="Probability",
		legend=false, color="purple", title="PDF of z")
png("PDF_of_z")=#

#create matrices for k and z
z= convert(Array, z)'
kk = k*ones(1,J)
zz = ones(H,1)*z

# use Ito's lemma to find the drift and variance of our optimization equation

μ = (-θ*log.(z).+σ.^2/2).*z # the drift from Ito's lemma
Σ_sq = σ.^2 .*z.^2#the variance from Ito's lemma
global B_switch_SS = B_switch(μ, Σ_sq, dz, H)

max_it = 100
ε = 0.1^(6)
Δ = 1000

# set up all of these empty matrices
Vaf, Vab, Vzf, Vzb, Vzz = [zeros(H,J) for i in 1:6]

# Now it's time to solve the model, first put in a guess for the value function
v0 = (zz.*kk.^α).^(1-γ)/(1-γ)/ρ
v=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

for n = 1:maxit
    V=v

    #Now set up the forward difference

    Vaf[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vaf[H,:] = (z.*k_max.^α .- δ.*k_max).^(-γ) # imposes a constraint

    #backward difference
    Vab[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vab[1,:] = (z.*k_min.^α .- δ.*k_min).^(-γ)

    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf.^(-1/γ)
    sf = zz .* kk.^α - δ.*kk - cf

    # consumption and saving backwards difference

    cb = Vab.^(-1.0/γ)
    sb = zz .* kk.^α - δ.*kk - cb
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = zz.*kk.^α - δ.*kk
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    If = sf.>0 # positive drift will ⇒ forward difference
    Ib = sb.<0 # negative drift ⇒ backward difference
    I0=(1.0.-If-Ib) # at steady state

    Va_upwind = Vaf.*If + Vab.*Ib + Va0.*I0 # need to include SS term

    global c = Va_upwind.^(-1/γ)
    u = (c.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min.(sb, zeros(H,1))/dk
    #println(X)
    Y = -max.(sf, zeros(H,1))/dk + min.(sb, zeros(H,1))/dk
    Z = max.(sf, zeros(H,1))/dk

    updiag = 0
       for j = 1:J
           updiag =[updiag; Z[1:H-1,j]; 0]
       end
    updiag =(updiag[:])

    centerdiag=reshape(Y, H*J, 1)
    centerdiag = (centerdiag[:]) # for tuples

   lowdiag = X[2:H, 1]
       for j = 2:J
           lowdiag = [lowdiag; 0; X[2:H,j]]
       end
   lowdiag=(lowdiag)

   # spdiags in Matlab allows for automatic trimming/adding of zeros
       # spdiagm does not do this
  AA = sparse(Diagonal(centerdiag))+ [zeros(1, H*J); sparse(Diagonal(lowdiag)) zeros(H*J-1,1)] + sparse(Diagonal(updiag))[2:end, 1:(H*J)] # trim first element

  A = AA + B_switch_SS
  B = (1/Δ + ρ)*sparse(I, H*J, H*J) - A

  u_stacked= reshape(u, H*J, 1)
  V_stacked = reshape(V,H*J, 1)

  b = u_stacked + (V_stacked./Δ)

  V_stacked = B\b

  V = reshape(V_stacked, H, J)

  V_change = V-v

  global v= V

  # need push function to add to an already existing array
  push!(dist, findmax(abs.(V_change))[1])
  if dist[n].< ε
      println("Value Function Converged Iteration=")
      println(n)
      break
  end

end

v1=v

# calculate the savings for kk
ss = zz.*kk.^α - δ.*kk - c

# Plot the savings vs. k
plot(kk, ss, grid=false,
		xlabel="k", ylabel="s(k,z)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Savings Policies")
plot!(k, zeros(H,1), color=:black, line=:dash)
png("OptimalSavings")


#==============================================================================

		Now, we have found the steady state,
				we can move on to misspecification and forecasting

		Our Agent willplot(guesses_θ[:], label="Estimates",
title="\$ \\textrm{Estimate of } \\theta \\textrm{ over time}\$", legend=:bottomright)
plot!(θ.*ones(T,1), label="True value", legend=:topright) now misspecify η the drift parameter from our process
		They will then forecast an AR(1) and use this to update their
		forecast of the process, since ϕ = 1-η.

		Therefore η = 1-ϕ

==============================================================================#

# Our Agent will now misspecify η the drift parameter from our process
# They will then forecast an AR(1) and use this to update

# Use the initai points to form a OLS estimate of our coefficients

X = [ones(T_obs,1) OU_process[1:T_obs]]
Y = OU_process[1+dt_inv:T_obs+dt_inv]
ϵ_OU = ε_OU[1:T_obs]*sqrt(dt)

η_g = (X'X)^(-1) * X'Y
θ_g = (1-η_g[2])
const_g =η_g[1]

#σ_g = sqrt(cov(Y-X*η_g))
σ_g =σ #or sigma g is known

R_g = X'X*dt
#R_g = [1 0 ; 0 1]#Use identity matrix for R_g?
guesses_θ,guesses_σ, guesses_const =[zeros(1,T-1) for i in 1:3]

Value_functions=[]


v =v0

for t = 1:T-1
	global v=v0
	global θ_g = θ_g
	global const_g = const_g
	global σ_g =σ_g
	global R_g = R_g
	guesses_θ[1,t] = θ_g
	guesses_σ[1,t] = σ_g
	guesses_const[1,t]=const_g

	μ = (-θ_g*log.(z).+σ_g.^2/2).*z # the drift from Ito's lemma
	Σ_sq = σ_g.^2 .*z.^2#the variance from Ito's lemma

	global B_switch_G = B_switch(μ, Σ_sq, dz, H)

	for n = 1:maxit
	    V=v
	    #Now set up the forward difference
	    Vaf[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
	    Vaf[H,:] = (z.*k_max.^α .- δ.*k_max).^(-γ) # imposes a constraint

	    #backward difference
	    Vab[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
	    Vab[1,:] = (z.*k_min.^α .- δ.*k_min).^(-γ)

	    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

	    # Consumption and savings functions
	    cf = Vaf.^(-1/γ)
	    sf = zz .* kk.^α - δ.*kk - cf

	    # consumption and saving backwards difference

	    cb = Vab.^(-1.0/γ)
	    sb = zz .* kk.^α - δ.*kk - cb
	    #println(sb)
	    #consumption and derivative of the value function at the steady state

	    c0 = zz.*kk.^α - δ.*kk
	    Va0 = c0.^(-γ)

	    # df chooses between the forward or backward difference

	    If = sf.>0 # positive drift will ⇒ forward difference
	    Ib = sb.<0 # negative drift ⇒ backward difference
	    I0=(1.0.-If-Ib) # at steady state

	    Va_upwind = Vaf.*If + Vab.*Ib + Va0.*I0 # need to include SS term

	    global c = Va_upwind.^(-1/γ)
	    u = (c.^(1-γ))/(1-γ)

	    # Now to constuct the A matrix
	    X = -min.(sb, zeros(H,1))/dk
	    #println(X)
	    Y = -max.(sf, zeros(H,1))/dk + min.(sb, zeros(H,1))/dk
	    Z = max.(sf, zeros(H,1))/dk

	    updiag = 0
	       for j = 1:J
	           updiag =[updiag; Z[1:H-1,j]; 0]
	       end
	    updiag =(updiag[:])

	    centerdiag=reshape(Y, H*J, 1)
	    centerdiag = (centerdiag[:]) # for tuples

	   lowdiag = X[2:H, 1]
	       for j = 2:J
	           lowdiag = [lowdiag; 0; X[2:H,j]]
	       end
	   lowdiag=(lowdiag)

	   # spdiags in Matlab allows for automatic trimming/adding of zeros
	       # spdiagm does not do this
	  AA = sparse(Diagonal(centerdiag))+ [zeros(1, H*J); sparse(Diagonal(lowdiag)) zeros(H*J-1,1)] + sparse(Diagonal(updiag))[2:end, 1:(H*J)] # trim first element

	  A = AA + B_switch_G
	  B = (1/Δ + ρ)*sparse(I, H*J, H*J) - A

	  u_stacked= reshape(u, H*J, 1)
	  V_stacked = reshape(V,H*J, 1)

	  b = u_stacked + (V_stacked./Δ)

	  V_stacked = B\b

	  V = reshape(V_stacked, H, J)

	  V_change = V-v

	  Vaf[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
	  Vaf[H,:] = (z.*k_max.^α .- δ.*k_max).^(-γ) # imposes a constraint

	  #throw out bad Value functions, good for debugging not for results
	  #if findmin(Vaf)[1]>0
	  	global v= V
	#end

	  # need push function to add to an already existing array
	  push!(dist, findmax(abs.(V_change))[1])
	  if dist[n].< ε
	      println("Value Function Converged Iteration=")
	      println(n)
	      break
	  end
	end

	push!(Value_functions, v)
	x = OU_process[T_obs+(t-1)*dt_inv+1:T_obs+(t)*dt_inv]
	y= OU_process[T_obs+(t-1)*dt_inv+2:T_obs+(t)*dt_inv+1]
	ϵ_OU = ε_OU[T_obs+(t-1)*dt_inv+1:T_obs+(t)*dt_inv]

	global ϕ_g = [const_g; 1.0-θ_g*dt]# prediction
	# trues values are ϕ = [0.0; 1-θ; σ]
	W = [ones(dt_inv,1) x]

	#Better to loop or treat as a vector?

	# This seems to converge more quickly, less noise
	global R_g = R_g + 0.01.*(W'*W*dt - R_g)
	global ϕ_g = ϕ_g + 0.01.*R_g^(-1)*W'*(y-W*ϕ_g)*dt

	# Slower?
	#=for j in 1:dt_inv
		global R_g = R_g + (0.01) .*(W[j,:]*W[j,:]' - R_g)
		global ϕ_g = ϕ_g + (0.01) .*R_g^(-1)*W[j,:]*(y[j,:][1]-ϕ_g'*W[j,:])
	end=#

	global const_g = ϕ_g[1]
	global θ_g = (1.0-ϕ_g[2])*dt_inv
	global σ_g =σ

	println("loop:$(t)")
end


plot(guesses_θ[:], label="Estimates",
title="\$ \\textrm{Estimate of } \\theta \\textrm{ over time}\$", legend=:bottomright)
plot!(θ.*ones(T,1), label="True value", legend=:topright)
png("Theta_estimates")

plot(guesses_const[:], label="Estimates",
title="\$ \\textrm{Estimate of the constant}\\textrm{ over time}\$", legend=:bottomright)
plot!(zeros(T,1), label="True value", legend=:topright)
png("const_estimates")

plot(guesses_σ[:], label="Estimates",
title="\$ \\textrm{Estimate of } \\sigma \\textrm{ over time}\$", legend=:bottomright)
plot!(σ.*ones(T,1), label="True value")
png("sigma_estimates")

all_drifts=(-guesses_θ[:].*log.(z) .+(guesses_σ[:].^2)/2 ).*z

plot(all_drifts[:,20], label="Estimates",title="Drift for Median Z", legend=:bottomright)
plot!(μ[20]*ones(T,1), label="True value")
png("drift_estimates_median")

plot(all_drifts[:,1], label="Estimates",title="Drift for Lowest Z", legend=:bottomright)
plot!(μ[1]*ones(T,1), label="True value")
png("drift_estimates_low")

plot(all_drifts[:,end], label="Estimates",title="Drift for Highest Z", legend=:bottomright)
plot!(μ[end]*ones(T,1), label="True value")
png("drift_estimates_high")

plot(Value_functions[1][:,20], label="Period 1",
	title="Value Functions For median Z", grid=false,
	legend=:bottomright, xlabel="k", line=:dashdot)
plot!(Value_functions[100][:,20], label="Period 100", line=:dot)
plot!(Value_functions[500][:,20], label="Period 500",line=:dashdotdot)
plot!(Value_functions[end][:,20], label="Period 10,000", color=:orange)
plot!(v1[:,20], label="True Value", line=:dash, color=:black)
png("Value_med_z")


plot(Value_functions[1][50,:], label="Period 1",
title="Value Functions For median K", grid=false,
 legend=:bottomright, xlabel="z",line=:dashdot)
plot!(Value_functions[200][50,:], label="Period 200",line=:dot)
plot!(Value_functions[5000][50,:], label="Period 5,000", color=:purple, line=:dashdotdot)
plot!(Value_functions[end][50,:], label="Period 10,000", color=:orange,linewidth = 1.25)
plot!(v1[50,:], label="True Value", line=:dash, color=:black)
png("Value_med_k")

#Simplistic Mean dynamics for θ
mean_θ = zeros(T-1,1)

for h in 1:T-1
	mean_θ[h] = mean(guesses_θ[1:h])
end

plot(mean_θ[:], label="Mean Estimates",
title="\$ \\textrm{Mean of } \\theta \\textrm{ Estimates Observed Over Time}\$", legend=:bottomright)
plot!(θ.*ones(T,1), label="True value", legend=:topright)
png("Theta_means")


#Simplistic Mean dynamics for const
mean_const = zeros(T-1,1)

for h in 1:T-1
	mean_const[h] = sum(guesses_const[1:h])*(1/h)
end

plot(mean_const[:], label="Mean Estimates",
title="\$ \\textrm{Mean of Constant Estimates Observed Over Time}\$", legend=:bottomright)
plot!(zeros(T,1), label="True value", legend=:topright)
png("const_means")

# Compare the actual process and the forecast
#=
OU_forecast = OrnsteinUhlenbeckProcess(θ_g, 0.0, σ_g, 0.0, 0.0)
OU_forecast.dt = dt

setup_next_step!(OU_forecast)
for j in 1:(dt_inv*T + (dt_inv-1)+T_obs)
    accept_step!(OU_forecast,dt)
end

# Now plot the process
plot(OU_process.t,OU_forecast.u, grid=false,
		label="Forecast", color=:blue,
		title="Ornstein-Uhlenbeck Process for z", legend=:bottomright)
plot!(OU_process.t,OU_process.u, label="Actual Process",
		color=:black)
png("OU_forecast")
=#
