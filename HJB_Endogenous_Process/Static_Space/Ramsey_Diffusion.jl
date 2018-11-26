#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an Ramsey Model with a diffusion process for capital

       This version of the code updates the grid for capital
        along with estimates for the unknown parameter

	Algorithm code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm
==============================================================================#

using Parameters, Distributions, Plots

@with_kw type Model_parameters
    γ= 2.0 #gamma parameter for CRRA utility
    ρ = 0.05 #the discount rate
    α = 1/3 # the curvature of the production function (cobb-douglas)
    δ = 0.05 # the depreciation rate
    σ = 2.0 #variance term
end

param = Model_parameters()
@unpack_Model_parameters(param)

#=============================================================================
 	k our capital follows a process that depends on
            dk = (s⋅f(k) - (δ-σ^2)k)dt - σ⋅k⋅dW_[t]
        Where W is a Wiener process
=============================================================================#
#K_starting point, for mean of z process

k_st = ((α)/(ρ+δ+σ^2))^(1/(1-α))

# create the grid for k
I = 1000 #number of points on grid
k_min = 0.1*k_st # min value
k_max = 10.*k_st # max value
k = linspace(k_min, k_max, I)
dk = (k_max-k_min)/(I-1)

#create matrices for k and z
kk = k*1

# use Ito's lemma to find the drift and variance of our optimization equation

max_it = 100
ε = 0.1^(6)
Δ = 1000

# set up all of these empty matrices
Vaf_1, Vab_1, c_1 = [zeros(I,1) for i in 1:3]

y = pdf.(Normal(.99, σ), k)
plot(k,y, grid=false,
		xlabel="k", ylabel="Probability",
		legend=false, color="purple", title="PDF of k")
png("PDF of k")

#============================================================================#
# Now it's time to solve the model, first put in a guess for the value function
v0 = (kk.^α).^(1-γ)/(1-γ)/ρ
v_1=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

for n = 1:maxit
    V=v_1

    #Now set up the forward difference

    Vaf_1[1:I-1,:] = (V[2:I, :] - V[1:I-1,:])/dk
    Vaf_1[I,:] = (k_max.^α - (δ+σ^2).*k_max).^(-γ) # imposes a constraint

    #backward difference
    Vab_1[2:I,:] = (V[2:I, :] - V[1:I-1,:])/dk
    Vab_1[1,:] = (k_min.^α - (δ+σ^2).*k_min).^(-γ)

    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf_1.^(-1/γ)
    drf =(1-(cf/kk.^α))*kk.^α-(δ+σ^2).*kk

    # consumption and saving backwards difference
    cb = Vab_1.^(-1.0/γ)
    drb = (1-(cb/k.^α))*kk.^α-(δ+σ^2).*kk
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = kk.^α - (δ+σ^2).*kk
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    If = drf.>0 # positive drift will ⇒ forward difference
    Ib = drb.<0 # negative drift ⇒ backward difference
    I0=(1-If-Ib) # at steady state

    Va_upwind = Vaf_1.*If + Vab_1.*Ib + Va0.*I0 # need to include SS term

    c_1 = Va_upwind.^(-1/γ)
    u = (c_1.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min(drb, 0)/dk + (σ.*kk/(2*dk))
    Y = -max(drf, 0)/dk + min(drb, 0)/dk - (σ.*kk/(dk))
    Z = max(drf, 0)/dk + (σ.*kk/(2*dk))

    updiag = 0
    updiag =[updiag; Z[1:I-1,1]; 0]
    updiag =(updiag[:])

    centerdiag=reshape(Y, I, 1)
    centerdiag = (centerdiag[:]) # for tuples

   lowdiag = X[2:I, 1]
   lowdiag=(lowdiag)

   # spdiags in Matlab allows for automatic trimming/adding of zeros
       # spdiagm does not do this
  A = spdiagm(centerdiag)+ [zeros(1, I); spdiagm(lowdiag) zeros(I-1,1)] + spdiagm(updiag)[2:end, 1:I] # trim first element

  B = (1/Δ + ρ)*speye(I) - A

  u_stacked= reshape(u, I, 1)
  V_stacked = reshape(V,I, 1)

  b = u_stacked + (V_stacked./Δ)

  V_stacked = B\b

  V = reshape(V_stacked, I, 1)

  V_change = V-v_1

  v_1= V
  # need push function to add to an already existing array
  push!(dist, findmax(abs(V_change))[1])
  if dist[n].< ε
      println("Value Function Converged Iteration=")
      println(n)
      break
  end

end

# calculate the savings for kk
ss_1 = kk.^α - (δ+σ^2).*kk

# Plot the savings vs. k
plot(k, c_1, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Consumption Policies")
plot!(k, zeros(I,1), line=:dash, color=:black)
png("OptimalCons")


plot(k, ss_1, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Savings Policies")
plot!(k, zeros(I,1), line=:dash, color=:black)
png("OptimalSavings")


# Plot the interest rate
r_1 = α.*kk.^(α-1)
plot(k, r_1, grid=false,
		xlabel="k", ylabel="r(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Interest Rate")
png("OptimalInterest")

plot(k, v_1, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max),
		legend=false, title="")
png("Value_function_vs_k")

#==================== Now resolve the model with σ misspecified =======#

# create a random distribution to pull from
Dist = Bernoulli(.45)
time = 100 # set the number of external time periods to zero

# Add in the misspecification
σ_g =5.0
Σ_g=[σ_g]
# set up all of these empty matrices
Vaf_2, Vab_2, c_2 = [zeros(I,1) for i in 1:3]
Val_2 =[]
savings=[]

maxit= 30 #set number of iterations (only need 6 to converge)
for i in 1:time
    # Now it's time to solve the model, first put in a guess for the value function
    v0 = (kk.^α).^(1-γ)/(1-γ)/ρ
    v_2=v0
    dist = [] # needs to be in the time loop to work properly
    indicator = rand(Dist)
    for n = 1:maxit
        V=v_2

        #Now set up the forward difference

        Vaf_2[1:I-1,:] = (V[2:I, :] - V[1:I-1,:])/dk
        Vaf_2[I,:] = (k_max.^α - (δ+σ_g^2).*k_max).^(-γ) # imposes a constraint

        #backward difference
        Vab_2[2:I,:] = (V[2:I, :] - V[1:I-1,:])/dk
        Vab_2[1,:] = (k_min.^α - (δ+σ_g^2).*k_min).^(-γ)

        #I_concave = Vab .> Vaf # indicator for whether the value function is concave

        # Consumption and savings functions
        cf = Vaf_2.^(-1/γ)
        drf =(1-(cf/kk.^α))*kk.^α-(δ+σ_g^2).*kk

        # consumption and saving backwards difference
        cb = Vab_1.^(-1.0/γ)
        drb = (1-(cb/k.^α))*kk.^α-(δ+σ_g^2).*kk
        #println(sb)
        #consumption and derivative of the value function at the steady state

        c0 = kk.^α - (δ+σ_g^2).*kk
        Va0 = c0.^(-γ)

        # df chooses between the forward or backward difference

        If = drf.>0 # positive drift will ⇒ forward difference
        Ib = drb.<0 # negative drift ⇒ backward difference
        I0=(1-If-Ib) # at steady state

        Va_upwind = Vaf_1.*If + Vab_1.*Ib + Va0.*I0 # need to include SS term

        c_2 = Va_upwind.^(-1/γ)
        u = (c_1.^(1-γ))/(1-γ)

        # Now to constuct the A matrix
        X = -min(drb, 0)/dk + (σ_g.*kk/(2*dk))
        Y = -max(drf, 0)/dk + min(drb, 0)/dk - (σ_g.*kk/(dk))
        Z = max(drf, 0)/dk + (σ_g.*kk/(2*dk))

        updiag = 0
        updiag =[updiag; Z[1:I-1,1]; 0]
        updiag =(updiag[:])

        centerdiag=reshape(Y, I, 1)
        centerdiag = (centerdiag[:]) # for tuples

       lowdiag = X[2:I, 1]
       lowdiag=(lowdiag)

       # spdiags in Matlab allows for automatic trimming/adding of zeros
           # spdiagm does not do this
      A = spdiagm(centerdiag)+ [zeros(1, I); spdiagm(lowdiag) zeros(I-1,1)] + spdiagm(updiag)[2:end, 1:I] # trim first element

      B = (1/Δ + ρ)*speye(I) - A

      u_stacked= reshape(u, I, 1)
      V_stacked = reshape(V,I, 1)

      b = u_stacked + (V_stacked./Δ)

      V_stacked = B\b

      V = reshape(V_stacked, I, 1)

      V_change = V-v_2

      v_2= V
      # need push function to add to an already existing array
      push!(dist, findmax(abs(V_change))[1])
      if dist[n].< ε
        println(n)
        push!(Val_2, v_2)
        break
      end
    end
    push!(Σ_g, σ_g)
    # calculate the savings for kk
    ss_2 = kk.^α - (δ+σ_g^2).*kk
    push!(savings, ss_2)
    # add in updating
    if indicator ==1
        σ_g = σ_g + .001(σ-σ_g)
    end
end


# Plot the savings vs. k
plot(k, c_2, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),
		label="Guess", title="Optimal Consumption Policies with")
plot!(k,c_1, label="Actual")
plot!(k, zeros(I,1), line=:dash, color=:black)
png("OptimalCons")

ss_2 = kk.^α - (δ+σ_g^2).*kk
plot(k, ss_2, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),label="Guess", title="Optimal Savings Policies")
plot!(k,ss_1, label="Actual")
plot!(k, zeros(I,1), line=:dash, color=:black, label="")
png("OptimalSavings_2")


plot(k, savings, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),label="", title="Optimal Savings Policies")
plot!(k, zeros(I,1), line=:dash, color=:black, label="", legend=:bottomleft)
plot!(k,ss_1, label="Actual")
png("OptimalSavings_All")


plot(k, v_1, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max), title="Value Functions",
        label="known sigma", color=:black, line=:dash,
        legend=:bottomright)
plot!(k,Val_2[1], label="guess, time=1", color=:blue)
plot!(k,Val_2[100], label="guess, time=100", color=:green)
plot!(k,Val_2[50], label="guess, time=50", color=:purple, line=:dot)
png("Value_function_vs_k_2")

plot(k, Val_2, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max), title="Value Functions", legend=false)
png("Value_function_vs_k_all")
