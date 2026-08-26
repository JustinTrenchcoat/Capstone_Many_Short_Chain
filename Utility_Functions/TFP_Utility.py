def kernel_setup(warmup_length, sampling_length,
                 num_sub_chains,num_super_chains,
                 naive, initialize_fn,ranKey,
                 target_log_prob_fn, init_step_size):
    num_warmup_short, num_sampling_short = warmup_length, sampling_length
    total_samples_short = num_warmup_short + num_sampling_short

    kernel_short = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn, init_step_size, 1)
    kernel_short = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(kernel_short, num_warmup_short)
    kernel_short = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel_short, num_warmup_short, target_accept_prob = 0.75,  #0.75,
        reduce_fn = tfp.math.reduce_log_harmonic_mean_exp)

    if (naive):
       # initialize each chain at a different location
        initial_state = initialize_fn((num_sub_chains,),key=ranKey)
        initial_state_super = initial_state
    else:
        # Chains within a super chain are all initialized at the same location
        initial_state_super = initialize_fn((num_super_chains,), key=ranKey)
        initial_state = jnp.repeat(initial_state_super, num_sub_chains // num_super_chains,axis = 0)

    return kernel_short, initial_state, total_samples_short, initial_state_super


def _reduce_variance_interval(x, axis=None, biased=True, keepdims=False):
    # ddof=0 is biased variance (N), ddof=1 is unbiased variance (N-1)
    ddof = 0 if biased else 1
    return jnp.var(x, axis=axis, ddof=ddof, keepdims=keepdims)


def nested_rhat_constrained(result_state, num_super_chains,idx):
     # since we use only N=1, W_k is reduced to 0
    num_sub_chains = result_state.shape[0] // num_super_chains
    num_dimensions = result_state.shape[1]

    chain_states = result_state.reshape(1, -1, num_sub_chains, num_dimensions)
    # chain_states.shape = (1,16,128,2), so it is (N,K,M,D).
    # f_bar 1*k, mean_subchain.shape =(1,16,2), (N,K,D)
    mean_subchain = jnp.mean(chain_states, axis=2)

    variance_chain = _reduce_variance_interval(chain_states, axis=2, biased=False)
    # print(variance_chain.shape) # (1,16,2)
    W = jnp.mean(variance_chain, axis=1)
    # print(f"W dim: {W.shape}") # (1,2)
    B = _reduce_variance_interval(mean_subchain, axis=1, biased=False) # variance of between super chain

    r_hat = jnp.sqrt(1+B/W)[:,idx]
    return r_hat

def mse_calculation(result, mean_benchmark,
                    var_benchmark, mse_list):
    mc_mean = result.mean(axis=0)
    squared_error = (mc_mean - mean_benchmark)**2
    factor = 1/var_benchmark
    factored_sq_err = factor*squared_error
    mse = (factored_sq_err).mean()
    mse_list.append(mse)
    return mse_list, factored_sq_err

# add a switch for recording the states
def simulation(keys, initialize_fn,
               warmup_length,sampling_length,
               naive, repitition,record_states,
               MSE_list, R_Hat_list,
               num_dim, state_list,
               mean_benchmark,var_benchmark,
               num_super_chains,num_sub_chains,
               target_log_prob_fn, init_step_size):
    result_mse = []
    # very long warmup phase might result in memory issue.
    if not record_states:
        skip_trace = True
    else:
        skip_trace = (warmup_length > 300)

    for sim in range(repitition):
        kernel_short, initial_state, total_samples_short, 
        initial_state_super = kernel_setup(warmup_length,sampling_length,
                                           num_sub_chains, num_super_chains,
                                           naive, initialize_fn,keys[sim],
                                           target_log_prob_fn, init_step_size)
    if not skip_trace:
        result = tfp.mcmc.sample_chain(
            total_samples_short, initial_state, kernel = kernel_short,
            seed =keys[sim], trace_fn=None)
        result_with_init = jnp.concatenate([initial_state[None, :, :], result],axis=0)
        result_short = result[-1,:,:]
        state_record = {
            "Warmup Length": warmup_length,
            "Iteration": sim,
            "States":np.asarray(result_with_init),
            "Initial Position":np.asarray(initial_state_super)
            }
    else:
        result = []
        result_with_init = []
        result_short = tfp.mcmc.sample_chain(
            total_samples_short, initial_state, kernel = kernel_short,
            seed =keys[sim], trace_fn=None)[-1,:,:]
        state_record = {}

    state_list.append(state_record)

    # # the f bar
    # mc_mean = result_short.mean(axis=0)
    # squared_error = (mc_mean - mean_benchmark)**2
    # # the factor was in fact 1/var
    # factor = 1/var_benchmark
    # # sq_err for all dimensions: a vector with dimension=num_dim
    # factored_sq_err = factor*squared_error
    # # avg over all dimension
    # mse = (factored_sq_err).mean()
    # result_mse.append(mse)

    result_mse, factored_sq_err = mse_calculation(result_short,mean_benchmark,var_benchmark,result_mse)

    for dim in range(num_dim):
        new_r_hat = nested_rhat_constrained(result_short, num_super_chains, dim)
        R_Hat_list.append({
            "Warmup Length": warmup_length,
            "Iteration":sim,
            "Dimension": dim,
            "Rhat": new_r_hat[-1],
            "MSE":factored_sq_err[dim]
            })
    del state_record, result_short, result, kernel_short, initial_state
    del initial_state_super, total_samples_short, result_with_init
    gc.collect()

    result_mse = np.array(result_mse)
    result_mse_best = result_mse.min(axis=0)
    result_mse_worst = result_mse.max(axis=0)
    mean_mse = result_mse.mean(axis=0)

    MSE_list.append({"Warmup Length":warmup_length,"Avg MSE": mean_mse,
                        "Best MSE": result_mse_best,"Worst MSE": result_mse_worst})
    if naive:
        print(f"Naive initialization. Warmup Length: {warmup_length}; mean of MSE is: {mean_mse}")
    else:
        print(f"Constrained initialization. Warmup Length: {warmup_length}; mean of MSE is: {mean_mse}")
    del result_mse
    gc.collect()
    jax.clear_caches()