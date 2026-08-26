def sample_setup(warmup_length,dimension,
                    num_total_chains, num_super_chains,
                    initialize_fn, randomKey,
                    target_log_prob_fn,init_step_size):
    adapt_key, init_key = random.split(randomKey)
    initial_position = initialize_fn((dimension,), init_key)
    adapt = blackjax.pathfinder_adaptation(
        blackjax.hmc,
        target_log_prob_fn,
        num_chains = num_total_chains, # question: can we make it less?
        n_paths = 20, # 20 in the Zhang et al.
        initial_step_size = init_step_size,
        num_integration_steps = 1,
        adaptation_info_fn=get_filter_adapt_info_fn()
        )
    (last_states, parameters), _ = adapt.run(
        adapt_key,
        initial_position,
        warmup_length
        )

    sample_keys = random.split(adapt_key, num_total_chains)
    def run_chain(key, state,step_size):
        kernel = blackjax.hmc(target_log_prob_fn,step_size = step_size,
                              inverse_mass_matrix = parameters["inverse_mass_matrix"],
                              num_integration_steps = parameters["num_integration_steps"])
        return kernel.step(key, state)
    
    sample_states, infos = jax.vmap(run_chain, in_axes=(0,0,0))(
        sample_keys,last_states,parameters['step_size'])
    del infos

    return sample_states.position
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

def simulation(warmup_length,num_total_chains, num_super_chains,
               initialize_fn, randomKeys,
               target_log_prob_fn,init_step_size,
               repitition,R_hat_list,MSE_list,
               true_mean,true_var, dimension):
    result_mse = []
    for sim in range(repitition):
        samples = sample_setup(warmup_length, dimension, 
                               num_total_chains, num_super_chains,
                               initialize_fn, randomKeys[sim],
                               target_log_prob_fn, init_step_size)
        dims = samples.shape[1]
        result_mse, factored_sq_err = mse_calculation(samples,true_mean,true_var,result_mse)

        for dim in range(dims):
            rhat = nested_rhat_constrained(samples, num_super_chains, dim)
            R_hat_list.append({
                "Warmup Length": warmup_length,
                "Iteration":sim,
                "Dimension": dim,
                "Rhat": rhat[-1],
                "MSE":factored_sq_err[dim]})
        del samples
        jax.clear_caches()
        gc.collect()
    mse_list = np.array(result_mse)
    mse_best = mse_list.min(axis=0)
    mse_worst = mse_list.max(axis=0)
    avg_mse = mse_list.mean(axis=0)
    MSE_list.append({"Warmup Length": warmup_length,"Avg MSE": avg_mse,
                     "Best MSE": mse_best,"Worst MSE": mse_worst})
    
    print(f"Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")
    del result_mse
    gc.collect()
    jax.clear_caches()
