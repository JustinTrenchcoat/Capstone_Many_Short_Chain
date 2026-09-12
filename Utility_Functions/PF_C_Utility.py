def initialization_setup(dimension, num_total_chains,num_super_chains,
                         target_log_prob_fn, initialize_fn,
                         randomKey, init_step_size, num_path):
    adapt_key, init_key, warmup_key = random.split(randomKey,3)
    initial_position = initialize_fn((dimension,), init_key)
    # let it return num_super_chains initial positions.
    adapt = blackjax.pathfinder_adaptation(
            blackjax.hmc,
            target_log_prob_fn,
            num_chains = num_super_chains, # question: can we make it less?
            n_paths = num_path, # To prevent OOM issue
            initial_step_size = init_step_size,
            num_integration_steps = 1,
            adaptation_info_fn=get_filter_adapt_info_fn()
            )
    (last_states, parameters), _ = adapt.run(adapt_key, initial_position,0)
        # last_states would be just the initialized positions:
        # print(f"last states are : {last_states[0]}") 
    num_sub_chains = num_total_chains//num_super_chains
    initial_position_super = last_states[0]
    
        # second step: based on the initialized positions we make all subchains
    true_initial_position = jnp.repeat(initial_position_super,num_sub_chains,axis=0)
    true_initial_position.block_until_ready()
    del adapt, last_states, parameters,initial_position,initial_position_super
    gc.collect()
    jax.clear_caches()
    return true_initial_position, warmup_key

def kernel_setup(warmup_length,dimension,
                    num_total_chains, num_super_chains,
                    initialize_fn, randomKey,
                    target_log_prob_fn,init_step_size):
    # 20 in the Zhang et al. Temporary change to 5 when doing IRT or other model with large dimension.
    if dimension >= 200:
        num_path = 5
    else:
        num_path = 20
    # this would be done in two steps:
    # first step: let adaptation just return the initializatin positions:
    true_initial_position, warmup_key = initialization_setup(dimension, num_total_chains,
                                                             num_super_chains,target_log_prob_fn,
                                                             initialize_fn,randomKey,init_step_size, num_path)
    warmup = blackjax.chees_adaptation(
        target_log_prob_fn,num_chains=num_total_chains,
        target_acceptance_rate=0.75)
    optimizer = optax.adam(learning_rate=0.001)
    key_warmup, key_sample = random.split(warmup_key)
    (warmup_last_states, warmup_parameters), _= warmup.run(
        key_warmup,
        true_initial_position,
        init_step_size,
        optimizer,
        warmup_length
        )
    sample_keys = random.split(key_sample, num_total_chains)
    kernel = blackjax.dhmc(target_log_prob_fn, **warmup_parameters).step
    del warmup_parameters, optimizer,warmup
    gc.collect()

    return kernel, sample_keys, warmup_last_states


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
               initialize_fn, randomKeys,dimension,
               target_log_prob_fn,init_step_size,
               repitition,R_hat_list,MSE_list,
               true_mean,true_var):
    result_mse = []
    for sim in range(repitition):
        kernel, sample_keys, last_states = kernel_setup(warmup_length,dimension,
                                                        num_total_chains, num_super_chains,
                                                        initialize_fn, randomKeys[sim],
                                                        target_log_prob_fn,init_step_size)
        sample_states, info = jax.vmap(kernel)(sample_keys, last_states)
        samples = sample_states.position
        # dims = samples.shape[1]
        result_mse, factored_sq_err = mse_calculation(samples,true_mean,true_var,result_mse)

        for dim in range(dimension):
            rhat = nested_rhat_constrained(samples, num_super_chains, dim)
            R_hat_list.append({
                "Warmup Length": warmup_length,
                "Iteration":sim,
                "Dimension": dim,
                "Rhat": rhat[-1],
                "MSE":factored_sq_err[dim]})
        del kernel, sample_keys, last_states, sample_states, info, samples
        gc.collect()
    mse_list = np.array(result_mse)
    mse_best = mse_list.min(axis=0)
    mse_worst = mse_list.max(axis=0)
    avg_mse = mse_list.mean(axis=0)
    MSE_list.append({"Warmup Length": warmup_length,"Avg MSE": avg_mse,
                        "Best MSE": mse_best,"Worst MSE": mse_worst})
    print(f"New Pathfinder Initialization. Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")

    del result_mse
    gc.collect()
    jax.clear_caches()
