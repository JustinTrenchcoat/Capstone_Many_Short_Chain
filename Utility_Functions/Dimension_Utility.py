def mem(msg):
    print(f"{msg}: {process.memory_info().rss / 1024**2:.1f} MB")

def Iso_builder(num_dim):
  # mean array:
  mu = jnp.full(num_dim, 0.0)
  # covariance matrix:
  cov = jnp.eye(num_dim)
  target = tfd.MultivariateNormalFullCovariance(
      loc=jnp.array(mu),
      covariance_matrix=cov)
  init_step_size = 0.5
  def target_log_prob_fn(x):
      return target.log_prob(x)
  def initialize(shape, key):
      return random.normal(key, shape + (num_dim,))

  mean_benchmark = target.mean()
  var_benchmark = target.variance()

  return target_log_prob_fn, initialize,init_step_size,mean_benchmark,var_benchmark


def kernel_setup(warmup_length,
                 num_total_chains, num_super_chains,
                 naive,initialize_fn, randomKey,
                 target_log_prob_fn,init_step_size):
    key, init_key = random.split(randomKey)
    
    if naive:
      initial_position = initialize_fn((num_total_chains,), init_key)
    else:
        num_sub_chains = num_total_chains//num_super_chains
        initial_position_super = initialize_fn((num_super_chains,), init_key)
        initial_position = jnp.repeat(initial_position_super,num_sub_chains,axis=0)

    warmup = blackjax.chees_adaptation(
        target_log_prob_fn,num_chains=num_total_chains,
        target_acceptance_rate=0.75)
    optimizer = optax.adam(learning_rate=0.001)
    key_warmup, key_sample = random.split(key)
    (last_states, parameters), _= warmup.run(
        key_warmup,
        initial_position,
        init_step_size,
        optimizer,
        warmup_length
        )
    sample_keys = random.split(key_sample, num_total_chains)
    kernel = blackjax.dhmc(target_log_prob_fn, **parameters).step
    return kernel, sample_keys, last_states


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
               naive,initialize_fn, randomKeys,
               target_log_prob_fn,init_step_size,
               repitition,R_hat_list,MSE_list,
               true_mean,true_var):
    result_mse = []
    for sim in range(repitition):
        kernel, sample_keys, last_states = kernel_setup(warmup_length,
                                                        num_total_chains, num_super_chains,
                                                        naive,initialize_fn, randomKeys[sim],
                                                        target_log_prob_fn,init_step_size)
        sample_states, info = jax.vmap(kernel)(sample_keys, last_states)
        samples = sample_states.position
        dims = samples.shape[1]
        result_mse, factored_sq_err = mse_calculation(samples,true_mean,true_var,result_mse)

        for dim in range(dims):
            rhat = nested_rhat_constrained(samples, num_super_chains, dim)
            R_hat_list.append({
                "Warmup Length": warmup_length,
                "Size": dims,
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
                        "Best MSE": mse_best,"Worst MSE": mse_worst,"Size":dims})
    if naive:
        print(f"Naive initialization. Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")
    else:
        print(f"Constrained initialization. Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")
    del result_mse
    gc.collect()
    jax.clear_caches()


def run_simulation(builder, dimension_list, warmup_length, repitition,
                   num_chains_short,num_super_chains,
                   Iso_RHat_c_list,Iso_MSE_c_list,
                   Iso_RHat_n_list,Iso_MSE_n_list):
    base_key = random.PRNGKey(0)
    keys = random.split(base_key, repitition)
    for num_dimension in dimension_list:
        mem(f"Simulation Start, D={num_dimension}")
        target_log_prob_fn, initialize,init_step_size,mean_benchmark,var_benchmark = builder(num_dimension)
        # for one demension setup:
        for length in warmup_length:
            simulation(length,num_chains_short, num_super_chains,
                   False,initialize, keys,
                   target_log_prob_fn,init_step_size,
                   repitition, Iso_RHat_c_list,Iso_MSE_c_list,
                   mean_benchmark,var_benchmark)
            simulation(length,num_chains_short, num_super_chains,
                   True,initialize, keys,
                   target_log_prob_fn,init_step_size,
                   repitition, Iso_RHat_n_list,Iso_MSE_n_list,
                   mean_benchmark,var_benchmark)
#======================================
# Plotting part
#======================================