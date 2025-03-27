import subprocess

# List of configurations to run
vbe_train_file = "vbe_train.py"
lmc_train_file = "lmcts_train.py"

num_runs = 5
num_bandits = int(1e4)
num_iterations = 2000
top_m = 1
lmc_num_iters = 50
ks_commands = [f"python {vbe_train_file} --data_dim 5 --data_power 5 --top_m {top_m} --iterations {num_iterations} --num_bandits {num_bandits} --learning_rate .01 --var_post ks --run {i} --seed {5*i}" for i in range(num_runs)]
bt_commands = [f"python {vbe_train_file} --data_dim 5 --data_power 5 --top_m {top_m} --iterations {num_iterations} --num_bandits {num_bandits} --learning_rate .01 --var_post beta --run {i} --seed {5*i}" for i in range(num_runs)]
th_commands = [f"python {vbe_train_file} --data_dim 5 --data_power 5 --top_m {top_m} --iterations {num_iterations} --num_bandits {num_bandits} --learning_rate .01 --var_post tanh-normal --run {i} --seed {5*i}" for i in range(num_runs)]
co_commands = [f"python {vbe_train_file} --data_dim 5 --data_power 5 --top_m {top_m} --iterations {num_iterations} --num_bandits {num_bandits} --learning_rate .01 --var_post concrete --run {i} --seed {5*i}" for i in range(num_runs)]
lm_commands = [f"python {lmc_train_file} --data_dim 5 --data_power 5 --top_m {top_m} --iterations {num_iterations} --num_bandits {num_bandits} --learning_rate 0.0005 --beta_inv 0.0001 --inner_num_iters {lmc_num_iters} --run {i} --seed {5*i}" for i in range(num_runs)]

# uncomment the runs you want to run
commands = [
    #*ks_commands, 
    #*bt_commands, 
    #*th_commands
    *co_commands
    #*lm_commands
    ]

# Iterate over each command and execute it
for cmd in commands:
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
    else:
        print(f"Command succeeded: {cmd}")