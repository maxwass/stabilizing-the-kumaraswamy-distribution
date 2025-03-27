import subprocess
import concurrent.futures

# List of configurations to run
eval_file = "iwae_eval.py" # "eval.py"

# uncomment the evaluations to run
commands = [
    # mnist
    #f"python {eval_file} --dataset mnist --var_post gaussian --likelihood cb",
    #f"python {eval_file} --dataset mnist --var_post gaussian --likelihood beta",
    #f"python {eval_file} --dataset mnist --var_post gaussian --likelihood ks",
    #f"python {eval_file} --dataset mnist --var_post ks --likelihood cb",
    #f"python {eval_file} --dataset mnist --var_post ks --likelihood beta",
    #f"python {eval_file} --dataset mnist --var_post ks --likelihood ks",
    #f"python {eval_file} --dataset mnist --var_post beta --likelihood cb", # --var_link_func exp",
    #f"python {eval_file} --dataset mnist --var_post beta --likelihood beta", # --var_link_func softplus",
    #f"python {eval_file} --dataset mnist --var_post beta --likelihood ks", # --var_link_func softplus",
    # cifar10
    #f"python {eval_file} --dataset cifar10 --var_post gaussian --likelihood cb",
    f"python {eval_file} --dataset cifar10 --var_post gaussian --likelihood beta",
    f"python {eval_file} --dataset cifar10 --var_post gaussian --likelihood ks",
    f"python {eval_file} --dataset cifar10 --var_post ks --likelihood cb",
    f"python {eval_file} --dataset cifar10 --var_post ks --likelihood beta",
    f"python {eval_file} --dataset cifar10 --var_post ks --likelihood ks",
    f"python {eval_file} --dataset cifar10 --var_post beta --likelihood cb", #  --var_link_func exp",
    # no stable model runs for below
    #f"python {eval_file} --dataset cifar10 --var_post beta --likelihood beta",# --var_link_func exp",
    #f"python {eval_file} --dataset cifar10 --var_post beta --likelihood ks", # --var_link_func exp"
]


# Iterate over each command and execute it
for cmd in commands:
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}\n\n")
    else:
        print(f"Command succeeded: {cmd}\n\n")