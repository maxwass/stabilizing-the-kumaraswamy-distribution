import subprocess

# List of configurations to run
train_file = "train.py"
commands = [
    # mnist
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post gaussian --likelihood cb",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post gaussian --likelihood beta",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post gaussian --likelihood ks",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post ks --likelihood cb",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post ks --likelihood beta",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post ks --likelihood ks",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post beta --likelihood cb --var_link_func exp",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post beta --likelihood beta --var_link_func softplus",
    f"python {train_file} --dataset mnist --latent_dimension 20 --var_post beta --likelihood ks --var_link_func softplus",
    # cifar10
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post gaussian --likelihood cb",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post gaussian --likelihood beta",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post gaussian --likelihood ks",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post ks --likelihood cb",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post ks --likelihood beta",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post ks --likelihood ks"
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post beta --likelihood cb --var_link_func exp",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post beta --likelihood beta --var_link_func exp",
    f"python {train_file} --dataset cifar10 --latent_dimension 50 --var_post beta --likelihood ks --var_link_func exp"
]


# Iterate over each command and execute it
for cmd in commands:
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
    else:
        print(f"Command succeeded: {cmd}")