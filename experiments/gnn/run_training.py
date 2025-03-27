import subprocess

# List of configurations to run
edge_latent_train_file = "train_edge_latent.py"
point_estimate_train_file = "train_gae.py"


num_runs = 5
datasets = ["Cora", "Citeseer", "Pubmed"]
for dataset in datasets:
    #dataset = "Pubmed"
    ks_commands = [f"python {edge_latent_train_file} --dataset {dataset} --latent_distrib ks --beta .05 --run {i} --seed {5*i}" for i in range(num_runs)]
    bt_commands = [f"python {edge_latent_train_file} --dataset {dataset} --latent_distrib beta --beta .05  --run {i} --seed {5*i}" for i in range(num_runs)]
    th_commands = [f"python {edge_latent_train_file} --dataset {dataset} --latent_distrib tanh-normal --beta .05 --run {i} --seed {5*i}" for i in range(num_runs)]
    pt_commands = [f"python {point_estimate_train_file} --dataset {dataset} --run {i} --seed {5*i}" for i in range(num_runs)]

    commands = [
        #*ks_commands, 
        #*th_commands
        #*bt_commands, 
        *pt_commands
        ]

    # Iterate over each command and execute it
    for cmd in commands:
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
        else:
            print(f"Command succeeded: {cmd}")