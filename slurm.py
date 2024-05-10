import argparse
from itertools import product
import json
import os, shutil

import pandas as pd
from git import Repo

from src import utils

def create_slurm_script(slurm_config):
    slurm_script = os.path.join(slurm_config.run_dir, 'run.slurm')

    with open(slurm_script, 'w') as f:
        f.write("#!/bin/bash\n")

        # Set job name the same as the experiment directory
        f.write("#SBATCH --job-name=%s\n" % slurm_config.exp_name)

        # Set email notification
        f.write("#SBATCH --mail-type=FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % slurm_config.email)

        # Set account
        f.write("#SBATCH --account=%s\n" % slurm_config.account)

        # Set partition
        f.write("#SBATCH --partition=%s\n" % slurm_config.partition)

        # Set number of nodes
        f.write("#SBATCH --nodes=%d\n" % slurm_config.nodes)

        # Set number of GPUs
        f.write("#SBATCH --gres=gpu:%d\n" % slurm_config.gpus)

        # Set number of tasks -- this should be equal to the number of GPUs
        f.write("#SBATCH --ntasks-per-node=%d\n" % slurm_config.gpus)

        # Set number of CPUs
        f.write("#SBATCH --cpus-per-task=%d\n" % slurm_config.cpus_per_task)

        # Set memory
        f.write("#SBATCH --mem=%s\n" % slurm_config.mem)

        # Set time limit
        f.write("#SBATCH --time=%s\n" % slurm_config.time)

        # Set exclude nodes
        f.write("#SBATCH --exclude=%s\n" % slurm_config.exclude)

        # Set constraint
        f.write("#SBATCH --constraint=%s\n" % slurm_config.constraint)

        # Set working directory
        f.write("#SBATCH --chdir=%s\n" % slurm_config.work_dir)

        # Export environment variables
        f.write("#SBATCH --export=ALL\n")

        # Set output and error files
        f.write("#SBATCH --output=%s\n" % os.path.join(
            slurm_config.run_dir, 'out.log'))
        f.write("#SBATCH --error=%s\n" % os.path.join(
            slurm_config.run_dir, 'out.log'))

        # Log commands
        f.write("set -x\n")

        # Log GPU info
        f.write("nvidia-smi\n")

        # Activate conda environment
        f.write(". %s\n" % slurm_config.conda_path)
        f.write("conda activate %s\n" % slurm_config.conda_env)

        # Set NCCL_DEBUG and PYTHONFAULTHANDLER
        f.write("export NCCL_DEBUG=INFO\n")
        f.write("export PYTHONFAULTHANDLER=1\n")

        # Clone and checkout git commit to experiment directory
        working_dir = os.path.join(slurm_config.work_dir, slurm_config.exp_name)
        f.write("if [ -d %s ]; then chmod -R +w %s; rm -rf %s; fi\n" % (
            working_dir, working_dir, working_dir))
        f.write("git clone %s %s\n" % (slurm_config.git_repo, working_dir))
        f.write("cd %s\n" % working_dir)
        f.write("git checkout %s\n" % slurm_config.commit_hash)

        # Untar dataset to `slurm_config.work_dir/data`
        for path in slurm_config.dataset_paths:
            f.write("tar -xvf %s -C %s\n" % (path, os.path.join(working_dir, 'data')))

        # Move local files
        for local_file in slurm_config.local_files:
            if os.path.exists(local_file):
                dest_dir = os.path.dirname(os.path.join(working_dir, local_file))
                f.write("mkdir -p %s\n" % dest_dir)
                f.write("cp -R %s %s\n" % (os.path.abspath(local_file), dest_dir))
            else:
                print("Warning: local file %s does not exist." % local_file)
        
        # Train command
        f.write("srun python -m src.trainer --config %s --run_dir %s --resume\n" %
                (os.path.join(slurm_config.run_dir, 'config.json'),
                 slurm_config.run_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_config", required=True, help="Path to the slurm configuration file")
    parser.add_argument("--exp_config", required=True, help="Path to the experiments configuration file")
    parser.add_argument("--run_dir", required=True, help="Path to the run directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generates configuration files and slurm "
                        "scripts without submitting jobs")
    args = parser.parse_args()

    # Assert no local changes and get the current commit hash
    repo = Repo(os.path.dirname(os.path.realpath(__file__)))
    branch = repo.active_branch
    if not args.dry_run:
        assert not repo.is_dirty(), "There are unstaged changes in the repository."
        assert len(list(repo.iter_commits('%s@{u}..%s' % (branch.name, branch.name)))) == 0, \
            "There are unpushed commits in the repository."
    commit_hash = repo.head.object.hexsha

    # Load slurm configuration
    slurm_config = utils.Params(args.slurm_config)

    # Create run_dir and config
    os.makedirs(args.run_dir)
    slurm_config.run_dir = os.path.abspath(args.run_dir)
    slurm_config.exp_name = os.path.basename(args.run_dir)
    shutil.copy(args.exp_config, os.path.join(args.run_dir, 'config.json'))

    # Write slurm script
    slurm_config.commit_hash = commit_hash
    slurm_config.git_repo = repo.remotes.origin.url
    create_slurm_script(slurm_config)
    print("Wrote slurm script to %s" % os.path.join(slurm_config.run_dir , 'run.slurm'))

    # Launch experiment
    if not args.dry_run:
        os.system("sbatch %s" % os.path.join(slurm_config.run_dir, 'run.slurm'))
        print("Launched experiment %s" % slurm_config.exp_name)
