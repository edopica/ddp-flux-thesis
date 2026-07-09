#!/bin/bash
set -e

if [ -z "$2" ]; then
    echo "Usage: $0 <remote_host> <command...>"
    echo "Example: $0 hpc make check"
    exit 1
fi

REMOTE_HOST=$1
shift
COMMAND="$@"
THESIS_DIR="~/projects/ddp-flux-thesis"

# If the remote host is 'hpc', we wrap the command with srun
if [ "$REMOTE_HOST" = "hpc" ]; then
    # Using salloc so that it allocates resources and lets the command run on the login node 
    # but with SLURM environment variables, allowing inner srun commands to work correctly.
    WRAPPER="salloc --ntasks=4 --gres=gpu:0 --mem=32G --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud"
    FULL_COMMAND="$WRAPPER $COMMAND"
else
    FULL_COMMAND="$COMMAND"
fi

echo "Executing on $REMOTE_HOST: $FULL_COMMAND"

ssh "$REMOTE_HOST" "cd $THESIS_DIR && bash -l -c \"$FULL_COMMAND\""
