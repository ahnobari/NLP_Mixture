# take swap amount as argument
# usage: ./make_swap.sh 200G
# 200G is the swap size in GB

#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <swap_size>"
    exit 1
fi

SWAP_SIZE=$1

sudo fallocate -l ${SWAP_SIZE}G ./swap
sudo chmod 600 ./swap
sudo mkswap ./swap
sudo swapon ./swap