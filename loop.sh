#!/bin/bash

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
        echo "** Trapped CTRL-C"
	exit 1
}

for f in $2/*; do
    if [ -f "$f" ]; then
        # Will not run if no directories are available
        # echo "$f"
	if [[ $f == *".mkv" ]]; then
		$1 $f
		# echo "$f"
	fi
	sleep 1
    fi
done

