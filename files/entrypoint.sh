#!/bin/sh

# supervisor keeps Jupyter notebook running
service supervisor start

# keep running as a service
/bin/bash -c "trap : TERM INT; sleep infinity & wait"
