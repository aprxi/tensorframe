#!/bin/sh

# supervisor keeps Jupyter notebook running
jupyter_home="/service/tfclient/.jupyter"
if [ ! -d "$jupyter_home" ];then
    mkdir -p "$jupyter_home"
fi
if [ ! -s "$jupyter_home/jupyter_notebook_config.py" ];then
    echo "c.NotebookApp.token = u''" >"$jupyter_home/jupyter_notebook_config.py"
fi
if [ ! -d "/service/tfclient/notebooks" ];then
    mkdir -p "/service/tfclient/notebooks"
fi

service supervisor start

# keep running as a service
/bin/bash -c "trap : TERM INT; sleep infinity & wait"
