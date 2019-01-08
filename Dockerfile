FROM nvidia/cuda

RUN apt-get update --fix-missing

RUN apt-get -y install python3-pip
RUN python3 -m pip install numpy pandas pyarrow
RUN python3 -m pip install jupyter

RUN useradd -u 2000 -U -d /home/tensorframe -m tensorframe -s /usr/sbin/nologin
COPY files/jupyter_notebook_config.py /home/tensorframe/.jupyter/jupyter_notebook_config.py
RUN mkdir -p /home/tensorframe/notebooks
RUN chown -R tensorframe:tensorframe /home/tensorframe

RUN mkdir -p /var/log/jupyter
RUN chown tensorframe:root /var/log/jupyter
COPY files/supervisor_jupyter /etc/supervisor/conf.d/jupyter.conf

RUN apt-get install -y supervisor
RUN service supervisor start

COPY files/entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
