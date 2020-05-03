# Dockerfile
FROM python:3.6.2

MAINTAINER pavel@novapulsar.com

ENV CONDA_ENV_PATH /opt/miniconda
ENV MY_CONDA_PY3ENV "sle-env"
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"
ENV \
	MAIN_PATH="/var/www" \
	FLASK_APP="sle" \
	DOLLAR="$"
	

ADD . $MAIN_PATH

RUN sed -i '/jessie-updates/d' /etc/apt/sources.list 

RUN apt-get update --yes --force-yes

RUN apt-get install --yes --force-yes \
	libgl1-mesa-glx \
	debconf \
        #python-qt4 \
	#python-pip \
	#python-dev \
        nginx \
        cron \
        htop \
   	gettext-base \
	locales \
	uuid-dev \
	libcap-dev \
	libpcre3-dev \
	build-essential \
	software-properties-common \
	python2.7 \
	python-pip \
	#uwsgi \
	#uwsgi-plugin-python3 \
	#&& add-apt-repository ppa:ubuntu-toolchain-r/test \
	#&& apt-get install gcc-snapshot \
        #&& apt-get install gcc-6 g++-6 -y \
        #&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 \
        && apt-get clean

#RUN locale-gen en_US.UTF-8
#RUN dpkg-reconfigure locales

RUN chmod +x $MAIN_PATH/bin/*.sh && mkdir -p /var/run
	
RUN pip2 install 'supervisor>=3.3.0'

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -bfp $CONDA_ENV_PATH \
    && rm ~/miniconda.sh \
    && chmod -R a+rx $CONDA_ENV_PATH

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR $MAIN_PATH

RUN conda update --quiet --yes conda
RUN conda create -y -n $MY_CONDA_PY3ENV python=3.6.2

#RUN dpkg-reconfigure python

COPY src/requirements $MAIN_PATH/src/requirements

RUN conda install --yes -n $MY_CONDA_PY3ENV -c conda-forge --file $MAIN_PATH/src/requirements/conda-forge.txt \
    && conda install --yes -n $MY_CONDA_PY3ENV -c conda-forge --file $MAIN_PATH/src/requirements/flask.txt

RUN bash -c '$CONDA_ACTIVATE && pip install -r $MAIN_PATH/src/requirements/pip-requirements.txt'

#RUN pip install git+https://github.com/Supervisor/supervisor@master

COPY deployment/configs/supervisord.conf /etc/supervisord.conf

#RUN bash -c 'cd /usr/bin && ./uwsgi --build-plugin "/usr/lib/uwsgi/plugins/python3 python36"' \
#    && mv python36_plugin.so /usr/lib/uwsgi/plugins/python36_plugin.so

RUN mkdir -p /var/log/supervisor

EXPOSE 80

WORKDIR $MAIN_PATH

COPY deployment/configs/nginx/ /etc/nginx/

CMD supervisord -c /etc/supervisord.conf
