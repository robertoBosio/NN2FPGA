FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG NN2FPGA_ROOT_DIR

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 vim curl wget git build-essential \
    zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev liblzma-dev libncurses5-dev zip time udev \
    libudev-dev python3 python3-pip python3-venv cmake libgtk2.0-dev \
    pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    locales libncurses5 libtinfo5 libncursesw5-dev unzip \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=en_US.UTF-8

# Add user (no home)
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --no-create-home --no-log-init ${USERNAME}

# Set workspace and environment variables
WORKDIR ${NN2FPGA_ROOT_DIR}
ENV PATH=${PATH}:${HOME}/.local/bin
ENV HISTFILE=${NN2FPGA_ROOT_DIR}/.docker_bash_history
ENV PROMPT_DIRTRIM=1
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Copy and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to user
USER ${USERNAME}
ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
