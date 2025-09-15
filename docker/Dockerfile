FROM vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609

# 设置为非交互模式
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    tmux \
    zsh \
    rclone \
    htop \
    net-tools \
    iproute2 \
    iperf3 \
    traceroute \
    openssh-server && \
    apt-get clean



RUN git clone https://github.com/Challenging6/SandboxFusion.git  /tmp/sandbox_new
RUN cp -rf /tmp/sandbox_new/* /root/sandbox/ && \
    cp -rf /tmp/sandbox_new/.* /root/sandbox/ 2>/dev/null || true



# CMD ["bash", "/root/sandbox/scripts/run.sh"]