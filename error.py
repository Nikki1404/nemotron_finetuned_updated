(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_asr_final# docker build -t nemotron_finetuned .
[+] Building 102.7s (9/14)                                                                                     docker:default
 => [internal] load build definition from Dockerfile                                                                     0.0s
 => => transferring dockerfile: 4.92kB                                                                                   0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                                  0.3s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                               0.0s
 => [internal] load .dockerignore                                                                                        0.0s
 => => transferring context: 2B                                                                                          0.0s
 => [1/9] FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07  0.0s
 => [internal] load build context                                                                                        0.0s
 => => transferring context: 35.70kB                                                                                     0.0s
 => CACHED [2/9] WORKDIR /srv                                                                                            0.0s
 => [3/9] RUN apt-get update &&     apt-get upgrade -y &&     apt-get dist-upgrade -y &&     apt-get install -y --no-i  97.6s
 => ERROR [4/9] RUN wget --no-check-certificate         https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz &&    4.6s
------
 > [4/9] RUN wget --no-check-certificate         https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz &&     tar -xzf Python-3.11.9.tgz &&     cd Python-3.11.9 &&     ./configure --enable-optimizations &&     make -j$(nproc) &&     make install &&     cd / && rm -rf Python-3.11.9*:
0.372 --2026-07-10 06:52:56--  https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
0.375 Resolving www.python.org (www.python.org)... 167.82.56.223, 167.82.60.223, 2a04:4e42:fea::223, ...
0.414 Connecting to www.python.org (www.python.org)|167.82.56.223|:443... connected.
0.643 Unable to establish SSL connection.
------
Dockerfile:35
--------------------
  34 |     # ── 2. Python 3.11 ────────────────────────────────────────────────────────────
  35 | >>> RUN wget --no-check-certificate \
  36 | >>>         https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
  37 | >>>     tar -xzf Python-3.11.9.tgz && \
  38 | >>>     cd Python-3.11.9 && \
  39 | >>>     ./configure --enable-optimizations && \
  40 | >>>     make -j$(nproc) && \
  41 | >>>     make install && \
  42 | >>>     cd / && rm -rf Python-3.11.9*
  43 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c wget --no-check-certificate         https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz &&     tar -xzf Python-3.11.9.tgz &&     cd Python-3.11.9 &&     ./configure --enable-optimizations &&     make -j$(nproc) &&     make install &&     cd / && rm -rf Python-3.11.9*" did not complete successfully: exit code: 4
