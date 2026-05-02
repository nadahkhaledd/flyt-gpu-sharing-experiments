FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libtirpc3t64 \
        libelf1t64 \
        libssl3t64 \
        libgssapi-krb5-2 \
        zlib1g \
        libzstd1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcudarto.so

COPY matmul_bench /app/matmul_bench
COPY cricket-client.so /app/cricket-client.so
COPY libs/ /app/libs/

WORKDIR /app

ENV LD_LIBRARY_PATH=/app:/app/libs:/usr/lib/x86_64-linux-gnu
ENV LD_PRELOAD=/app/cricket-client.so

ENTRYPOINT ["/app/matmul_bench"]
CMD ["2048", "10"]
