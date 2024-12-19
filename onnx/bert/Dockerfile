FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder

# Only use runtime as final image
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN mkdir -p /usr/local/cuda/lib64
COPY --from=builder /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.4 /usr/local/cuda/lib64/

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH