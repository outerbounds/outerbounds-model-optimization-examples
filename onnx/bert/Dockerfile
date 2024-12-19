# Stage 1: Builder
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder

# Just using runtime as final image
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Create the directory if it doesn't exist
RUN mkdir -p /usr/local/cuda/lib64

# Copy the NVRTC libraries from builder
# Based on your find output, let's pick these two:
COPY --from=builder /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.4 /usr/local/cuda/lib64/

# Add the directory containing these libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY reqs.txt /tmp/reqs.txt
RUN pip install -r /tmp/reqs.txt
