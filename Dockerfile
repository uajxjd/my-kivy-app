# Dockerfile  (放项目根目录)
FROM python:3.11-slim

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY pyproject.toml /app/
COPY src/ /app/src/
WORKDIR /app
RUN pip install -e .

# 权重与数据挂载点
VOLUME ["/data", "/weights"]
ENV PYTHONPATH=/app

# 默认命令：训练 + 预测
CMD ["fc3d", "/data/UAFC3D.csv", "--model", "transformer_large"]