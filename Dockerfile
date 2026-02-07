# ============================================================
# QTrade — Live Trading 轻量镜像
#
# 只包含 Live Trading 需要的依赖（不含 vectorbt/numba）
# 适合 Google Cloud e2-micro (1GB RAM)
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖（不含 vectorbt 和 numba）
RUN pip install --no-cache-dir \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0" \
    "requests>=2.31" \
    "pyarrow>=14.0.0"

# 复制项目文件
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/

# 创建数据和报告目录
RUN mkdir -p data reports/live

# 环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 默认命令：Paper Trading
CMD ["python", "scripts/run_live.py", "-c", "config/rsi_adx_atr.yaml", "--paper"]
