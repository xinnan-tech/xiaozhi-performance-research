# 在预装 PyTorch 2.7.0 + Python 3.12 + CUDA 11.8 的 Ubuntu 22.04 镜像容器上安装 vLLM_Qwen2.5-72B-Instruct并以vLLM启动文档

## 环境检查
### 确认 GPU 可用性
nvidia-smi
应输出你的 GPU 信息，确认驱动版本≥ 525，确认显存足以部署Qwen2.5-72B-Instruct模型
### 检查 CUDA 和 PyTorch
python -c "import torch; print(torch.version.cuda)"
预期输出：
2.7.0 True 11.8
## 安装 vLLM

### 直接安装vLLM及其依赖
pip install vllm

### 如果CUDA版本不支持最新vLLM版本，可克隆vLLM源码编译

#### 安装编译依赖
sudo apt update
sudo apt install -y build-essential git cmake ninja-build

#### 克隆vLLM仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.5.4  # 或是适用的任何版本

#### 编译安装 vLLM
pip install -e ".[tensorizer]"

## 下载 Qwen2.5-72B-Instruct 模型

### 一、通过Hugging Face下载

#### 安装 huggingface_hub（如未安装）
pip install huggingface_hub

#### 登录（不一定需要，如果提示需要模型访问权限则需登录申请模型访问权限）
huggingface-cli login
输入Hugging Face申请的Read Token

#### 下载模型到本地目录（bash）
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    local_dir="/path/to/Qwen2.5-72B-Instruct",
    local_dir_use_symlinks=False  # 避免软链接，节省 inode
)

### 二、通过ModelScope下载
#### 安装 modelscope（如未安装）
pip install modelscope

#### 下载模型到本地目录(Python)
from modelscope import snapshot_download
model_dir = snapshot_download(
    'qwen/Qwen2.5-72B-Instruct',
    cache_dir='/path/to/Qwen2.5-72B-Instruct'
)
print(f"模型已下载至: {model_dir}")

## 启动vLLM服务

### 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \  # 模型名称
  --tensor-parallel-size 4 \           # 张量并行度，根据GPU数量调整
  --dtype bfloat16 \                   # 数据类型，bfloat16 对 Qwen2.5 模型更友好
  --max-model-len 32768 \              # 最大模型长度，根据模型规格调整
  --gpu-memory-utilization 0.92 \      # GPU 内存利用率，根据实际显存调整
  --trust-remote-code \                # 允许加载远程代码，必要时开启
  --host 0.0.0.0 \                     # 绑定到所有网络接口，允许外部访问
  --port 8000 \                        # API 服务端口，根据实际情况调整
  --enable-chunked-prefill \           # 启用分块填充，提高吞吐量
  --max-num-seqs 256 \                 # 最大并发序列数，根据 GPU 内存调整
  --disable-log-stats \                # 禁用日志统计，减少日志输出
