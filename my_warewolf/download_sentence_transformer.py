import os
from sentence_transformers import SentenceTransformer

# 指定镜像站（关键步骤）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# # 加载模型（代码会通过镜像站下载或读取缓存）
model_qa = SentenceTransformer('checkpoints\model_qa')
model_sym = SentenceTransformer('checkpoints\model_sym')



# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="sentence-transformers/multi-qa-mpnet-base-cos-v1",
#     local_dir="checkpoints/model_qa",
#     resume_download=True,
#     endpoint="https://hf-mirror.com"  # 镜像源
# )

# snapshot_download(
#     repo_id="sentence-transformers/all-mpnet-base-v2",
#     local_dir="checkpoints/model_sym",
#     resume_download=True,
#     endpoint="https://hf-mirror.com"  # 镜像源
# )