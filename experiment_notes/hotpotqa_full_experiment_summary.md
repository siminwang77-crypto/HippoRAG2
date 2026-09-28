# FD-RAG / HippoRAG 复现实验记录

## 1. 实验信息

- 数据集：HotPotQA
- 评测样本数：1000
- RAG 类型：HippoRAG
- LLM：Llama 3.1 8B
- LLM 路径：`/root/autodl-tmp/llama31-8b`
- Embedding 模型：`nvidia/NV-Embed-v2`
- Embedding Provider：`nvembed`
- GPU：NVIDIA RTX 4090 24GB
- OpenIE 模式：online
- OpenIE 最大生成 token 数：4096
- Embedding batch size：1
- OpenIE：复用已有离线 OpenIE 结果
- Index：重新构建

## 2. 实验命令

```bash
python main.py \
  --dataset hotpotqa \
  --num_samples 1000 \
  --rag_type hipporag \
  --llm_name Transformers//root/autodl-tmp/llama31-8b \
  --embedding_name nvidia/NV-Embed-v2 \
  --embedding_provider nvembed \
  --embedding_batch_size 1 \
  --openie_mode online \
  --openie_triple_max_tokens 4096 \
  --force_openie_from_scratch false \
  --force_index_from_scratch true \
  --save_dir outputs/hotpotqa_full
