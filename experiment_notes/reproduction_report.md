# HippoRAG2 Reproduction Report

## Environment

- GPU: NVIDIA RTX4090 24GB
- Platform: AutoDL
- Python environment: hipporag2
- LLM: Llama-3.1-8B
- Embedding: NV-Embed-v2


## Modifications

1. NV-Embed-v2 memory optimization
2. LLM loading optimization
3. OpenIE triple extraction repair
4. Offline OpenIE result reuse
5. 4bit quantization experiments


## Main Experiment

Dataset:
HotPotQA

Samples:
1000


Configuration:

- RAG:
  HippoRAG2

- OpenIE:
  online mode

- Index:
  rebuilt


## Results

| Metric | Score |
|---|---|
| ExactMatch | 0.515 |
| F1 | 0.6499 |


## Notes

The experiment was conducted on NVIDIA RTX4090 24GB.
