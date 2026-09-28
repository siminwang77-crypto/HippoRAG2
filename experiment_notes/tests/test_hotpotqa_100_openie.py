import json
from collections import Counter

from hipporag.information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from hipporag.utils.config_utils import BaseConfig


MODEL_PATH = "/root/autodl-tmp/llama31-8b"
DATA_PATH = "reproduce/dataset/hotpotqa_corpus.json"


def main():
    print("=" * 80)
    print("HotPotQA 100-doc OpenIE integration test")
    print("=" * 80)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus = corpus[:100]

    chunks = {}
    for i, doc in enumerate(corpus):
        title = doc.get("title", "")
        text = doc.get("text", "")

        chunks[str(i)] = {
            "content": f"{title}\n{text}"
        }

    print(f"Loaded documents: {len(chunks)}")

    config = BaseConfig(
        llm_name=MODEL_PATH,
        max_new_tokens=4096,
        openie_mode="offline",
        openie_ner_max_tokens=512,
        openie_triple_max_tokens=4096,
    )

    openie = VLLMOfflineOpenIE(config)

    print("\n开始执行正式 batch_openie() ...")
    print("这一步会同时测试 NER、Triple extraction 和我们刚刚加入的 repair。")
    print()

    ner_results, triple_results = openie.batch_openie(chunks)

    print("\n" + "=" * 80)
    print("结果统计")
    print("=" * 80)

    print(f"NER documents: {len(ner_results)}")
    print(f"Triple documents: {len(triple_results)}")

    total_triples = 0
    invalid_triples = []
    empty_triples = 0

    for chunk_id, result in triple_results.items():
        triples = result.triples

        if not triples:
            empty_triples += 1

        total_triples += len(triples)

        for triple in triples:
            if (
                not isinstance(triple, list)
                or len(triple) != 3
                or not all(isinstance(x, str) for x in triple)
            ):
                invalid_triples.append((chunk_id, triple))

    print(f"Total triples: {total_triples}")
    print(f"Empty triple documents: {empty_triples}")
    print(f"Invalid triples: {len(invalid_triples)}")

    if invalid_triples:
        print("\n❌ 发现非法 triple：")
        for chunk_id, triple in invalid_triples[:20]:
            print(f"  chunk={chunk_id}: {triple}")

        raise RuntimeError(
            f"Found {len(invalid_triples)} invalid triples."
        )

    print("\n" + "=" * 80)
    print("✅ 100 个真实 HotPotQA 文档 OpenIE 集成测试通过")
    print("✅ 所有最终 triples 均满足 [subject, predicate, object]")
    print("=" * 80)


if __name__ == "__main__":
    main()
