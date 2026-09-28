import json

from hipporag.information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from hipporag.utils.config_utils import BaseConfig


MODEL_PATH = "/root/autodl-tmp/llama31-8b"


def main():
    config = BaseConfig(
        llm_name=MODEL_PATH,
        max_new_tokens=4096,
        openie_mode="offline",
        openie_ner_max_tokens=512,
        openie_triple_max_tokens=4096,
    )

    openie = VLLMOfflineOpenIE(config)

    # 这里使用真实 HotPotQA 文本风格的测试文档。
    # 目的是测试正式 batch_openie() -> JSON parsing -> repair -> TripleRawOutput 路径。
    chunks = {
        "test_1": {
            "content": (
                "Alaskan Sami were indigenous people who lived in Alaska. "
                "The group left Alaska after selling herds."
            )
        },
        "test_2": {
            "content": (
                "During the Civil War, a port and riverboat landing "
                "was abandoned."
            )
        },
        "test_3": {
            "content": (
                "Raymond is a summer recreation area known for "
                "outdoor activities."
            )
        },
    }

    print("=" * 80)
    print("开始测试 HippoRAG OpenIE triple repair 集成")
    print("=" * 80)

    ner_results, triple_results = openie.batch_openie(chunks)

    print("\nNER 结果：")
    for chunk_id, result in ner_results.items():
        print(chunk_id, result.unique_entities)

    print("\nTriple 结果：")
    for chunk_id, result in triple_results.items():
        print(chunk_id)
        print(result.triples)

        for triple in result.triples:
            assert isinstance(triple, list), triple
            assert len(triple) == 3, triple
            assert all(isinstance(x, str) for x in triple), triple

    print("\n" + "=" * 80)
    print("✅ 集成测试通过：所有最终 triples 都是严格的 [subject, predicate, object]")
    print("=" * 80)


if __name__ == "__main__":
    main()
