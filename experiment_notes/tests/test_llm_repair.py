import json

from hipporag.llm.vllm_offline import VLLMOffline
from hipporag.utils.config_utils import BaseConfig


MODEL_PATH = "/root/autodl-tmp/llama31-8b"


def repair_with_llm(client, triple):
    prompt = f"""You are repairing an RDF triple extraction result.

The input contains exactly two strings:

{json.dumps(triple, ensure_ascii=False)}

Convert it into exactly one valid RDF triple.

Rules:
1. Output JSON only.
2. The output MUST be an object with the key "triples".
3. "triples" MUST contain exactly one triple.
4. The triple MUST contain exactly three strings:
   [subject, predicate, object]
5. Keep the subject unchanged.
6. Split the second input string into a predicate and an object.
7. Preserve the original meaning.
8. Do not add information that is not present in the input.
9. Never output fewer or more than three elements.

Example:

Input:
["Raymond", "is a summer recreation area"]

Output:
{{"triples":[["Raymond","is","a summer recreation area"]]}}

Now repair this input:

{json.dumps(triple, ensure_ascii=False)}
"""

    messages = [[{"role": "user", "content": prompt}]]

    responses, metadata = client.batch_infer(
        messages,
        max_tokens=128,
        json_template="triples"
    )

    return responses[0]


if __name__ == "__main__":
    config = BaseConfig(
        llm_name=MODEL_PATH,
        max_new_tokens=128,
        openie_mode="offline"
    )

    client = VLLMOffline(config)

    test_cases = [
        ["Alaskan Sami", "left Alaska after selling herds"],
        ["Civil War", "had port and riverboat landing abandoned"],
        ["Raymond", "is a summer recreation area"],
    ]

    for triple in test_cases:
        print("=" * 70)
        print("原始:", triple)

        try:
            result = repair_with_llm(client, triple)

            print("LLM 原始输出:")
            print(result)

            try:
                parsed = json.loads(result)

                print("解析结果:")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))

                valid = (
                    isinstance(parsed, dict)
                    and isinstance(parsed.get("triples"), list)
                    and len(parsed["triples"]) == 1
                    and isinstance(parsed["triples"][0], list)
                    and len(parsed["triples"][0]) == 3
                    and all(
                        isinstance(x, str)
                        for x in parsed["triples"][0]
                    )
                )

                if valid:
                    print("状态: VALID")
                else:
                    print("状态: INVALID_FORMAT")

            except Exception as e:
                print("JSON 解析失败:", repr(e))

        except Exception as e:
            print("LLM 调用失败:", repr(e))
