import json
from typing import Dict, Tuple

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline

logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)
        self.ner_max_tokens = global_config.openie_ner_max_tokens
        self.triple_max_tokens = global_config.openie_triple_max_tokens

    def close(self) -> None:
        """Release the internally owned vLLM engine."""
        llm_model = getattr(self, "llm_model", None)
        if llm_model is None:
            return
        close = getattr(llm_model, "close", None)
        if callable(close):
            close()
        else:
            client = getattr(llm_model, "client", None)
            client_close = getattr(client, "close", None)
            if callable(client_close):
                client_close()
            else:
                model_executor = getattr(getattr(client, "llm_engine", None), "model_executor", None)
                shutdown = getattr(model_executor, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    logger.warning("The vLLM engine exposes no explicit close or shutdown method; releasing HippoRAG's reference only.")
        self.llm_model = None

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_input_messages = [self.prompt_template_manager.render(name='ner', passage=p) for p in chunk_passages.values()]
        ner_output, ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=self.ner_max_tokens)

        triple_extract_input_messages = [self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=named_entities
        ) for passage, named_entities in zip(chunk_passages.values(), ner_output)]
        triple_output, triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=self.triple_max_tokens)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                unique_entities = json.loads(response)["named_entities"]
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunks.keys(), ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                triples = json.loads(response)["triples"]

                repaired_triples = []

                for triple in triples:
                    if not isinstance(triple, (list, tuple)):
                        logger.warning(
                            f"Invalid triple type for chunk_id {chunk_id}: {triple}"
                        )
                        continue

                    triple = [str(value).strip() for value in triple]

                    # 正常三元组：直接保留
                    if len(triple) == 3:
                        repaired_triples.append(triple)

                    # 4 元组及以上：
                    # 将第 3 项之后的内容合并到 object
                    elif len(triple) > 3:
                        repaired = [
                            triple[0],
                            triple[1],
                            " ".join(value for value in triple[2:] if value)
                        ]

                        logger.warning(
                            f"Repairing {len(triple)}-element triple "
                            f"for chunk_id {chunk_id}: "
                            f"{triple} -> {repaired}"
                        )

                        repaired_triples.append(repaired)

                    # 2 元组：使用 LLM 修复
                    elif len(triple) == 2:
                        repair_prompt = f"""
You are repairing an RDF triple extraction result.

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

                        try:
                            repair_messages = [[
                                {
                                    "role": "user",
                                    "content": repair_prompt
                                }
                            ]]

                            repair_response, _ = self.llm_model.batch_infer(
                                repair_messages,
                                json_template="triples",
                                max_tokens=128
                            )

                            repaired_json = json.loads(repair_response[0])
                            repaired_result = repaired_json["triples"]

                            if (
                                isinstance(repaired_result, list)
                                and len(repaired_result) == 1
                                and isinstance(repaired_result[0], list)
                                and len(repaired_result[0]) == 3
                                and all(
                                    isinstance(x, str)
                                    for x in repaired_result[0]
                                )
                            ):
                                repaired_triple = repaired_result[0]
                                repaired_triples.append(repaired_triple)

                                logger.warning(
                                    f"LLM repaired 2-element triple "
                                    f"for chunk_id {chunk_id}: "
                                    f"{triple} -> {repaired_triple}"
                                )
                            else:
                                logger.warning(
                                    f"LLM repair returned invalid format "
                                    f"for chunk_id {chunk_id}: "
                                    f"{repair_response[0]}"
                                )

                        except Exception as repair_error:
                            logger.warning(
                                f"LLM repair failed for chunk_id {chunk_id}: "
                                f"{repair_error}"
                            )

                    # 0/1 元组：无法可靠修复，直接丢弃
                    else:
                        logger.warning(
                            f"Dropping invalid {len(triple)}-element triple "
                            f"for chunk_id {chunk_id}: {triple}"
                        )

                triples = repaired_triples

            except Exception as e:
                triples = []
                logger.warning(f"Could not parse response from OpenIE: {e}")

            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            triple_raw_output = TripleRawOutput(chunk_id, response, triples, {})
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunks.keys(), triple_raw_outputs)}

        return ner_results_dict, triple_results_dict
