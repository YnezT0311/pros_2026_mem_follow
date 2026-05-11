from typing import Any, Dict, List


class MethodAdapter:
    backend_name = ""
    supports_parallel_mcq = False

    def preload(
        self,
        stage_batches: List[Dict[str, Any]],
        context_messages: List[Dict[str, str]],
        ask_period: str,
    ) -> None:
        raise NotImplementedError

    def answer_mcq(self, question: str, choices: Dict[str, str]) -> Dict[str, Any]:
        raise NotImplementedError

    def debug_payload(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        return None
