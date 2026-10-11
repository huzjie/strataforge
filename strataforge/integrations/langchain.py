"""LangChain chat-model adapter (optional; import only when LangChain is present).

Because strataforge exposes an OpenAI-compatible HTTP API, the easiest real
integration is LangChain's ChatOpenAI pointed at http://127.0.0.1:8080/v1.
This class wraps that with sensible defaults.
"""


class StrataChatModel:
    def __init__(self, base_url="http://127.0.0.1:8080/v1",
                 model="qwen3.8-flash-next", temperature=0.0):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def to_chat_openai(self):
        """Return a configured langchain ChatOpenAI (imports langchain lazily)."""
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(
            base_url=self.base_url,
            api_key="sk-strataforge",
            model=self.model,
            temperature=self.temperature,
        )

    def to_langchain_llm(self):
        """Return a langchain LLM bound to the local server."""
        from langchain_openai import OpenAI  # type: ignore
        return OpenAI(
            base_url=self.base_url,
            api_key="sk-strataforge",
            model=self.model,
            temperature=self.temperature,
        )
