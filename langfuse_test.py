import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langfuse import Langfuse


def build_client() -> Langfuse:
    load_dotenv()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip('"')

    if not public_key or not secret_key:
        raise RuntimeError("Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY in .env")

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def main() -> None:
    langfuse = build_client()
    trace_name = "langfuse-smoke-test"

    with langfuse.start_as_current_observation(
        name=trace_name,
        as_type="span",
        input={"prompt": "Say hello to Langfuse"},
        metadata={"source": "local-smoke-test", "timestamp": datetime.now(timezone.utc).isoformat()},
    ) as span:
        response = "Hello from a simple Langfuse logging test."
        span.update(output={"response": response, "status": "success"})

        span.create_event(
            name="smoke-test-event",
            input={"step": "event"},
            output={"message": "Event logged successfully"},
            metadata={"trace_name": trace_name},
        )

    langfuse.flush()

    trace_url = langfuse.get_trace_url(trace_id=span.trace_id)
    print("Langfuse smoke test complete")
    print(f"Trace ID: {span.trace_id}")
    print(f"Trace URL: {trace_url}")


if __name__ == "__main__":
    main()
