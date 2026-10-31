from collections.abc import Generator
from contextlib import contextmanager

import httpx


class ModelUnavailableError(Exception):
    pass


@contextmanager
def downloading(name: str, revision: str) -> Generator[None]:
    """Turns a failed download of a pinned model, or of one of its files, into an error that says how to get it."""
    try:
        yield
    # huggingface_hub wraps a refused connection in an OSError but re-raises a proxy's refusal as httpx.ProxyError.
    except (OSError, httpx.TransportError) as error:
        reason = next(iter(str(error).strip().splitlines()), type(error).__name__).rstrip(".")
        raise ModelUnavailableError(
            f"{name} (revision {revision[:12]}) isn't in the Hugging Face cache and couldn't be downloaded: {reason}. "
            "Run repolore once with access to huggingface.co, and the same HF_HOME, to download it."
        ) from error
