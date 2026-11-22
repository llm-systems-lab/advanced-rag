import itertools
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from langchain_core.documents import Document

from generation import stream_answer

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


@contextmanager
def _loading(message: str = "Thinking") -> Generator[None]:
    """Show a spinner while waiting for the model response."""
    if not sys.stdout.isatty():
        yield
        return

    stop = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.is_set():
                break
            sys.stdout.write(f"\r{DIM}{message} {frame}{RESET}")
            sys.stdout.flush()
            time.sleep(0.1)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def _format_sources(sources: list[Document]) -> str:
    seen: list[str] = []
    for doc in sources:
        source = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 0))
        label = f"{source} (page {page})"
        if label not in seen:
            seen.append(label)
    return "\n".join(f"  - {label}" for label in seen)


def _render_stream(question: str) -> None:
    """Stream the assistant's answer to the terminal, then list its sources."""
    with _loading():
        sources, stream = stream_answer(question)
        iterator = iter(stream)
        first = next(iterator, "")

    print(f"\n{CYAN}{BOLD}Assistant{RESET}")
    sys.stdout.write(first)
    sys.stdout.flush()
    for chunk in iterator:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print()

    if sources:
        print(f"\n{DIM}Sources:{RESET}")
        print(_format_sources(sources))
    print()


def main() -> None:
    """Run the interactive question-answer loop until the user types 'exit'."""
    print(f"{BOLD}Product Research Assistant{RESET} {DIM}(type 'exit' to quit){RESET}")
    while True:
        try:
            question = input(f"{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            _render_stream(question)
        except Exception as exc:
            print(f"\n{RED}Error:{RESET} {exc}\n")
            continue

    print(f"{DIM}Goodbye.{RESET}")


if __name__ == "__main__":
    main()
