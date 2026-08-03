"""Basic usage example for Cognema."""

from cognema import Memory


def main() -> None:
    memory = Memory()

    memory.observe(
        "George discussed cognitive memory algorithms",
        metadata={"source": "conversation", "topic": "cognitive-memory"},
        tags=["research", "memory"],
    )
    memory.observe("The team agreed to prototype deterministic recall first.")

    results = memory.recall("What did George discuss about memory?")
    for result in results:
        print(f"{result.score:.2f} :: {result.event.content}")

    memory.sleep()


if __name__ == "__main__":
    main()
