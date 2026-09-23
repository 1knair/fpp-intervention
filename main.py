"""Command-line entry point for safe, chunked FPP data processing."""

# argparse defines the command-line interface; logging reports progress and warnings.
import argparse
import logging

# Path provides platform-independent file and directory paths.
from pathlib import Path

# These helpers locate participants and run the processing pipeline.
from src.data_loader import get_data_root, get_participants
from src.pipeline import export_cleaned_dataset, process_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a basic summary for each discovered simulator drive."
    )
    # Select the dataset location, or leave it unset to use the configured fallback.
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Dataset root; otherwise FPP_DATA_ROOT or OneDrive fallback is used.",
    )
    # Exactly one selection mode is required, which prevents conflicting requests
    # such as processing one participant while also asking only for a list.
    selection = parser.add_mutually_exclusive_group(required=True)
    # Process one participant folder by its numeric ID.
    selection.add_argument(
        "--participant",
        help="Participant ID, for example: --participant 002",
    )
    # Process every participant directory discovered in the dataset.
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process every currently discovered participant.",
    )
    # Show available participant IDs, without processing telemetry files.
    selection.add_argument(
        "--list-participants",
        action="store_true",
        help="List discovered participant IDs without reading telemetry.",
    )
    # Save the processed summaries instead of printing them to the terminal.
    parser.add_argument("--output", type=Path, help="Write summaries to this CSV.")
    parser.add_argument(
        "--clean-output",
        type=Path,
        help="Write cleaned row-level CSV files and a cleaning report here.",
    )
    return parser


def _is_within(path: Path, directory: Path) -> bool:
    # Resolve both paths, then check whether ``path`` is inside ``directory``.
    # relative_to() raises ValueError when the path is outside that directory.
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    # Parse the user's options, configure messages, and locate the dataset root.
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_root = get_data_root(args.data_root)

    # Listing participants is a separate, read-only mode, so it exits early.
    if args.list_participants:
        participants = get_participants(data_root)
        print(f"Discovered {len(participants)} participant(s):")
        print(" ".join(participant.name for participant in participants))
        return

    # Protect raw data by refusing to place generated output anywhere inside it.
    if args.output and _is_within(args.output, data_root):
        raise ValueError("Refusing to write output inside the raw dataset root")
    if args.clean_output and _is_within(args.clean_output, data_root):
        raise ValueError("Refusing to write output inside the raw dataset root")

    # None means all participants; otherwise pass the one requested ID as a list.
    participant_ids = None if args.all else [args.participant]

    if args.clean_output:
        report = export_cleaned_dataset(
            data_root,
            args.clean_output,
            participant_ids,
        )
        print(f"Wrote {len(report)} cleaned drive file(s) to {args.clean_output}")
        if not args.output:
            return

    # Pass the resolved root and either a list of IDs or None (meaning all IDs).
    summaries = process_dataset(data_root, participant_ids)
    logging.info("Created %d basic drive summary row(s)", len(summaries))

    # With --output, create its parent directories and save a headered CSV without
    # a pandas index. Otherwise, print either an empty-result message or the table.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summaries.to_csv(args.output, index=False)
        print(f"Wrote {len(summaries)} row(s) to {args.output}")
    elif summaries.empty:
        print("No matching valid drive files were found.")
    else:
        print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
