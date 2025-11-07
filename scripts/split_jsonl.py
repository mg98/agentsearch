import os
from pathlib import Path

def split_jsonl(input_file: str, max_size_mb: float = 100):
    """Split a JSONL file into chunks under max_size_mb."""
    input_path = Path(input_file)
    output_dir = input_path.parent
    base_name = input_path.stem

    max_size_bytes = max_size_mb * 1024 * 1024
    current_size = 0
    file_index = 0
    current_file = None
    lines_in_chunk = 0

    try:
        with open(input_path, 'r') as infile:
            for line in infile:
                line_size = len(line.encode('utf-8'))

                if current_file is None or current_size + line_size > max_size_bytes:
                    if current_file:
                        current_file.close()
                        print(f"Created {output_path} with {lines_in_chunk} lines ({current_size / (1024*1024):.2f} MB)")

                    file_index += 1
                    output_path = output_dir / f"{base_name}_part{file_index}.jsonl"
                    current_file = open(output_path, 'w')
                    current_size = 0
                    lines_in_chunk = 0

                current_file.write(line)
                current_size += line_size
                lines_in_chunk += 1

        if current_file:
            current_file.close()
            print(f"Created {output_path} with {lines_in_chunk} lines ({current_size / (1024*1024):.2f} MB)")

        print(f"\nSplit complete: {file_index} files created")

    except Exception as e:
        if current_file:
            current_file.close()
        raise e

if __name__ == "__main__":
    split_jsonl("data/batch_questions_dedup.jsonl", max_size_mb=90)
