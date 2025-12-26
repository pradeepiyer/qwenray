#!/usr/bin/env python3
"""Generate Ray Q&A training data from documentation using OpenAI."""

import json
import re
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = "You are an expert on Ray distributed computing."

QA_GENERATION_PROMPT = """Based on the following Ray documentation, generate 2-3 high-quality \
question-answer pairs that would help someone learn about Ray.

Requirements:
- Questions should be practical and specific
- Answers should be detailed and include code examples where relevant
- Focus on the key concepts and usage patterns in the documentation
- Answers should be self-contained (don't reference "the documentation above")

Documentation:
---
{content}
---

Return your response as a JSON array of objects with "question" and "answer" fields. Example:
[
  {{"question": "How do I create a Ray actor?", "answer": "To create a Ray actor..."}},
  {{"question": "What is the purpose of ray.get()?", "answer": "ray.get() is used to..."}}
]

Return ONLY the JSON array, no other text."""


def get_ray_dir(ray_path: Path | None = None) -> Path:
    """Get path to Ray repository."""
    if ray_path is not None:
        if not ray_path.exists():
            raise FileNotFoundError(f"Ray directory not found: {ray_path}")
        return ray_path

    # Default to ~/ray
    default_path = Path.home() / "ray"
    if default_path.exists():
        print(f"Using Ray repo at {default_path}")
        return default_path

    raise FileNotFoundError(
        "Ray directory not found. Please specify --ray-dir or clone ray to ~/ray"
    )


def find_markdown_files(ray_dir: Path) -> list[Path]:
    """Find all markdown and RST files in the docs."""
    doc_source = ray_dir / "doc" / "source"
    files = []
    for pattern in ["**/*.md", "**/*.rst"]:
        files.extend(doc_source.glob(pattern))
    return sorted(files)


def extract_chunks(file_path: Path, max_chunk_size: int = 3000) -> list[str]:
    """Extract content chunks from a documentation file."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    # Skip files that are too short or mostly config
    if len(content) < 200:
        return []

    # Split by headers (markdown or RST)
    header_pattern = r"(?:^#{1,3}\s+.+$|^[=-]{3,}$)"
    sections = re.split(header_pattern, content, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(current_chunk) + len(section) < max_chunk_size:
            current_chunk += "\n\n" + section
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out chunks that are too short or look like config/metadata
    return [c for c in chunks if len(c) > 300 and not c.startswith("```yaml")]


def generate_qa_pairs(client: OpenAI, chunk: str) -> list[dict]:
    """Use OpenAI to generate Q&A pairs from a documentation chunk."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": QA_GENERATION_PROMPT.format(content=chunk)},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        if not content:
            return []

        # Extract JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        pairs = json.loads(content)
        return pairs if isinstance(pairs, list) else []
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Error generating Q&A: {e}")
        return []


def to_chatml(question: str, answer: str) -> dict:
    """Convert Q&A pair to ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate samples based on question similarity."""
    seen_questions: set[str] = set()
    unique = []

    for sample in samples:
        question = sample["messages"][1]["content"].lower().strip()
        # Simple dedup based on first 50 chars
        key = question[:50]
        if key not in seen_questions:
            seen_questions.add(key)
            unique.append(sample)

    return unique


def main(
    output_path: Path = Path("data/ray_dataset_small.jsonl"),
    target_samples: int = 500,
    ray_dir_path: Path | None = None,
):
    """Generate Ray training dataset."""
    client = OpenAI()

    ray_dir = get_ray_dir(ray_dir_path)

    # Find documentation files
    md_files = find_markdown_files(ray_dir)
    print(f"Found {len(md_files)} documentation files")

    # Extract chunks from all files
    all_chunks: list[str] = []
    for file_path in md_files:
        chunks = extract_chunks(file_path)
        all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} content chunks")

    # Generate Q&A pairs until we have enough
    samples: list[dict] = []
    chunk_idx = 0

    while len(samples) < target_samples and chunk_idx < len(all_chunks):
        chunk = all_chunks[chunk_idx]
        chunk_idx += 1

        print(f"Processing chunk {chunk_idx}/{len(all_chunks)} (have {len(samples)} samples)...")

        pairs = generate_qa_pairs(client, chunk)
        for pair in pairs:
            if "question" in pair and "answer" in pair:
                samples.append(to_chatml(pair["question"], pair["answer"]))

        if len(samples) >= target_samples:
            break

    # Deduplicate
    samples = deduplicate(samples)
    print(f"After deduplication: {len(samples)} samples")

    # Trim to target
    samples = samples[:target_samples]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Ray Q&A training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ray_dataset_small.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Target number of samples",
    )
    parser.add_argument(
        "--ray-dir",
        type=Path,
        default=None,
        help="Path to Ray repository (defaults to ~/ray)",
    )

    args = parser.parse_args()
    main(output_path=args.output, target_samples=args.samples, ray_dir_path=args.ray_dir)
