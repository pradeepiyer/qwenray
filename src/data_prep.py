#!/usr/bin/env python3
"""Generate training data from Ray and Anyscale documentation using OpenAI."""

import json
import re
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = "You are an expert on Ray distributed computing and the Anyscale platform."

QA_GENERATION_PROMPT = """Based on the following documentation, generate 2-3 high-quality \
question-answer pairs that would help someone learn about Ray or Anyscale.

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
        return default_path

    raise FileNotFoundError(
        "Ray directory not found. Please specify --ray-dir or clone ray to ~/ray"
    )


def get_anyscale_dir(anyscale_path: Path | None = None) -> Path:
    """Get path to Anyscale docs repository."""
    if anyscale_path is not None:
        if not anyscale_path.exists():
            raise FileNotFoundError(f"Anyscale directory not found: {anyscale_path}")
        return anyscale_path

    # Default to ~/docs
    default_path = Path.home() / "docs"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Anyscale docs directory not found. Please specify --anyscale-dir or clone to ~/docs"
    )


def find_ray_files(ray_dir: Path) -> list[Path]:
    """Find all markdown and RST files in Ray docs."""
    doc_source = ray_dir / "doc" / "source"
    files = []
    for pattern in ["**/*.md", "**/*.rst"]:
        files.extend(doc_source.glob(pattern))
    return sorted(files)


def find_anyscale_files(anyscale_dir: Path) -> list[Path]:
    """Find all markdown files in Anyscale docs."""
    doc_source = anyscale_dir / "docs"
    return sorted(doc_source.glob("**/*.md"))


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


def generate_samples_from_chunks(
    client: OpenAI,
    chunks: list[str],
    target_samples: int,
    source_name: str,
) -> list[dict]:
    """Generate Q&A samples from a list of chunks until target is reached."""
    samples: list[dict] = []
    chunk_idx = 0

    while len(samples) < target_samples and chunk_idx < len(chunks):
        chunk = chunks[chunk_idx]
        chunk_idx += 1

        print(
            f"[{source_name}] Processing chunk {chunk_idx}/{len(chunks)} "
            f"(have {len(samples)}/{target_samples} samples)..."
        )

        pairs = generate_qa_pairs(client, chunk)
        for pair in pairs:
            if "question" in pair and "answer" in pair:
                samples.append(to_chatml(pair["question"], pair["answer"]))

        if len(samples) >= target_samples:
            break

    return samples[:target_samples]


def main(
    output_path: Path = Path("data/ray_anyscale_10k.jsonl"),
    sources: list[str] = ["ray"],
    ray_samples: int = 500,
    anyscale_samples: int = 0,
    ray_dir_path: Path | None = None,
    anyscale_dir_path: Path | None = None,
):
    """Generate training dataset from Ray and/or Anyscale docs."""
    client = OpenAI()
    all_samples: list[dict] = []

    # Process Ray docs
    if "ray" in sources and ray_samples > 0:
        ray_dir = get_ray_dir(ray_dir_path)
        print(f"Using Ray repo at {ray_dir}")

        ray_files = find_ray_files(ray_dir)
        print(f"Found {len(ray_files)} Ray documentation files")

        ray_chunks: list[str] = []
        for file_path in ray_files:
            ray_chunks.extend(extract_chunks(file_path))
        print(f"Extracted {len(ray_chunks)} Ray content chunks")

        ray_result = generate_samples_from_chunks(
            client, ray_chunks, ray_samples, "Ray"
        )
        print(f"Generated {len(ray_result)} Ray samples")
        all_samples.extend(ray_result)

    # Process Anyscale docs
    if "anyscale" in sources and anyscale_samples > 0:
        anyscale_dir = get_anyscale_dir(anyscale_dir_path)
        print(f"Using Anyscale docs at {anyscale_dir}")

        anyscale_files = find_anyscale_files(anyscale_dir)
        print(f"Found {len(anyscale_files)} Anyscale documentation files")

        anyscale_chunks: list[str] = []
        for file_path in anyscale_files:
            anyscale_chunks.extend(extract_chunks(file_path))
        print(f"Extracted {len(anyscale_chunks)} Anyscale content chunks")

        anyscale_result = generate_samples_from_chunks(
            client, anyscale_chunks, anyscale_samples, "Anyscale"
        )
        print(f"Generated {len(anyscale_result)} Anyscale samples")
        all_samples.extend(anyscale_result)

    # Deduplicate
    all_samples = deduplicate(all_samples)
    print(f"After deduplication: {len(all_samples)} samples")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Ray/Anyscale Q&A training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ray_anyscale_10k.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="ray",
        help="Comma-separated sources: ray,anyscale (default: ray)",
    )
    parser.add_argument(
        "--ray-samples",
        type=int,
        default=500,
        help="Target Ray samples (default: 500)",
    )
    parser.add_argument(
        "--anyscale-samples",
        type=int,
        default=0,
        help="Target Anyscale samples (default: 0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Total samples (shortcut for ray-only, backwards compatible)",
    )
    parser.add_argument(
        "--ray-dir",
        type=Path,
        default=None,
        help="Path to Ray repository (default: ~/ray)",
    )
    parser.add_argument(
        "--anyscale-dir",
        type=Path,
        default=None,
        help="Path to Anyscale docs repository (default: ~/docs)",
    )

    args = parser.parse_args()

    # Parse sources
    sources = [s.strip() for s in args.sources.split(",")]

    # Backwards compatibility: --samples sets ray-samples for ray-only
    ray_samples = args.ray_samples
    if args.samples is not None and "anyscale" not in sources:
        ray_samples = args.samples

    main(
        output_path=args.output,
        sources=sources,
        ray_samples=ray_samples,
        anyscale_samples=args.anyscale_samples,
        ray_dir_path=args.ray_dir,
        anyscale_dir_path=args.anyscale_dir,
    )
