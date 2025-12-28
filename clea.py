import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -----------------------------
# Token counting
# -----------------------------

def naive_count_tokens(s: str) -> int:
    # Splits into words and punctuation tokens.
    return len(re.findall(r"\w+|[^\w\s]", s, re.UNICODE))


def tiktoken_count_tokens(s: str, model: Optional[str] = None) -> int:
    """
    Counts tokens using tiktoken if installed.
    Falls back to naive if tiktoken is unavailable or errors.
    """
    try:
        import tiktoken  # type: ignore
        # Default to a large-context encoding if model unspecified.
        # o200k_base works for GPT-4o family; cl100k_base is common fallback.
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(s))
    except Exception:
        return naive_count_tokens(s)


def count_tokens(s: str, prefer_tiktoken: bool = True) -> int:
    if prefer_tiktoken:
        return tiktoken_count_tokens(s)
    return naive_count_tokens(s)

# -----------------------------
# Utilities
# -----------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def to_text(source: Union[str, List[str], None]) -> str:
    if source is None:
        return ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def rstrip_trailing_ws(s: str) -> str:
    # Remove trailing whitespace per line
    return "\n".join(line.rstrip() for line in s.splitlines()) + ("\n" if s.endswith("\n") else "")


def strip_trailing_blank_lines(s: str) -> str:
    lines = s.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def normalize_source_text(s: str) -> str:
    s = to_text(s)
    s = normalize_newlines(s)
    s = rstrip_trailing_ws(s)
    s = strip_trailing_blank_lines(s)
    return s


def ensure_endswith_newline(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"


def longest_backtick_run(s: str) -> int:
    return max((len(m.group(0)) for m in re.finditer(r"`+", s)), default=0)


def fenced_block(text: str, lang: Optional[str] = None) -> str:
    """
    Returns a fenced code block using a fence long enough that any backticks in text won't break it.
    """
    text = text or ""
    n = longest_backtick_run(text)
    fence = "`" * max(3, n + 1)
    header = f"{fence}{lang}\n" if lang else f"{fence}\n"
    body = ensure_endswith_newline(text)
    return f"{header}{body}{fence}\n"


def iter_ipynb_files(paths: Iterable[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for nb in path.rglob("*.ipynb"):
                yield nb
        elif path.is_file() and path.suffix == ".ipynb":
            yield path
        else:
            # Skip non-notebook files silently
            continue

# -----------------------------
# Output extraction
# -----------------------------

IMAGE_MIME_PREFIXES = ("image/png", "image/jpeg", "image/svg+xml")
TEXTUAL_MIME_PREFER = ("text/plain", "application/json")

def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # Many nb outputs use list of lines
        return "".join(str(x) for x in value)
    if isinstance(value, (dict,)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    return str(value)


def pretty_json_if_needed(s: str) -> str:
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return s


def prune_output_to_text(output: Dict[str, Any],
                         strip_ansi_opt: bool,
                         remove_images: bool) -> Tuple[str, Dict[str, int]]:
    """
    Convert a single output object to a plain text representation.
    Returns (text, stats)
    """
    stats = {"images_removed": 0}
    otype = output.get("output_type", "")
    text = ""

    if otype == "stream":
        # 'text' can be str or list
        text = _as_text(output.get("text", ""))

    elif otype in ("execute_result", "display_data"):
        data = output.get("data", {}) or {}
        # Remove images from data
        if remove_images:
            for k in list(data.keys()):
                if any(k.startswith(pref) for pref in IMAGE_MIME_PREFIXES):
                    data.pop(k, None)
                    stats["images_removed"] += 1

        # Prefer text/plain, fall back to JSON, ignore HTML/LaTeX
        chosen = None
        for mime in TEXTUAL_MIME_PREFER:
            if mime in data:
                chosen = mime
                break

        if chosen == "text/plain":
            text = _as_text(data.get("text/plain", ""))
        elif chosen == "application/json":
            raw = _as_text(data.get("application/json", ""))
            text = pretty_json_if_needed(raw)
        else:
            # If nothing textual remains, nothing to add.
            text = ""

    elif otype == "error":
        ename = _as_text(output.get("ename", "")).strip()
        evalue = _as_text(output.get("evalue", "")).strip()
        tb = output.get("traceback") or []
        tb_text = _as_text(tb)
        pieces = []
        if ename or evalue:
            pieces.append(f"{ename}: {evalue}".strip(": ").strip())
        if tb_text:
            pieces.append(tb_text)
        text = "\n".join(pieces).strip() + ("\n" if pieces else "")

    # Clean text
    text = normalize_newlines(text)
    if strip_ansi_opt:
        text = strip_ansi(text)
    return text, stats


def extract_outputs_text(cell: Dict[str, Any],
                         strip_ansi_opt: bool,
                         remove_images: bool,
                         max_lines: Optional[int]) -> Tuple[str, Dict[str, int]]:
    """
    Extract combined plain-text output for a cell.
    Truncates to max_lines if specified.
    """
    outputs = cell.get("outputs") or []
    all_parts: List[str] = []
    stats = {"images_removed": 0, "truncated_outputs": 0}

    for out in outputs:
        txt, st = prune_output_to_text(out, strip_ansi_opt=strip_ansi_opt, remove_images=remove_images)
        stats["images_removed"] += st.get("images_removed", 0)
        if txt:
            all_parts.append(txt)

    combined = normalize_newlines("".join(all_parts))

    # Truncate if needed
    if max_lines is not None and max_lines > 0:
        lines = combined.splitlines()
        if len(lines) > max_lines:
            head = lines[:max_lines]
            combined = "\n".join(head) + "\n… [output truncated]\n"
            stats["truncated_outputs"] += 1

    return ensure_endswith_newline(combined) if combined else "", stats

# -----------------------------
# Cleaning notebook dict
# -----------------------------

def clean_notebook_dict(nb: Dict[str, Any],
                        allowed_nb_metadata: Optional[List[str]] = None,
                        keep_text_outputs_in_ipynb: bool = False,
                        strip_ansi_opt: bool = True,
                        max_output_lines: Optional[int] = 200,
                        include_markdown_cells_in_ipynb: bool = True) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Returns a minimized notebook dict and stats.
    - Removes heavy metadata
    - Optionally keeps text-only outputs (no images/HTML)
    - Normalizes sources as a single string to reduce JSON overhead
    """
    allowed_nb_metadata = allowed_nb_metadata or ["loutre"]

    stats = {
        "images_removed": 0,
        "cells_total": 0,
        "cells_kept": 0,
        "outputs_truncated": 0,
    }

    cleaned: Dict[str, Any] = {
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
        "metadata": {},
        "cells": [],
    }

    # Notebook metadata: keep only whitelisted keys
    nb_meta = nb.get("metadata", {}) or {}
    cleaned["metadata"] = {k: v for k, v in nb_meta.items() if k in allowed_nb_metadata}

    for cell in nb.get("cells", []):
        stats["cells_total"] += 1
        ctype = cell.get("cell_type")

        if ctype == "markdown" and not include_markdown_cells_in_ipynb:
            continue

        new_cell: Dict[str, Any] = {"cell_type": ctype}

        # Normalize source to single string
        src = normalize_source_text(cell.get("source", ""))
        new_cell["source"] = src

        # Remove cell metadata, id, execution_count
        # (Keep nothing to minimize tokens)
        # Optionally filter attachments
        # Attachments are often heavy; drop them
        # for maximum token reduction.
        # new_cell["attachments"] intentionally omitted.

        if ctype == "code":
            if keep_text_outputs_in_ipynb:
                # Keep only text-like outputs, truncate long outputs, no metadata
                text, st = extract_outputs_text(
                    cell,
                    strip_ansi_opt=strip_ansi_opt,
                    remove_images=True,
                    max_lines=max_output_lines,
                )
                stats["images_removed"] += st["images_removed"]
                stats["outputs_truncated"] += st["truncated_outputs"]

                outputs: List[Dict[str, Any]] = []
                if text:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": text
                    })
                new_cell["outputs"] = outputs
                new_cell["execution_count"] = None
            else:
                # Drop outputs entirely to minimize tokens
                new_cell["outputs"] = []
                new_cell["execution_count"] = None

        cleaned["cells"].append(new_cell)
        stats["cells_kept"] += 1

    return cleaned, stats

# -----------------------------
# Markdown export
# -----------------------------

def notebook_to_markdown(nb: Dict[str, Any],
                         include_markdown_cells: bool = False,
                         strip_ansi_opt: bool = True,
                         max_output_lines: Optional[int] = 200) -> Tuple[str, Dict[str, int]]:
    """
    Convert a notebook dict to Markdown where each code cell is:
      cell
      ```python
      <code>
      ```
      ```output
      <output>
      ```
    If include_markdown_cells is True, markdown cells are included as:
      cell
      ```python
      # (markdown)
      <markdown>
      ```
      ```output
      ```
    """
    parts: List[str] = []
    stats = {"images_removed": 0, "outputs_truncated": 0, "code_cells": 0, "md_cells": 0}

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "code":
            stats["code_cells"] += 1
            code = normalize_source_text(cell.get("source", ""))

            # Extract output text only
            out_text, st = extract_outputs_text(
                cell,
                strip_ansi_opt=strip_ansi_opt,
                remove_images=True,
                max_lines=max_output_lines,
            )
            stats["images_removed"] += st["images_removed"]
            stats["outputs_truncated"] += st["truncated_outputs"]

            # Compose blocks
            parts.append("cell\n")
            parts.append(fenced_block(code, lang="python"))
            parts.append(fenced_block(out_text, lang="output"))

        elif ctype == "markdown" and include_markdown_cells:
            stats["md_cells"] += 1
            md = normalize_source_text(cell.get("source", ""))
            # Represent markdown cell as a python block (commented) to match requested layout
            md_as_code = "# (markdown)\n" + "\n".join("# " + line if line.strip() else "#" for line in md.splitlines())
            md_as_code = ensure_endswith_newline(md_as_code)
            parts.append("cell\n")
            parts.append(fenced_block(md_as_code, lang="python"))
            parts.append(fenced_block("", lang="output"))

        else:
            # Skip markdown/raw by default to reduce tokens
            continue

    md_text = "".join(parts)
    return md_text, stats

# -----------------------------
# I/O and main
# -----------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def process_notebook(
    path: Path,
    outdir: Optional[Path],
    inplace: bool,
    include_markdown_cells_in_md: bool,
    include_markdown_cells_in_ipynb: bool,
    keep_text_outputs_in_ipynb: bool,
    strip_ansi_outputs: bool,
    max_output_lines: Optional[int],
    prefer_tiktoken: bool,
    allowed_nb_metadata: Optional[List[str]],
    md_suffix: str,
    quiet: bool,
) -> None:
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    try:
        raw = read_text(path)
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
        return

    try:
        nb = json.loads(raw)
    except Exception as e:
        print(f"❌ Invalid JSON in {path}: {e}")
        return

    if not isinstance(nb, dict) or "cells" not in nb:
        print(f"❌ Not a valid .ipynb structure: {path}")
        return

    if not quiet:
        print(f"📄 Processing: {path}")

    initial_tokens = count_tokens(raw, prefer_tiktoken=prefer_tiktoken)

    # Build Markdown export from original notebook (preserves outputs while converting to text)
    md_text, md_stats = notebook_to_markdown(
        nb,
        include_markdown_cells=include_markdown_cells_in_md,
        strip_ansi_opt=strip_ansi_outputs,
        max_output_lines=max_output_lines
    )

    # Clean notebook structure for optional in-place write
    cleaned_nb, clean_stats = clean_notebook_dict(
        nb,
        allowed_nb_metadata=allowed_nb_metadata,
        keep_text_outputs_in_ipynb=keep_text_outputs_in_ipynb,
        strip_ansi_opt=strip_ansi_outputs,
        max_output_lines=max_output_lines,
        include_markdown_cells_in_ipynb=include_markdown_cells_in_ipynb,
    )
    cleaned_json = json.dumps(cleaned_nb, ensure_ascii=False, indent=1)
    final_tokens = count_tokens(cleaned_json, prefer_tiktoken=prefer_tiktoken)
    md_tokens = count_tokens(md_text, prefer_tiktoken=prefer_tiktoken)

    # Output destinations
    target_outdir = outdir or path.parent
    md_out = target_outdir / (path.stem + md_suffix)
    write_text(md_out, md_text)

    if inplace:
        write_json(path, cleaned_nb)

    if not quiet:
        img_removed = md_stats["images_removed"] + clean_stats["images_removed"]
        print(f"✅ Done: {path.name}")
        print(f"   - Markdown: {md_out.name} (tokens ~ {md_tokens:,})")
        if inplace:
            print(f"   - In-place cleaned .ipynb written")
        print(f"   - Images removed: {img_removed}")
        if md_stats["outputs_truncated"] or clean_stats["outputs_truncated"]:
            print(f"   - Outputs truncated: {md_stats['outputs_truncated'] + clean_stats['outputs_truncated']}")
        print(f"   - Tokens (ipynb): {initial_tokens:,} -> {final_tokens:,} (Δ {initial_tokens - final_tokens:,})")
        print("-" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean Jupyter notebooks and export compact Markdown with text-only outputs."
    )
    parser.add_argument("paths", nargs="+", help="Notebook files or directories (processed recursively).")
    parser.add_argument("--outdir", type=str, default=None, help="Directory for Markdown output (default: alongside).")
    parser.add_argument("--md-suffix", type=str, default=".clean.md", help="Suffix for Markdown files.")
    parser.add_argument("--inplace", action="store_true", help="Write minimized .ipynb back to disk.")
    parser.add_argument("--include-markdown", action="store_true",
                        help="Include markdown cells in the Markdown export (as commented code).")
    parser.add_argument("--include-markdown-ipynb", action="store_true",
                        help="Keep markdown cells in the cleaned .ipynb.")
    parser.add_argument("--keep-text-outputs", action="store_true",
                        help="Keep text-only outputs (truncated) in the cleaned .ipynb.")
    parser.add_argument("--no-strip-ansi", action="store_true", help="Do not strip ANSI escape codes from outputs.")
    parser.add_argument("--max-output-lines", type=int, default=200,
                        help="Truncate outputs to at most this many lines (0 or negative disables truncation).")
    parser.add_argument("--no-tiktoken", action="store_true", help="Disable tiktoken and use naive token counting.")
    parser.add_argument("--allow-meta", nargs="*", default=["loutre"],
                        help="Allowed top-level notebook metadata keys to preserve.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output.")

    args = parser.parse_args(argv)

    outdir = Path(args.outdir) if args.outdir else None
    prefer_tiktoken = not args.no_tiktoken
    strip_ansi_outputs = not args.no_strip_ansi
    max_lines = None if args.max_output_lines is None or args.max_output_lines <= 0 else args.max_output_lines

    files = list(iter_ipynb_files(args.paths))
    if not files:
        print("No .ipynb files found.")
        return 1

    for f in files:
        process_notebook(
            path=f,
            outdir=outdir,
            inplace=args.inplace,
            include_markdown_cells_in_md=args.include_markdown,
            include_markdown_cells_in_ipynb=args.include_markdown_ipynb,
            keep_text_outputs_in_ipynb=args.keep_text_outputs,
            strip_ansi_outputs=strip_ansi_outputs,
            max_output_lines=max_lines,
            prefer_tiktoken=prefer_tiktoken,
            allowed_nb_metadata=args.allow_meta,
            md_suffix=args.md_suffix,
            quiet=args.quiet,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
