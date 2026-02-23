"""
Sample Query Runner - FinReg Navigator

Usage:
    python sample_query.py
    python sample_query.py --query "When was the Finance Act 2025 passed?"
    python sample_query.py --pdf path/to/doc.pdf --query "Does this comply with SBP rules?"
    python sample_query.py --image path/to/chart.png --query "What does this chart show?"
    python sample_query.py --query "..." --verbose
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from logs.logging_config import setup_logging
from src.core.router import Router


def run(query: str, files: list = None, verbose: bool = False):
    router = Router()  # also calls setup_logging() internally

    print("\n" + "=" * 70)
    print(f"  QUERY  : {query}")
    if files:
        print(f"  FILES  : {[f[0] for f in files]}")
    print(f"  VERBOSE: {verbose}")
    print("=" * 70)

    kwargs = {"query": query}
    if files:
        kwargs["files"] = files

    result = router.handle_input(**kwargs)

    # Progress steps
    steps = result.get("progress", [])
    if steps:
        print("\n📋 STEPS:")
        for s in steps:
            print(f"   {s}")

    # Mode + context summary
    print(f"\n⚙️  MODE : {result.get('mode', 'unknown')}")
    print(f"\n📦 CONTEXT:")
    print(f"   Regulatory KB chunks : {len(result.get('regulatory_text',  []))}")
    print(f"   Uploaded doc chunks  : {len(result.get('uploaded_text',    []))}")
    print(f"   Web results          : {len(result.get('web_results',      []))}")
    print(f"   Image references     : {len(result.get('images',           []))}")

    web = result.get("web_results", [])
    if web:
        print("\n🌐 WEB SOURCES:")
        for r in web:
            print(f"   • {r['title']}")
            print(f"     {r['url']}")

    imgs = result.get("images", [])
    if imgs:
        print("\n🖼️  IMAGES:")
        for p in imgs:
            print(f"   • {p}")

    print("\n" + "─" * 70)
    print("💬 ANSWER:")
    print("─" * 70)
    print(result.get("answer", "No answer returned."))
    print("=" * 70 + "\n")
    return result


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   type=str,  default=None)
    parser.add_argument("--pdf",     type=str,  default=None)
    parser.add_argument("--image",   type=str,  default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    files = None
    if args.pdf or args.image:
        files = []
        for path_str in filter(None, [args.pdf, args.image]):
            p = Path(path_str)
            if not p.exists():
                print(f"ERROR: File not found: {p}")
                sys.exit(1)
            files.append((p.name, p.read_bytes()))

    if not args.query:
        print("No --query provided. Running two default tests.\n")
        run("What are the capital requirements for an EMI license in Pakistan?")
        run("When was Pakistan Finance Act 2025 passed in Pakistan?")
    else:
        run(query=args.query, files=files, verbose=args.verbose)


if __name__ == "__main__":
    main()