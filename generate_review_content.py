import os
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "dataset_labeled",
    "master_dataset",
    "test_images",
    "reference_images"
}

OUTPUT = "REVIEW_CONTEXT.md"

with open(OUTPUT, "w", encoding="utf-8") as out:

    out.write("# REPOSITORY REVIEW CONTEXT\n\n")

    out.write("## PROJECT TREE\n\n")

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = root.count(os.sep)
        indent = "  " * level
        out.write(f"{indent}{os.path.basename(root)}/\n")

        for file in files:
            if file.endswith(".py"):
                out.write(f"{indent}  {file}\n")

    out.write("\n\n# PYTHON FILE CONTENTS\n\n")

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:

                    out.write("\n")
                    out.write("=" * 80)
                    out.write("\n")
                    out.write(f"FILE: {path}\n")
                    out.write("=" * 80)
                    out.write("\n\n")

                    out.write(f.read())
                    out.write("\n\n")

            except Exception as e:
                out.write(f"\nERROR READING {path}: {e}\n")

print(f"Generated {OUTPUT}")