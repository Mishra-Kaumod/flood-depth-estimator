import os

IGNORE = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules"
}

with open("docs/CODE_INVENTORY.md", "w") as f:

    f.write("# CODE INVENTORY\n\n")

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE]

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                        content = fp.read()

                    f.write(f"## {path}\n")
                    f.write(f"Lines: {len(content.splitlines())}\n\n")

                except:
                    pass

print("Inventory Created")
