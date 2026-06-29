import os

IGNORE = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build"
}

report = []
report.append("# CODEBASE AUDIT REPORT\n")

python_files = []
requirements = []
dockerfiles = []
yaml_files = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE]

    for file in files:
        path = os.path.join(root, file)

        if file.endswith(".py"):
            python_files.append(path)

        if "requirements" in file.lower():
            requirements.append(path)

        if file == "Dockerfile":
            dockerfiles.append(path)

        if file.endswith((".yaml", ".yml")):
            yaml_files.append(path)

report.append(f"Python Files: {len(python_files)}")
report.append(f"Requirements Files: {len(requirements)}")
report.append(f"Dockerfiles: {len(dockerfiles)}")
report.append(f"YAML Files: {len(yaml_files)}")

with open("audit/codebase_audit_report.md", "w") as f:
    f.write("\n".join(report))

print("Audit report generated")


