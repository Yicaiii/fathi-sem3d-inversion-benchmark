from datetime import datetime
import argparse
import json
import shutil
import sys

from scripts.fathi_benchmark.runtime_paths import repository_root, resolve_path


ROOT = repository_root()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--force", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config_path = resolve_path(
    args.config,
    base=ROOT,
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = (
    resolve_path(
        config["run_result_root"],
        base=ROOT,
    )
    / transition
)

run_data_root = (
    resolve_path(
        config["run_data_root"],
        base=ROOT,
    )
    / f"iter_{kp1:03d}"
)

candidate_root = run_result_root / "candidates"

strict_workspace_name = config.get(
    "strict_forward_workspace_name",
    "strict_full_forward_000",
)
template_dir = (
    run_data_root
    / "forward_dudx_mgcap_full_batches"
    / strict_workspace_name
)

forward_root = (
    run_data_root
    / "candidate_forward_workspaces"
)

required_template = [
    template_dir / "input.spec",
    template_dir / "mesh.input",
    template_dir / "material.input",
    template_dir / "material.spec",
]

missing = [p for p in required_template if not p.exists()]
if missing:
    print("Missing template files:")
    for p in missing:
        print(" ", p)
    sys.exit(1)

def ignore_runtime(dirpath, names):
    ignored = set()
    runtime_dirs = {
        "traces",
        "prot",
        "res",
        "logs",
        "__pycache__",
    }
    runtime_suffixes = (
        ".log",
        ".out",
        ".err",
    )
    for n in names:
        if n in runtime_dirs:
            ignored.add(n)
        if n.endswith(runtime_suffixes):
            ignored.add(n)
    return ignored

records = []

candidate_dirs = sorted(
    candidate_root.glob(
        "line_search_neg_mtilde_*"
    )
)

existing_workspaces = [
    forward_root / cand_dir.name
    for cand_dir in candidate_dirs
    if (forward_root / cand_dir.name).exists()
]

if existing_workspaces and not args.force:
    print(
        "Refusing to overwrite existing candidate "
        "forward workspaces:"
    )

    for workspace in existing_workspaces:
        print(" ", workspace)

    print(
        "Re-run with --force only after confirming "
        "workspace replacement is intended."
    )
    sys.exit(3)

forward_root.mkdir(
    parents=True,
    exist_ok=True,
)

for cand_dir in candidate_dirs:
    cand_h5 = cand_dir / "mat/h5"
    if not cand_h5.exists():
        records.append({
            "candidate": cand_dir.name,
            "ok": False,
            "reason": "missing_candidate_h5",
        })
        continue

    ws = forward_root / cand_dir.name

    if ws.exists():
        shutil.rmtree(ws)

    shutil.copytree(template_dir, ws, ignore=ignore_runtime)

    ws_h5 = ws / "mat/h5"
    if ws_h5.exists():
        shutil.rmtree(ws_h5)
    ws_h5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cand_h5, ws_h5)

    required_ws = [
        ws / "input.spec",
        ws / "mesh.input",
        ws / "material.input",
        ws / "material.spec",
        ws / "mat/h5/Mat_0_Kappa.h5",
        ws / "mat/h5/Mat_0_Mu.h5",
        ws / "mat/h5/Mat_0_Density.h5",
    ]

    missing_ws = [p for p in required_ws if not p.exists()]

    records.append({
        "candidate": cand_dir.name,
        "candidate_dir": str(cand_dir),
        "workspace": str(ws),
        "missing": [str(p) for p in missing_ws],
        "ok": len(missing_ws) == 0,
    })

all_ok = len(records) > 0 and all(r["ok"] for r in records)

report_dir = resolve_path(
    "benchmark_fathi_strict/reports/candidate_generation",
    base=ROOT,
)
report_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "template_dir": str(template_dir),
    "forward_root": str(forward_root),
    "force": bool(args.force),
    "records": records,
    "result": "PASS" if all_ok else "CHECK_NEEDED",
}

json_out = report_dir / f"{transition}_candidate_forward_workspace_prepare.json"
txt_out = report_dir / f"{transition}_candidate_forward_workspace_prepare.txt"

json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 4D prepare candidate forward workspaces")
lines.append("============================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"template_dir = {template_dir}")
lines.append(f"forward_root = {forward_root}")
lines.append("")
for r in records:
    lines.append("-" * 80)
    lines.append(f"candidate = {r['candidate']}")
    lines.append(f"workspace = {r.get('workspace')}")
    lines.append(f"ok = {r['ok']}")
    if r.get("missing"):
        lines.append("missing:")
        for p in r["missing"]:
            lines.append(f"  {p}")
    if r.get("reason"):
        lines.append(f"reason = {r['reason']}")

lines.append("")
lines.append("Interpretation:")
lines.append("  Candidate forward workspaces are prepared but SEM3D has NOT been executed.")
lines.append("  Next stage is to run candidate forward one candidate at a time.")
lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if payload["result"] != "PASS":
    sys.exit(2)
