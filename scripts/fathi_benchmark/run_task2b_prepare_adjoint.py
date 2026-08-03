from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import sys

try:
    from scripts.fathi_benchmark.runtime_paths import repository_root
except ModuleNotFoundError:
    from runtime_paths import repository_root

ROOT = repository_root()

parser = argparse.ArgumentParser()
parser.add_argument("--context", required=True)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

ctx_path = Path(args.context).expanduser()
if not ctx_path.is_absolute():
    ctx_path = ROOT / ctx_path
ctx_path = ctx_path.resolve()

ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
transition = ctx["transition"]

adj_base = Path(ctx["output_adjoint_batches_dir"]).expanduser()
if not adj_base.is_absolute():
    adj_base = ROOT / adj_base
adj_base = adj_base.resolve()

config_path = Path(ctx["config_path"]).expanduser()
if not config_path.is_absolute():
    config_path = ROOT / config_path
config_path = config_path.resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))

profile_value = ctx.get(
    "benchmark_profile_config",
    config.get(
        "benchmark_profile_config",
        "configs/fathi_reduced_3x3_12p5.json",
    ),
)
profile_path = Path(profile_value).expanduser()
if not profile_path.is_absolute():
    profile_path = ROOT / profile_path
profile_path = profile_path.resolve()

batch_count = int(
    ctx.get(
        "adjoint_batch_count",
        config.get("adjoint_batch_count", 10),
    )
)
if batch_count <= 0:
    raise RuntimeError(f"Invalid adjoint_batch_count: {batch_count}")

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)
out_txt = report_dir / f"{transition}_prepare_adjoint_task.txt"
out_json = report_dir / f"{transition}_prepare_adjoint_task.json"

legacy_modules = [
    "scripts.fathi_benchmark.generic_from_legacy.455A_extract_old_adjoint_source_format_generic",
    "scripts.fathi_benchmark.generic_from_legacy.455B_prepare_strict_adjoint_batches_from_residual_generic",
    "scripts.fathi_benchmark.generic_from_legacy.455C_audit_strict_adjoint_batches_generic",
]
standalone_module = "scripts.fathi_benchmark.prepare_full_adjoint_from_strict"


def inspect_batches():
    records = []
    for comp in ["x", "y", "z"]:
        for i in range(batch_count):
            batch = f"batch_{i:03d}"
            directory = adj_base / comp / batch
            record = {
                "component": comp,
                "batch": batch,
                "dir": str(directory),
                "has_dir": directory.exists(),
                "has_input_spec": (directory / "input.spec").exists(),
                "has_material_h5": (
                    directory / "mat/h5/Mat_0_Kappa.h5"
                ).exists(),
                "has_traces": (
                    (directory / "traces").exists()
                    and len(list((directory / "traces").glob("capteurs.*.h5"))) > 0
                ),
            }
            record["ok_prepared"] = (
                record["has_dir"]
                and record["has_input_spec"]
                and record["has_material_h5"]
            )
            records.append(record)
    return records


def extract_result(stdout):
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("RESULT ="):
            return stripped.split("=", 1)[1].strip()
    return None


def run_command(label, command):
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_result = extract_result(proc.stdout)
    return {
        "label": label,
        "command": command,
        "returncode": proc.returncode,
        "child_result": child_result,
        "stdout_tail": proc.stdout.splitlines()[-80:],
        "stderr_tail": proc.stderr.splitlines()[-80:],
        "ok": (
            proc.returncode == 0
            and child_result is not None
            and child_result.startswith("PASS")
        ),
    }


def valid_template_roots(value):
    return (
        isinstance(value, dict)
        and all(value.get(component) for component in ("x", "y", "z"))
    )


template_roots = ctx.get("adjoint_template_roots")
preparation_mode = (
    "legacy_templates"
    if valid_template_roots(template_roots)
    else "standalone_current_strict"
)

if preparation_mode == "legacy_templates":
    planned_commands = [
        [
            sys.executable,
            "-m",
            module,
            "--context",
            str(ctx_path),
        ]
        for module in legacy_modules
    ]
else:
    standalone_command = [
        sys.executable,
        "-m",
        standalone_module,
        "--context",
        str(ctx_path),
        "--benchmark-spec",
        str(profile_path),
        "--batch-count",
        str(batch_count),
    ]
    if args.execute:
        standalone_command.append("--execute")
    if args.force:
        standalone_command.append("--force")
    planned_commands = [standalone_command]

created = datetime.now().isoformat()
records = inspect_batches()
prepared_target = 3 * batch_count
prepared_count = sum(1 for record in records if record["ok_prepared"])
trace_count_ready = sum(1 for record in records if record["has_traces"])

payload = {
    "created": created,
    "transition": transition,
    "task_type": "prepare_adjoint",
    "context": str(ctx_path),
    "config_path": str(config_path),
    "benchmark_profile_config": str(profile_path),
    "adjoint_batches_dir": str(adj_base),
    "adjoint_batch_count": batch_count,
    "prepared_target": prepared_target,
    "preparation_mode": preparation_mode,
    "execute": args.execute,
    "force": args.force,
    "prepared_count_before": prepared_count,
    "trace_count_ready_before": trace_count_ready,
    "result": None,
}

lines = [
    "Executable task: prepare_adjoint",
    "================================",
    "",
    f"created = {created}",
    f"transition = {transition}",
    f"context = {ctx_path}",
    f"config = {config_path}",
    f"benchmark_profile_config = {profile_path}",
    f"adjoint_batches_dir = {adj_base}",
    f"adjoint_batch_count = {batch_count}",
    f"preparation_mode = {preparation_mode}",
    f"execute = {args.execute}",
    f"force = {args.force}",
    "",
    f"prepared_count_before = {prepared_count} / {prepared_target}",
    f"trace_count_ready_before = {trace_count_ready} / {prepared_target}",
    "",
]

if prepared_count == prepared_target and not args.force:
    payload["result"] = "PASS_ALREADY_EXISTS"
    lines.append("Adjoint batch workspaces are already prepared.")
    lines.append("No preparation command was launched.")

elif not args.execute:
    payload["result"] = "PASS_PLAN_ONLY"
    lines.append("Plan only. Would run:")
    for command in planned_commands:
        lines.append("  " + " ".join(command))

else:
    run_records = []
    ok = True

    commands_to_run = []
    for command in planned_commands:
        command_to_run = list(command)
        if (
            preparation_mode == "legacy_templates"
            and args.force
            and "455B_prepare_strict_adjoint_batches_from_residual_generic"
            in command_to_run[2]
        ):
            command_to_run.append("--allow-overwrite")
        commands_to_run.append(command_to_run)

    for index, command in enumerate(commands_to_run):
        label = (
            legacy_modules[index]
            if preparation_mode == "legacy_templates"
            else standalone_module
        )
        lines.extend(["", "Running:", "  " + " ".join(command)])
        run_record = run_command(label, command)
        run_records.append(run_record)

        lines.append(f"returncode = {run_record['returncode']}")
        lines.append(f"child_result = {run_record['child_result']}")
        lines.append("stdout tail:")
        for line in run_record["stdout_tail"]:
            lines.append("  " + line)
        if run_record["stderr_tail"]:
            lines.append("stderr tail:")
            for line in run_record["stderr_tail"]:
                lines.append("  " + line)

        if not run_record["ok"]:
            ok = False
            break

    records_after = inspect_batches()
    prepared_count_after = sum(
        1 for record in records_after if record["ok_prepared"]
    )
    trace_count_ready_after = sum(
        1 for record in records_after if record["has_traces"]
    )
    payload["script_runs"] = run_records
    payload["prepared_count_after"] = prepared_count_after
    payload["trace_count_ready_after"] = trace_count_ready_after
    payload["result"] = (
        "PASS_EXECUTED"
        if ok and prepared_count_after == prepared_target
        else "FAIL_PREPARE_ADJOINT"
    )

    lines.extend(
        [
            "",
            f"prepared_count_after = {prepared_count_after} / {prepared_target}",
            f"trace_count_ready_after = {trace_count_ready_after} / {prepared_target}",
        ]
    )

payload["records_preview"] = records[:5]
lines.extend(["", f"json = {out_json}", "", f"RESULT = {payload['result']}"])

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if payload["result"].startswith("FAIL"):
    sys.exit(1)
