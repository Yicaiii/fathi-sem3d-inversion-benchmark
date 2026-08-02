from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math


@dataclass(frozen=True)
class ObjectiveTerms:
    j_data: float
    tv_lambda: float
    tv_mu: float
    alpha_lambda: float
    alpha_mu: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} is not finite: {value}")
        if self.j_data < 0.0:
            raise ValueError("j_data must be non-negative")
        if self.tv_lambda < 0.0 or self.tv_mu < 0.0:
            raise ValueError("TV values must be non-negative")
        if self.alpha_lambda < 0.0 or self.alpha_mu < 0.0:
            raise ValueError("TV weights must be non-negative")

    @property
    def j_regularization(self) -> float:
        self.validate()
        return self.alpha_lambda * self.tv_lambda + self.alpha_mu * self.tv_mu

    @property
    def j_total(self) -> float:
        return self.j_data + self.j_regularization

    def to_dict(self) -> dict:
        return {**asdict(self), "j_regularization": self.j_regularization, "j_total": self.j_total}


def decide_acceptance(parent: ObjectiveTerms, candidate: ObjectiveTerms, absolute_tolerance: float = 0.0) -> dict:
    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative")
    delta = candidate.j_total - parent.j_total
    accepted = delta < -absolute_tolerance
    return {
        "parent": parent.to_dict(),
        "candidate": candidate.to_dict(),
        "delta_j_total": delta,
        "absolute_tolerance": absolute_tolerance,
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECT",
    }


def load_tv_values(path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    tv_lambda = payload.get("tv_lambda_hat", payload.get("tv_lambda"))
    tv_mu = payload.get("tv_mu_hat", payload.get("tv_mu"))
    if tv_lambda is None or tv_mu is None:
        raise KeyError(f"Cannot find TV values in {path}")
    return float(tv_lambda), float(tv_mu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-j-data", type=float, required=True)
    parser.add_argument("--candidate-j-data", type=float, required=True)
    parser.add_argument("--parent-tv-json", required=True)
    parser.add_argument("--candidate-tv-json", required=True)
    parser.add_argument("--alpha-lambda", type=float, required=True)
    parser.add_argument("--alpha-mu", type=float, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parent_tv_lambda, parent_tv_mu = load_tv_values(args.parent_tv_json)
    candidate_tv_lambda, candidate_tv_mu = load_tv_values(args.candidate_tv_json)
    result = decide_acceptance(
        ObjectiveTerms(args.parent_j_data, parent_tv_lambda, parent_tv_mu, args.alpha_lambda, args.alpha_mu),
        ObjectiveTerms(args.candidate_j_data, candidate_tv_lambda, candidate_tv_mu, args.alpha_lambda, args.alpha_mu),
        args.absolute_tolerance,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("RESULT = PASS_TOTAL_OBJECTIVE_EVALUATION")


if __name__ == "__main__":
    main()
