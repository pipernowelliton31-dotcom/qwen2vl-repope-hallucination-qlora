"""Fast deterministic acceptance test for the registered checkpoint-selection policy."""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.select_checkpoint import choose

baseline = {"precision": .95, "recall": .70, "f1": .81, "yes_ratio": .4, "true_yes_ratio": .5, "overall_fpr": .03, "adversarial_fpr": .05}
candidates = [
    {"path": "bad_adversarial", "precision": .95, "recall": .99, "f1": .97, "yes_ratio": .9, "true_yes_ratio": .5, "overall_fpr": .04, "adversarial_fpr": .08},
    {"path": "bad_overall", "precision": .95, "recall": .99, "f1": .97, "yes_ratio": .5, "true_yes_ratio": .5, "overall_fpr": .041, "adversarial_fpr": .05},
    {"path": "higher_f1", "precision": .95, "recall": .802, "f1": .88, "yes_ratio": .5, "true_yes_ratio": .5, "overall_fpr": .04, "adversarial_fpr": .05},
    {"path": "lower_f1", "precision": .95, "recall": .803, "f1": .86, "yes_ratio": .5, "true_yes_ratio": .5, "overall_fpr": .03, "adversarial_fpr": .04},
]
assert choose(baseline, candidates)["selected"]["path"] == "higher_f1"
assert choose(baseline, candidates[:2])["status"] == "no_eligible_checkpoint"
assert choose(baseline, candidates)["constraints"]["overall_fpr_max"] == .04
print("selection-rule tests passed")
