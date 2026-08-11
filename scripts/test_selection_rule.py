"""Fast deterministic acceptance test for the registered checkpoint-selection policy."""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.select_checkpoint import choose

baseline = {"precision": .95, "recall": .70, "f1": .81, "yes_ratio": .4, "true_yes_ratio": .5, "adversarial_fpr": .05}
candidates = [
    {"path": "bad", "precision": .95, "recall": .99, "f1": .97, "yes_ratio": .9, "true_yes_ratio": .5, "adversarial_fpr": .08},
    {"path": "higher_f1", "precision": .95, "recall": .802, "f1": .88, "yes_ratio": .5, "true_yes_ratio": .5, "adversarial_fpr": .05},
    {"path": "lower_f1", "precision": .95, "recall": .803, "f1": .86, "yes_ratio": .5, "true_yes_ratio": .5, "adversarial_fpr": .04},
]
assert choose(baseline, candidates)["selected"]["path"] == "higher_f1"
assert choose(baseline, [candidates[0]])["status"] == "no_eligible_checkpoint"
print("selection-rule tests passed")
