import json
from pathlib import Path
import tempfile
import unittest

from gplsi_joint_v2.artifacts import atomic_json
from gplsi_joint_v2.config import DATASETS, default_config
from gplsi_joint_v2.preflight import preflight


class PreflightTests(unittest.TestCase):
    def test_poisson_only_methods_and_deferred_workload_are_explicit(self):
        config = default_config()
        config.update(K_values=[7], visium_panels=[500],
                      primary_K={dataset: 7 for dataset in DATASETS},
                      primary_panel={dataset: 500 for dataset in DATASETS},
                      initialization_seeds=[config["seeds"]["estimator_seed"]],
                      retention_secondary=[])
        with tempfile.TemporaryDirectory(prefix="joint-preflight-") as tmp:
            root = Path(tmp)
            for dataset in DATASETS:
                splits = [{"dataset": dataset, "protocol": "section_holdout" if dataset == "visium_dlpfc"
                           else "animal_holdout" if dataset.startswith("merfish") else "patient_holdout",
                           "split_id": "first", "role_counts": {"train": 10, "held_out": 5}}]
                if dataset == "visium_dlpfc":
                    splits.append({"dataset": dataset, "protocol": "spatial_half", "split_id": "half",
                                   "role_counts": {"train": 8, "spatial_half": 7}})
                atomic_json(root / "data/manifests/joint_v2/splits" / dataset / "splits.json", splits)
            result = preflight(root, config)
            self.assertEqual(result["workload"]["stage_tasks_total"], 401)
            self.assertEqual(result["deferred_workload"]["stage_tasks_total"], 179)
            self.assertEqual(result["full_design_workload"]["stage_tasks_total"], 580)
            self.assertEqual(result["scientific_variants_per_base"]["native_outputs"], 34)
            self.assertEqual(result["scientific_variants_per_base"]["document_hunters"],
                             ["spa_current", "svs_star", "palm_accelerated"])
            self.assertEqual(result["scientific_variants_per_base"]["native_GpLSI_A"],
                             "26 W x fixed-W Poisson =26")
            self.assertEqual(sum(s["launch_membership"] == "initial" for s in result["frozen_splits"]), 3)
            self.assertGreater(result["storage"]["deferred_factor_and_numeric_scores_bytes"], 0)
            self.assertEqual(set(result["storage"]["deferred_factor_and_numeric_scores_bytes_by_dataset"]),
                             {"visium_dlpfc"})
            self.assertEqual(json.loads((root / "reports/joint_v2/PREFLIGHT.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()
