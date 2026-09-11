"""Frozen, label-independent outer roles; labels are allowed only for stratification."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd


def array_hash(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _balanced_cycle(groups: dict[str, str], holdout: int, seed: int) -> list[str]:
    """Choose a prespecified seeded balanced circular ordering, never outcome scores.

    Circular length-holdout windows guarantee every biological unit is held out
    exactly holdout times. Selection minimizes deviation of group counts from
    cohort proportions; sorted tuple tie-breaking is deterministic.
    """
    units = sorted(groups)
    labels = sorted(set(groups.values()))
    counts = Counter(groups.values())
    target = np.array([holdout * counts[g] / len(units) for g in labels])
    rng = np.random.default_rng(seed)
    candidates = [units] + [list(rng.permutation(units)) for _ in range(2048)]
    def objective(order):
        windows = [Counter(groups[order[(i+j) % len(order)]] for j in range(holdout))
                   for i in range(len(order))]
        # Keep every genotype/group represented among training biological units.
        extinct = sum(any(w[g] >= counts[g] for g in labels) for w in windows)
        worst = max(max(w.values()) for w in windows)
        deviation = sum(float(np.square(np.array([w[g] for g in labels])-target).sum())
                        for w in windows)
        return extinct, worst, round(deviation, 12), tuple(order)
    selected = min(candidates, key=objective)
    if objective(selected)[0]:
        raise ValueError("No feasible balanced ordering retaining every group in training")
    return selected


def build_outer_splits(obs: pd.DataFrame, dataset: str,
                       outer_split_seed: int = 26091001) -> list[dict]:
    """Return explicit section/biological assignments and coordinate half rules.

    Required public design metadata: obs_id, bio_id, section_id, graph_id, x,y.
    Visium also position and replicate; other platforms split_group only.
    No count or annotation array is accepted by this function.
    """
    required = {"obs_id", "bio_id", "section_id", "graph_id", "x", "y"}
    if not required.issubset(obs):
        raise ValueError(f"Missing design fields: {required-set(obs)}")
    if obs.obs_id.duplicated().any():
        raise ValueError("Observation IDs must be globally unique")
    splits = []
    common = {"dataset": dataset, "outer_split_seed": outer_split_seed,
              "observation_order_sha256": array_hash(obs.obs_id)}
    if dataset == "visium_dlpfc":
        sections = obs[["bio_id", "section_id", "position", "replicate"]].drop_duplicates()
        if len(sections) != 12 or sections.groupby("bio_id").size().tolist() != [4,4,4]:
            raise ValueError("Visium must contain three donors with four sections each")
        maps = {}
        for bio, rows in sections.groupby("bio_id", sort=True):
            positions = sorted(rows.position.unique())
            if len(positions) != 2 or set(rows.replicate.astype(int)) != {1,2}:
                raise ValueError("Verified two-position/two-replicate Visium metadata required")
            maps[bio] = {f"{'a' if r.position == positions[0] else 'b'}{int(r.replicate)}":
                         str(r.section_id) for r in rows.itertuples()}
            if set(maps[bio]) != {"a1","a2","b1","b2"}:
                raise ValueError("Duplicate/missing position-replicate mapping")
        rotations = [(('a1','b1'),'a2','b2'), (('a2','b2'),'b1','a1'),
                     (('a2','b1'),'b2','a1')]
        for i, (train, primary, additional) in enumerate(rotations,1):
            assignments = [{"bio_id":bio,"train_sections":[mapping[t] for t in train],
                            "primary_test":[mapping[primary]],"additional_test":[mapping[additional]],
                            "position_mapping":mapping} for bio,mapping in maps.items()]
            splits.append({**common,"split_id":f"section_rotation_{i:02d}",
                           "protocol":"section_holdout","assignments":assignments})
        for axis, low_name, high_name in [('x','left','right'),('y','bottom','top')]:
            thresholds = {str(section):float(np.median(group[axis]))
                          for section,group in obs.groupby("section_id",sort=True)}
            for side in [low_name, high_name]:
                splits.append({**common,"split_id":f"spatial_{side}",
                               "protocol":"spatial_half","axis":axis,
                               "test_side":"low" if side==low_name else "high",
                               "thresholds":thresholds,
                               "tie_rule":"coordinate <= median is low; coordinate > median is high",
                               "test_set":"spatial_half"})
    elif dataset in {"merfish_trem2_5xfad","xenium_uc"}:
        grouped = obs[["bio_id","split_group"]].drop_duplicates()
        if grouped.bio_id.duplicated().any():
            raise ValueError("A biological unit has inconsistent stratification groups")
        groups = dict(zip(grouped.bio_id.astype(str), grouped.split_group.astype(str)))
        holdout = 2 if dataset == "merfish_trem2_5xfad" else 3
        expected_n = 15 if holdout == 2 else 20
        if len(groups)!=expected_n:
            raise ValueError(f"Expected {expected_n} biological units, got {len(groups)}")
        cycle = _balanced_cycle(groups, holdout, outer_split_seed)
        appearances = Counter()
        section_map = {str(bio):sorted(map(str,rows.section_id.unique()))
                       for bio,rows in obs.groupby("bio_id",sort=True)}
        for i in range(len(cycle)):
            test = sorted(cycle[(i+j)%len(cycle)] for j in range(holdout))
            train = sorted(set(cycle)-set(test))
            record = {**common,"split_id":f"leave_{holdout}_out_{i+1:02d}",
                      "protocol":"animal_holdout" if holdout==2 else "patient_holdout",
                      "train_biological_ids":train,"test_biological_ids":test,
                      "test_group_counts":dict(Counter(groups[t] for t in test)),
                      "train_group_counts":dict(Counter(groups[t] for t in train)),
                      "balancing":"2048 prespecified seed permutations; circular windows; no outcomes"}
            if holdout==2:
                selected = {bio:section_map[bio][appearances[bio]%len(section_map[bio])]
                            for bio in train}
                record["selected_training_sections"] = selected
                record["seen_animal_new_section"] = {
                    bio:[s for s in section_map[bio] if s != selected[bio]] for bio in train
                    if len(section_map[bio])>1}
                appearances.update(train)
            splits.append(record)
    else:
        raise ValueError(dataset)
    for record in splits:
        roles = assign_roles(obs, record)
        record["role_counts"] = {str(k):int(v) for k,v in Counter(roles).items()}
        record["role_order_sha256"] = array_hash(roles)
        record["split_sha256"] = hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return splits


def assign_roles(obs: pd.DataFrame, split: dict) -> np.ndarray:
    if array_hash(obs.obs_id) != split["observation_order_sha256"]:
        raise ValueError("Frozen split observation order mismatch")
    roles = np.full(len(obs), "unassigned",dtype=object)
    if split["protocol"] == "section_holdout":
        for assignment in split["assignments"]:
            for role,key in [("train","train_sections"),("primary_test","primary_test"),
                             ("additional_test","additional_test")]:
                mask=(obs.bio_id==assignment["bio_id"]) & obs.section_id.isin(assignment[key])
                roles[mask]=role
    elif split["protocol"] == "spatial_half":
        thresholds=obs.section_id.map(split["thresholds"]).to_numpy(float)
        low=obs[split["axis"]].to_numpy(float)<=thresholds
        test=low if split["test_side"]=="low" else ~low
        roles[:]='train'; roles[test]=split["test_set"]
    else:
        heldout=obs.bio_id.isin(split["test_biological_ids"]).to_numpy()
        roles[:]='train'
        roles[heldout]='unseen_animal' if split["protocol"]=='animal_holdout' else 'unseen_patient'
        if split["protocol"]=='animal_holdout':
            for bio,sections in split["seen_animal_new_section"].items():
                roles[(obs.bio_id==bio)&obs.section_id.isin(sections)]='seen_animal_new_section'
    if np.any(roles=='unassigned') or not np.any(roles=='train'):
        raise ValueError("Incomplete split or no training observations")
    if "role_order_sha256" in split and array_hash(roles)!=split["role_order_sha256"]:
        raise ValueError("Frozen split role mismatch")
    return roles
