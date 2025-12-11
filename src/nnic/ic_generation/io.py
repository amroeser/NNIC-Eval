from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import json

from nnic.ic_generation.rule_extraction import AtomicCondition, ICRule


def load_ic_rules_from_json(path: Path) -> List[ICRule]:
    """Lädt IC-Regeln aus einer JSON-Datei, die zuvor mit `ICRule.to_serializable`
    erzeugt wurde, und rekonstruiert eine Liste von `ICRule`-Objekten.

    Dies ermöglicht eine semi-automatische Revision von Regeln: Die JSON-Datei kann
    manuell editiert werden (z.B. Regeln löschen/ändern), anschließend werden die
    angepassten Regeln erneut in die NNIC-Pipeline eingespeist.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f) or []

    rules: List[ICRule] = []
    for raw in data:
        antecedent_clauses_raw = raw.get("antecedent_clauses", [])
        antecedent_clauses: List[List[AtomicCondition]] = []
        for clause in antecedent_clauses_raw:
            clause_atoms: List[AtomicCondition] = []
            for cond in clause:
                clause_atoms.append(
                    AtomicCondition(
                        feature=str(cond["feature"]),
                        op=str(cond["op"]),
                        threshold=float(cond["threshold"]),
                    )
                )
            antecedent_clauses.append(clause_atoms)

        rule = ICRule(
            id=int(raw.get("id", len(rules))),
            antecedent_clauses=antecedent_clauses,
            consequent=list(raw.get("consequent", [])),
            support=int(raw.get("support", 0)),
            error_fraction=float(raw.get("error_fraction", 0.0)),
            tree_index=int(raw.get("tree_index", -1)),
        )
        rules.append(rule)

    return rules
