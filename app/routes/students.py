from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, current_app, render_template, request

from services.csv_search import CsvSearchStore

students_bp = Blueprint("students", __name__)

STUDENT_HEADERS = [
    "nom",
    "email",
    "ecole",
    "niveau",
    "domaine",
    "poste",
    "competences",
    "ville",
    "disponibilite",
    "portfolio",
    "date_inscription",
]


def get_students_store() -> CsvSearchStore:
    path = current_app.config["STUDENTS_CSV"]
    return CsvSearchStore(
        path=path,
        headers=STUDENT_HEADERS,
        category_map={
            "poste": ["poste", "domaine"],
            "domaine": ["domaine", "poste"],
            "competences": ["competences"],
            "niveau": ["niveau"],
            "ecole": ["ecole"],
            "localisation": ["ville"],
            "nom": ["nom"],
        },
        logger=current_app.logger,
    )


@students_bp.route("/etudiants/inscription", methods=["GET", "POST"])
def register():
    store = get_students_store()
    success = False
    errors = []
    form = {
        "nom": "",
        "email": "",
        "ecole": "",
        "niveau": "",
        "domaine": "",
        "poste": "",
        "competences": "",
        "ville": "",
        "disponibilite": "",
        "portfolio": "",
    }

    if request.method == "POST":
        for key in form.keys():
            form[key] = request.form.get(key, "").strip()

        required = {
            "nom": "Nom complet",
            "email": "Email",
            "ecole": "École",
            "niveau": "Niveau",
        }
        for field, label in required.items():
            if not form.get(field):
                errors.append(f"Le champ « {label} » est obligatoire.")

        if not errors:
            payload = {**form}
            payload["date_inscription"] = datetime.now(timezone.utc).isoformat()
            store.append_row(payload)
            success = True
            form = {k: "" for k in form}

    return render_template(
        "student_register.html",
        form=form,
        success=success,
        errors=errors,
    )
