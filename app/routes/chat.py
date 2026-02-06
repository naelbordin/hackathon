from __future__ import annotations

from typing import Dict, List, TypedDict

from flask import Blueprint, current_app, render_template, request, session

from services.csv_search import CsvSearchStore
from services.store import Store, get_store, normalize_text

chat_bp = Blueprint("chat", __name__)


class HistoryItem(TypedDict):
    role: str
    text: str


class ChatState(TypedDict):
    step: str
    role: str
    poste: str
    domaine: str
    competences: str
    niveau: str
    ecole: str
    localisation: str
    history: List[HistoryItem]
    last_user: str


ROLE_QUESTION = "Vous êtes plutôt étudiant ou entreprise ?"

ROLE_ALIASES = {
    "etudiant": {"etudiant", "étudiant", "apprenant", "student", "eleve", "élève"},
    "entreprise": {"entreprise", "employeur", "societe", "société", "company", "recruteur"},
}

QUESTIONS_BY_ROLE = {
    "etudiant": [
        ("poste", "Quel poste, alternance ou stage recherchez-vous ?"),
        ("domaine", "Quel secteur ou domaine d’activité ?"),
        ("competences", "Quelles compétences clés souhaitez-vous mobiliser ?"),
        ("localisation", "Quelle ville ou zone géographique ?"),
    ],
    "entreprise": [
        ("poste", "Quel poste souhaitez-vous pourvoir ?"),
        ("domaine", "Quel domaine d’activité pour ce besoin ?"),
        ("competences", "Quelles compétences clés recherchez-vous ?"),
        ("niveau", "Quel niveau d’étude (ou niveau RNCP) ?"),
        ("ecole", "Une école ou formation ciblée ? (optionnel)"),
        ("localisation", "Quelle localisation souhaitée ?"),
    ],
}

COMPANY_HEADERS = [
    "nom",
    "secteur",
    "poste",
    "competences",
    "ville",
    "email",
    "site",
    "description",
]

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


def init_state() -> ChatState:
    return {
        "step": "0",
        "role": "",
        "poste": "",
        "domaine": "",
        "competences": "",
        "niveau": "",
        "ecole": "",
        "localisation": "",
        "history": [],
        "last_user": "",
    }


def resolve_role(raw: str) -> str:
    text = normalize_text(raw or "")
    for role, aliases in ROLE_ALIASES.items():
        if any(a in text for a in aliases):
            return role
    return ""


def build_query_by_cat(state: ChatState) -> Dict[str, str]:
    return {
        "poste": state.get("poste", "").strip(),
        "domaine": state.get("domaine", "").strip(),
        "competences": state.get("competences", "").strip(),
        "niveau": state.get("niveau", "").strip(),
        "ecole": state.get("ecole", "").strip(),
        "localisation": state.get("localisation", "").strip(),
    }


def build_query_by_cat_rncp(state: ChatState, store: Store) -> Dict[str, str]:
    return {
        "job_title": state.get("poste", "").strip(),
        "domaine": state.get("domaine", "").strip(),
        "niveau": store.normalize_niveau_input(state.get("niveau", "")),
        "competences": state.get("competences", "").strip(),
        "rome": "",
        "formacode": "",
    }


def get_companies_store() -> CsvSearchStore:
    return CsvSearchStore(
        path=current_app.config["COMPANIES_CSV"],
        headers=COMPANY_HEADERS,
        category_map={
            "poste": ["poste", "description"],
            "domaine": ["secteur", "description"],
            "competences": ["competences", "description"],
            "localisation": ["ville"],
            "nom": ["nom"],
        },
        logger=current_app.logger,
    )


def get_students_store() -> CsvSearchStore:
    return CsvSearchStore(
        path=current_app.config["STUDENTS_CSV"],
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


@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    if "state" not in session:
        session["state"] = init_state()

    state: ChatState = dict(session["state"])
    role = resolve_role(request.args.get("role", "")) or state.get("role", "")
    if role and role not in QUESTIONS_BY_ROLE:
        role = ""
    if role and role != state.get("role"):
        state = init_state()
        state["role"] = role
    questions = QUESTIONS_BY_ROLE.get(role, [])
    results = []
    keywords = []
    done = False
    history = list(state.get("history", []))
    last_user = state.get("last_user", "")
    result_type = ""
    analysis = {}

    if request.method == "POST":
        action = request.form.get("action", "send")
        user_input = request.form.get("query", "").strip()

        if action == "restart":
            state = init_state()
            session["state"] = state
            return render_template("home.html")

        if not state.get("role"):
            if user_input:
                history.append({"role": "user", "text": user_input})
                selected_role = resolve_role(user_input)
                if selected_role:
                    state = init_state()
                    state["role"] = selected_role
                    history.append({"role": "bot", "text": "Parfait, lançons la recherche."})
                else:
                    history.append(
                        {"role": "bot", "text": "Merci d’indiquer « étudiant » ou « entreprise »."}
                    )
        else:
            step_idx = int(state.get("step", "0"))
            if step_idx >= len(questions):
                state = init_state()
                step_idx = 0
                history = []
                last_user = ""
                state["role"] = role

            if step_idx < len(questions):
                if action == "skip":
                    key, _ = questions[step_idx]
                    state[key] = ";"
                    state["step"] = str(step_idx + 1)
                    last_user = ""
                elif user_input:
                    key, _ = questions[step_idx]
                    state[key] = user_input
                    state["step"] = str(step_idx + 1)
                    history.append({"role": "user", "text": user_input})
                    last_user = user_input

            if int(state["step"]) >= len(questions):
                done = True
                history.append({"role": "bot", "text": "Voici les meilleurs résultats selon vos critères."})

        state["history"] = history
        state["last_user"] = last_user
        session["state"] = state

    role = state.get("role", "")
    questions = QUESTIONS_BY_ROLE.get(role, [])
    step_idx = int(state.get("step", "0")) if role else 0
    if role and (step_idx < 0 or step_idx > len(questions)):
        state = init_state()
        session["state"] = state
        step_idx = 0
        done = False
        history = []
        last_user = ""
        role = ""
        questions = []

    done = done or (role and step_idx >= len(questions))
    current_question = ""
    if not role:
        current_question = ROLE_QUESTION
        if not history or history[-1].get("text") != current_question:
            history.append({"role": "bot", "text": current_question})
    elif not done and step_idx < len(questions):
        current_question = questions[step_idx][1]
        if not history or history[-1].get("text") != current_question:
            history.append({"role": "bot", "text": current_question})
    elif done and not history:
        history.append({"role": "bot", "text": "Voici les meilleurs résultats selon vos critères."})
    elif not history:
        history.append({"role": "bot", "text": "Posez-moi une nouvelle recherche si besoin."})

    if state.get("history") != history:
        state["history"] = history
        session["state"] = state

    if done:
        query_by_cat = build_query_by_cat(state)
        if role == "etudiant":
            store = get_store(current_app)
            rncp_query = build_query_by_cat_rncp(state, store)
            results, keywords = store.search(rncp_query)
            result_type = "rncp"
            if results:
                analysis["status"] = "Fiches RNCP correspondant à votre recherche."
            else:
                analysis["status"] = "Aucune fiche RNCP trouvée avec ces critères."
        elif role == "entreprise":
            store = get_students_store()
            results, keywords = store.search(query_by_cat)
            result_type = "students"
            if results:
                analysis["status"] = "Profils étudiants correspondant à votre recherche."
            else:
                analysis["status"] = "Aucun profil étudiant trouvé avec ces critères."

    return render_template(
        "index.html",
        history=history,
        current_question=current_question,
        last_user=last_user,
        results=results,
        keywords=keywords,
        step=step_idx,
        total_steps=(len(questions) + (0 if role else 1)),
        done=done,
        default_question=questions[0][1] if questions else ROLE_QUESTION,
        role=role,
        analysis=analysis,
        result_type=result_type,
    )
