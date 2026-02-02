from __future__ import annotations

import csv
import io
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "export-fiches-csv-2026-02-02"))

STOPWORDS_FR = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "et", "eux", "il", "je", "la", "le", "leur", "lui", "ma", "mais", "me",
    "meme", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
    "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te",
    "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j",
    "l", "m", "n", "s", "t", "y", "ete", "etre", "avoir", "faire", "stage", "stages",
}

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class Fiche:
    numero: str
    intitule: str = ""
    abrege_libelle: str = ""
    abrege_intitule: str = ""
    niveau: str = ""
    niveau_intitule: str = ""
    actif: str = ""
    romes: List[str] = field(default_factory=list)
    rome_labels: List[str] = field(default_factory=list)
    formacodes: List[str] = field(default_factory=list)
    formacode_labels: List[str] = field(default_factory=list)
    blocs: List[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        parts = [
            self.numero,
            self.intitule,
            self.abrege_libelle,
            self.abrege_intitule,
            self.niveau,
            self.niveau_intitule,
            " ".join(self.romes),
            " ".join(self.rome_labels),
            " ".join(self.formacodes),
            " ".join(self.formacode_labels),
            " ".join(self.blocs),
        ]
        return " ".join(p for p in parts if p)

    def text_by_category(self) -> Dict[str, str]:
        return {
            "job_title": " ".join(
                p
                for p in [
                    self.intitule,
                    self.abrege_libelle,
                    self.abrege_intitule,
                ]
                if p
            ),
            "domaine": " ".join(
                p
                for p in [
                    " ".join(self.rome_labels),
                    " ".join(self.formacode_labels),
                ]
                if p
            ),
            "niveau": " ".join(p for p in [self.niveau, self.niveau_intitule] if p),
            "competences": " ".join(p for p in self.blocs if p),
            "rome": " ".join(p for p in [" ".join(self.romes), " ".join(self.rome_labels)] if p),
            "formacode": " ".join(p for p in [" ".join(self.formacodes), " ".join(self.formacode_labels)] if p),
        }


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [t.strip("'") for t in WORD_RE.findall(text) if t.strip("'")]


def split_keywords(query: str) -> List[str]:
    parts = [p.strip() for p in query.split(";")]
    return [p for p in parts if p]


def extract_keywords(query: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    raw = split_keywords(query)
    normalized = []
    pairs = []
    for item in raw:
        norm = normalize_text(item)
        if not norm:
            continue
        if " " not in norm and len(norm) <= 2:
            continue
        if " " not in norm and norm in STOPWORDS_FR:
            continue
        normalized.append(norm)
        pairs.append((item, norm))
    return raw, normalized, pairs


def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def load_data() -> Dict[str, Fiche]:
    fiches: Dict[str, Fiche] = {}

    standard_path = os.path.join(DATA_DIR, "export_fiches_CSV_Standard_2026_02_01.csv")
    for row in read_csv(standard_path):
        numero = row.get("Numero_Fiche", "").strip()
        if not numero:
            continue
        fiches[numero] = Fiche(
            numero=numero,
            intitule=row.get("Intitule", "").strip(),
            abrege_libelle=row.get("Abrege_Libelle", "").strip(),
            abrege_intitule=row.get("Abrege_Intitule", "").strip(),
            niveau=row.get("Nomenclature_Europe_Niveau", "").strip(),
            niveau_intitule=row.get("Nomenclature_Europe_Intitule", "").strip(),
            actif=row.get("Actif", "").strip(),
        )

    def ensure(numero: str) -> Fiche:
        if numero not in fiches:
            fiches[numero] = Fiche(numero=numero)
        return fiches[numero]

    rome_path = os.path.join(DATA_DIR, "export_fiches_CSV_Rome_2026_02_01.csv")
    for row in read_csv(rome_path):
        numero = row.get("Numero_Fiche", "").strip()
        if not numero:
            continue
        fiche = ensure(numero)
        code = row.get("Codes_Rome_Code", "").strip()
        label = row.get("Codes_Rome_Libelle", "").strip()
        if code:
            fiche.romes.append(code)
        if label:
            fiche.rome_labels.append(label)

    formacode_path = os.path.join(DATA_DIR, "export_fiches_CSV_Formacode_2026_02_01.csv")
    for row in read_csv(formacode_path):
        numero = row.get("Numero_Fiche", "").strip()
        if not numero:
            continue
        fiche = ensure(numero)
        code = row.get("Formacode_Code", "").strip()
        label = row.get("Formacode_Libelle", "").strip()
        if code:
            fiche.formacodes.append(code)
        if label:
            fiche.formacode_labels.append(label)

    blocs_path = os.path.join(DATA_DIR, "export_fiches_CSV_Blocs_De_Compétences_2026_02_01.csv")
    for row in read_csv(blocs_path):
        numero = row.get("Numero_Fiche", "").strip()
        if not numero:
            continue
        fiche = ensure(numero)
        label = row.get("Bloc_Competences_Libelle", "").strip()
        if label:
            fiche.blocs.append(label)

    return fiches


FICHES = load_data()
TOKEN_INDEX: Dict[str, Counter] = {}
SEARCH_TEXT: Dict[str, str] = {}
CATEGORY_TEXT: Dict[str, Dict[str, str]] = {}
CATEGORY_INDEX: Dict[str, Dict[str, Counter]] = {}
DB_VOCAB: set[str] = set()
DOMAIN_LABELS: Dict[str, str] = {}
DOMAIN_TO_SKILLS: Dict[str, set[str]] = {}
DOMAIN_TOKENS: Dict[str, List[str]] = {}
SKILL_VOCAB: set[str] = set()

for numero, fiche in FICHES.items():
    text = normalize_text(fiche.searchable_text())
    SEARCH_TEXT[numero] = text
    tokens = tokenize(text)
    TOKEN_INDEX[numero] = Counter(tokens)
    DB_VOCAB.update(tokens)

    per_cat = {}
    per_cat_index = {}
    for cat, cat_text in fiche.text_by_category().items():
        norm_text = normalize_text(cat_text)
        per_cat[cat] = norm_text
        per_cat_index[cat] = Counter(tokenize(norm_text))
    CATEGORY_TEXT[numero] = per_cat
    CATEGORY_INDEX[numero] = per_cat_index

    for label in fiche.rome_labels + fiche.formacode_labels:
        norm_label = normalize_text(label)
        if norm_label:
            DOMAIN_LABELS[norm_label] = label
            if norm_label not in DOMAIN_TOKENS:
                DOMAIN_TOKENS[norm_label] = [
                    t for t in tokenize(norm_label) if len(t) > 2 and t not in STOPWORDS_FR
                ]
            DOMAIN_TO_SKILLS.setdefault(norm_label, set()).update(
                {
                    t
                    for t in tokenize(" ".join(fiche.blocs))
                    if len(t) > 2 and t not in STOPWORDS_FR and "'" not in t
                }
            )
    for bloc in fiche.blocs:
        for token in tokenize(bloc):
            if len(token) > 2 and token not in STOPWORDS_FR:
                SKILL_VOCAB.add(token)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")


def keyword_variants(keyword: str) -> List[str]:
    variants = {keyword}
    m = re.search(r"\\bniveau\\s*(\\d)\\b", keyword)
    if m:
        variants.add(f"niv{m.group(1)}")
    m = re.search(r"\\bniv\\s*(\\d)\\b", keyword)
    if m:
        variants.add(f"niveau {m.group(1)}")
    return list(variants)


def niveau_value(fiche: Fiche) -> int:
    candidates = [fiche.niveau, fiche.niveau_intitule]
    for item in candidates:
        if not item:
            continue
        text = normalize_text(item)
        m = re.search(r"(?:niv|niveau)\\s*(\\d)", text)
        if m:
            return int(m.group(1))
    return 999


def actif_rank(fiche: Fiche) -> int:
    text = normalize_text(fiche.actif)
    if "active" in text or "actif" in text:
        return 0
    return 1


def score_fiche(
    pairs_by_cat: Dict[str, List[Tuple[str, str]]],
    numero: str,
) -> Tuple[int, int, List[str], int]:
    text_by_cat = CATEGORY_TEXT[numero]
    index_by_cat = CATEGORY_INDEX[numero]
    matched = []
    score = 0
    expected = 0
    for cat, pairs in pairs_by_cat.items():
        if not pairs:
            continue
        expected += len(pairs)
        counter = index_by_cat.get(cat, Counter())
        text = text_by_cat.get(cat, "")
        for display, kw in pairs:
            variants = keyword_variants(kw)
            hit = False
            for v in variants:
                if " " in v:
                    if v in text:
                        hit = True
                        score += len(tokenize(v)) or 1
                        break
                else:
                    count = counter.get(v, 0)
                    if count > 0:
                        hit = True
                        score += count
                        break
            if hit:
                matched.append(display)
    return len(matched), score, matched, expected


def normalize_niveau_input(value: str) -> str:
    trimmed = value.strip()
    if re.fullmatch(r"\\d", trimmed):
        return f"niveau {trimmed}"
    return trimmed


def requested_niveau(value: str) -> int | None:
    if not value:
        return None
    text = normalize_text(value)
    m = re.search(r"(?:niv|niveau)\\s*(\\d)", text)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\\d", text):
        return int(text)
    return None


def search(
    query_by_cat: Dict[str, str],
    limit: int = 20,
) -> Tuple[List[Tuple[Fiche, int, List[str]]], List[Tuple[str, str]]]:
    niveau_limit = requested_niveau(query_by_cat.get("niveau", ""))
    pairs_by_cat: Dict[str, List[Tuple[str, str]]] = {}
    keywords_display = []

    for cat, value in query_by_cat.items():
        if not value:
            pairs_by_cat[cat] = []
            continue
        if cat == "niveau":
            raw = split_keywords(value)
            keywords_display.extend([(cat, r) for r in raw])
            pairs_by_cat[cat] = []
            continue
        raw, _, pairs = extract_keywords(value)
        keywords_display.extend([(cat, r) for r in raw])
        pairs_by_cat[cat] = pairs

    if not any(pairs_by_cat.values()):
        return [], []

    scored = []
    for numero in TOKEN_INDEX.keys():
        fiche = FICHES[numero]
        if niveau_limit is not None:
            fiche_niveau = niveau_value(fiche)
            if fiche_niveau > niveau_limit:
                continue
        matched_count, score, matched, expected = score_fiche(pairs_by_cat, numero)
        if matched_count > 0:
            scored.append((fiche, matched_count, expected, score, matched))

    scored.sort(
        key=lambda x: (
            -(x[1] / max(x[2], 1)),
            -x[1],
            -x[3],
            actif_rank(x[0]),
            niveau_value(x[0]),
        )
    )

    results = []
    for fiche, matched_count, expected, score, matched in scored[:limit]:
        results.append((fiche, score, matched, matched_count, expected))
    return results, keywords_display


def build_role_analysis(role: str, query_by_cat: Dict[str, str], results: List[Tuple[Fiche, int, List[str], int, int]]) -> Dict[str, str]:
    analysis = {}
    if role == "ecole":
        if not results:
            analysis["status"] = "Aucune fiche RNCP compatible trouvée avec ces critères."
        else:
            analysis["status"] = "Fiches RNCP compatibles identifiées pour vérification pédagogique."
        if query_by_cat.get("niveau"):
            analysis["niveau"] = f"Niveau demandé : {query_by_cat.get('niveau')}"
        else:
            analysis["niveau"] = "Niveau non précisé par l’école."
    elif role == "employeur":
        if not results:
            analysis["status"] = "Aucune fiche RNCP évidente : l’offre semble trop spécifique ou incomplète."
        else:
            analysis["status"] = "Fiches RNCP compatibles avec les attentes exprimées."
        if query_by_cat.get("competences"):
            analysis["competences"] = "Compétences clés reconnues dans le référentiel."
        else:
            analysis["competences"] = "Compétences non précisées : l’offre gagnerait en clarté."
    return analysis


def counter_has(counter: Counter, token: str) -> bool:
    return counter.get(token, 0) > 0


QUESTIONS_BY_ROLE = {
    "apprenant": [
        ("job_title", "Quel intitulé de poste vise l’étudiant ? (ex: assistant comptable ; technicien maintenance)"),
        ("domaine", "Quel domaine d’activité ? (ex: comptabilité ; froid industriel ; robotique)"),
        ("competences", "Quelles compétences clés doivent apparaître ? (ex: maintenance système ; diagnostic ; soudure)"),
        ("niveau", "Niveau RNCP souhaité (le résultat ne dépassera pas ce niveau) ? (ex: 3 ou niveau 3)"),
        ("rome", "Code/libellé ROME ciblé ? (optionnel, laissez vide si inconnu)"),
        ("formacode", "Formacode/libellé ciblé ? (optionnel, laissez vide si inconnu)"),
    ],
    "ecole": [
        ("job_title", "Quel intitulé ou formation vise l’offre ? (ex: développeur web ; assistant RH)"),
        ("domaine", "Quel domaine d’activité couvre l’offre ?"),
        ("competences", "Quelles missions/compétences sont attendues ?"),
        ("niveau", "Niveau RNCP visé par l’école ?"),
    ],
    "employeur": [
        ("job_title", "Quel poste l’entreprise souhaite pourvoir ?"),
        ("domaine", "Quel domaine d’activité pour l’entreprise ?"),
        ("competences", "Quelles compétences clés attendues ?"),
        ("niveau", "Niveau RNCP souhaité ?"),
    ],
}


def init_state() -> Dict[str, str]:
    return {
        "step": "0",
        "role": "apprenant",
        "job_title": "",
        "domaine": "",
        "competences": "",
        "niveau": "",
        "rome": "",
        "formacode": "",
        "history": [],
        "last_user": "",
    }


def build_query_by_cat(state: Dict[str, str]) -> Dict[str, str]:
    return {
        "job_title": state.get("job_title", "").strip(),
        "domaine": state.get("domaine", "").strip(),
        "niveau": normalize_niveau_input(state.get("niveau", "")),
        "competences": state.get("competences", "").strip(),
        "rome": state.get("rome", "").strip(),
        "formacode": state.get("formacode", "").strip(),
    }


def pick_title_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    for line in lines[:8]:
        norm = normalize_text(line)
        if re.search(r"\\b(cv|curriculum|profil|poste|intitule)\\b", norm):
            return line
    return lines[0]


def parse_resume_text(text: str) -> Dict[str, str]:
    norm = normalize_text(text)
    niveau = ""
    m = re.search(r"(?:niv|niveau)\\s*(\\d)", norm)
    if m:
        niveau = f"niveau {m.group(1)}"
    title_line = pick_title_line(text)
    title_norm = normalize_text(title_line)
    cv_tokens = tokenize(norm)
    cv_counts = Counter(
        t for t in cv_tokens if len(t) > 2 and t not in STOPWORDS_FR and "'" not in t
    )
    title_tokens = {
        t for t in tokenize(title_norm) if len(t) > 2 and t not in STOPWORDS_FR
    }
    best_label = ""
    best_score = 0
    for norm_label, original in DOMAIN_LABELS.items():
        label_tokens = DOMAIN_TOKENS.get(norm_label, [])
        if not label_tokens:
            continue
        score = sum(cv_counts.get(t, 0) for t in label_tokens)
        title_bonus = sum(1 for t in label_tokens if t in title_tokens)
        score += title_bonus * 2
        has_strong = any(len(t) >= 6 and cv_counts.get(t, 0) > 0 for t in label_tokens)
        if score >= 2 or has_strong:
            if score > best_score:
                best_score = score
                best_label = original
    domain_hits = [best_label] if best_label else []

    filtered = [
        t
        for t in cv_tokens
        if len(t) > 2
        and t not in STOPWORDS_FR
        and "'" not in t
    ]
    generic_stop = {
        "projet", "projets", "mise", "les", "langue", "creation", "expert",
        "recherche", "apporter", "numerique", "sciences", "gestion", "developpement",
    }
    skill_vocab = SKILL_VOCAB
    if best_label:
        norm_domain = normalize_text(best_label)
        if norm_domain in DOMAIN_TO_SKILLS:
            skill_vocab = DOMAIN_TO_SKILLS[norm_domain]
    skill_hits = [t for t in filtered if t in skill_vocab and t not in generic_stop]
    skill_counts = Counter(skill_hits)
    skills = [w for w, _ in skill_counts.most_common(10)]

    return {
        "job_title": "",
        "domaine": " ; ".join(domain_hits),
        "competences": " ; ".join(skills),
        "niveau": niveau,
        "rome": "",
        "formacode": "",
    }


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import pdfplumber  # type: ignore

        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            return ""


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "state" not in session:
        session["state"] = init_state()

    state = dict(session["state"])
    role = request.args.get("role") or state.get("role", "apprenant")
    if role not in QUESTIONS_BY_ROLE:
        role = "apprenant"
    state["role"] = role
    questions = QUESTIONS_BY_ROLE[role]
    results = []
    keywords = []
    done = False
    history = list(state.get("history", []))
    last_user = state.get("last_user", "")

    if request.method == "POST":
        action = request.form.get("action", "send")
        user_input = request.form.get("query", "").strip()
        step_idx = int(state.get("step", "0"))

        if action == "restart":
            state = init_state()
            session["state"] = state
            return render_template("home.html")

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

    step_idx = int(state.get("step", "0"))
    if step_idx < 0 or step_idx > len(questions):
        state = init_state()
        session["state"] = state
        step_idx = 0
        done = False
        history = []
        last_user = ""
    done = done or step_idx >= len(questions)
    current_question = ""
    if not done and step_idx < len(questions):
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

    analysis = {}
    if done:
        query_by_cat = build_query_by_cat(state)
        results, keywords = search(query_by_cat)
        analysis = build_role_analysis(role, query_by_cat, results)

    return render_template(
        "index.html",
        history=history,
        current_question=current_question,
        last_user=last_user,
        results=results,
        keywords=keywords,
        step=int(state.get("step", "0")),
        total_steps=len(questions),
        done=done,
        default_question=questions[0][1] if questions else "",
        role=role,
        analysis=analysis,
    )


@app.route("/resume", methods=["GET", "POST"])
def resume():
    error = ""
    results = []
    keywords = []
    if request.method == "POST":
        file = request.files.get("cv")
        if not file or not file.filename:
            error = "Veuillez ajouter un CV au format .pdf."
        elif not file.filename.lower().endswith(".pdf"):
            error = "Format non supporté. Utilisez un fichier .pdf."
        else:
            file_bytes = file.read()
            text = extract_pdf_text(file_bytes)
            if not text.strip():
                error = "Impossible de lire ce PDF. Essayez un autre fichier."
            else:
                query_by_cat = parse_resume_text(text)
                results, keywords = search(query_by_cat)
    return render_template(
        "resume.html",
        results=results,
        keywords=keywords,
        error=error,
        done=bool(results or error),
        total_steps=len(QUESTIONS_BY_ROLE["apprenant"]),
    )


if __name__ == "__main__":
    app.run(debug=True)
