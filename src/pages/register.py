import streamlit as st
import base64
import os
import re
import csv
from pathlib import Path
from datetime import date, datetime


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GENRES = [
    "Comedy", "Drama", "Romance", "Crime", "Action", "Thriller", "Documentary",
    "Adventure", "Science Fiction", "Animation", "Family", "Mystery", "Horror",
    "Fantasy", "War", "Music", "Western", "History", "TV Movie",
]


def get_base64_of_bin_file(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "users.csv"


def _load_users() -> list[dict]:
    path = _csv_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _username_or_email_exists(username: str, email: str) -> tuple[bool, bool]:
    users = _load_users()
    username_l = username.strip().lower()
    email_l = email.strip().lower()
    taken_user = any((u.get("username") or "").strip().lower() == username_l for u in users)
    taken_email = any((u.get("email") or "").strip().lower() == email_l for u in users)
    return taken_user, taken_email


def _next_user_id() -> int:
    users = _load_users()
    ids = []
    for u in users:
        try:
            ids.append(int(u.get("user_id", 0)))
        except (ValueError, TypeError):
            pass
    return max(ids) + 1 if ids else 1


def _save_user(new_user: dict) -> bool:
    path = _csv_path()
    fieldnames = [
        "user_id", "username", "email", "password",
        "gender", "age", "favorite_genres", "created_at"
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    try:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(new_user)
        return True
    except Exception:
        return False


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", email.strip()))


def _password_strength(password: str):
    EMPTY, RED, ORANGE, YELLOW, GREEN = "#1e293b", "#ef4444", "#f97316", "#eab308", "#22c55e"
    if not password:
        return 0, "", [EMPTY] * 4

    score = sum([
        len(password) >= 8,
        bool(re.search(r"[A-Z]", password)),
        bool(re.search(r"[0-9]", password)),
        bool(re.search(r"[^A-Za-z0-9]", password)),
    ])

    level = max(1, score)
    palettes = {
        1: ("Débil",   [RED,    EMPTY,  EMPTY,  EMPTY]),
        2: ("Regular", [ORANGE, ORANGE, EMPTY,  EMPTY]),
        3: ("Buena",   [YELLOW, YELLOW, YELLOW, EMPTY]),
        4: ("Fuerte",  [GREEN,  GREEN,  GREEN,  GREEN]),
    }
    label, colors = palettes[level]
    return level, label, colors


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def show_register():
    # Si ya está logueado, ir al dashboard
    if st.session_state.get("logged_in"):
        st.query_params["page"] = "dashboard"
        st.rerun()
        return

    # ── Background ────────────────────────────────────────────────────────────
    bin_str = ""
    try:
        bin_str = get_base64_of_bin_file(os.path.join("assets", "fondo-login.png"))
    except Exception:
        pass

    logo_str = ""
    try:
        logo_str = get_base64_of_bin_file(os.path.join("assets", "logo-streaming.png"))
    except Exception:
        pass

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        :root {{
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --glass-bg: rgba(30, 41, 59, 0.78);
            --glass-border: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #818cf8;
            --input-bg: rgba(15, 23, 42, 0.82);
            --error-bg: rgba(127, 29, 29, 0.92);
            --success-bg: rgba(20, 83, 45, 0.92);
        }}

        .stApp {{
            background: linear-gradient(rgba(15,23,42,0.62), rgba(15,23,42,0.62)),
                        url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Outfit', sans-serif;
        }}

        .main .block-container {{
            max-width: 100% !important;
            padding: 2vh 0 6vh 0 !important;
        }}

        div[data-testid="stForm"] {{
            background: var(--glass-bg) !important;
            backdrop-filter: blur(22px) !important;
            -webkit-backdrop-filter: blur(22px) !important;
            border-radius: 2.4vh !important;
            border: 1px solid var(--glass-border) !important;
            padding: 4.5vh 3.2vw 5.5vh 3.2vw !important;
            box-shadow: 0 4vh 8vh -2vh rgba(0,0,0,0.72) !important;
            width: 52vw !important;
            min-width: 480px;
            max-width: 780px;
            margin: auto;
        }}

        div[data-testid="stForm"] > div {{ border: none !important; }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] {{
            background-color: var(--input-bg) !important;
            border-radius: 0.9vh !important;
            border: 1px solid var(--glass-border) !important;
            transition: border-color 0.2s;
        }}

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: #6366f1 !important;
        }}

        input, textarea {{
            color: var(--text-main) !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        span[data-baseweb="tag"] {{
            background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(168,85,247,0.35)) !important;
            border: 1px solid rgba(129,140,248,0.4) !important;
            border-radius: 999px !important;
            color: #c7d2fe !important;
            font-size: 1.25vh !important;
            padding: 0.2vh 0.8vw !important;
        }}

        div[data-baseweb="datepicker"] input {{
            color: var(--text-main) !important;
        }}

        li[role="option"] {{
            background: #1e293b !important;
            color: var(--text-main) !important;
        }}
        li[role="option"]:hover {{
            background: rgba(99,102,241,0.25) !important;
        }}

        label, .stSelectbox label, .stMultiSelect label,
        .stDateInput label, .stRadio label p {{
            color: var(--text-muted) !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.4vh !important;
            font-weight: 500 !important;
            letter-spacing: 0.04em !important;
        }}

        .stButton button {{
            width: 100%;
            background: var(--primary-gradient) !important;
            color: white !important;
            border: none !important;
            padding: 1.4vh 1.5vw !important;
            border-radius: 0.9vh !important;
            font-weight: 600 !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.6vh !important;
            letter-spacing: 0.04em !important;
            transition: all 0.25s ease !important;
            margin-top: 1vh;
        }}

        .stButton button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 1.2vh 2.5vh -0.5vh rgba(99,102,241,0.55) !important;
        }}

        .section-label {{
            color: var(--accent);
            font-size: 1.15vh;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 2.8vh 0 1.2vh 0;
            display: flex;
            align-items: center;
            gap: 0.6vw;
        }}

        .section-label::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, rgba(129,140,248,0.35), transparent);
        }}

        .strength-wrap {{
            display: flex;
            gap: 0.4vw;
            margin: 0.6vh 0;
        }}

        .sbar {{
            flex: 1;
            height: 3px;
            border-radius: 999px;
            transition: background 0.3s ease;
        }}

        [data-testid="stAlert"] {{
            border-radius: 1vh !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            backdrop-filter: blur(8px) !important;
        }}

        [data-testid="stAlert"] * {{ color: #f8fafc !important; }}

        [data-testid="stAlert"][kind="error"]   {{ background: var(--error-bg) !important; }}
        [data-testid="stAlert"][kind="success"] {{ background: var(--success-bg) !important; }}

        #MainMenu, footer, header {{ visibility: hidden; }}
        div.stTextInput {{ margin-bottom: -0.5vh !important; }}

        .radio-inline div[data-testid="stRadio"] > div {{
            flex-direction: row !important;
            gap: 1.5vw;
        }}

        .radio-inline div[data-testid="stRadio"] label span {{
            color: var(--text-main) !important;
            font-size: 1.5vh !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Layout ─────────────────────────────────────────────────────────────────
    _, mid, _ = st.columns([1, 5, 1])

    with mid:
        with st.form("register_form", clear_on_submit=False):

            # Header con logo
            if logo_str:
                st.markdown(
                    f"""
                    <div style="text-align:center;margin-bottom:2.5vh;">
                        <img src="data:image/png;base64,{logo_str}"
                             alt="StreamVortex"
                             style="max-width:110px;width:27.5%;height:auto;" />
                    </div>
                    <p style="color:#94a3b8;font-size:1.7vh;text-align:center;margin-bottom:3vh;">
                        Crea tu cuenta y empieza a explorar
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="text-align:center;margin-bottom:2.5vh;">
                        <h2 style="color:#f8fafc;margin:0;">StreamVortex</h2>
                    </div>
                    <p style="color:#94a3b8;font-size:1.7vh;text-align:center;margin-bottom:3vh;">
                        Crea tu cuenta y empieza a explorar
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

            # ── SECTION: Cuenta ──────────────────────────────────────────────
            st.markdown('<div class="section-label">Datos de cuenta</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                username = st.text_input("Nombre de usuario", placeholder="ada_stream")
            with col_b:
                email = st.text_input("Correo electrónico", placeholder="ada@ejemplo.com")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            with col_p2:
                confirm_password = st.text_input("Confirmar contraseña", type="password", placeholder="••••••••")

            strength, strength_label, bar_colors = _password_strength(password)
            if strength_label:
                st.markdown(
                    f"""
                    <div class="strength-wrap">
                        {''.join(f'<div class="sbar" style="background:{c};"></div>' for c in bar_colors)}
                    </div>
                    <span style="color:{bar_colors[strength-1]};font-size:1.25vh;font-family:Outfit,sans-serif;">
                        Contraseña {strength_label}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

            # ── SECTION: Perfil ──────────────────────────────────────────────
            st.markdown('<div class="section-label">Perfil personal</div>', unsafe_allow_html=True)

            col_g, col_d = st.columns([1, 1])
            with col_g:
                gender = st.radio(
                    "Género",
                    options=["Hombre", "Mujer", "Otro"],
                    horizontal=True,
                )
            with col_d:
                birth_date = st.date_input(
                    "Fecha de nacimiento",
                    value=date(1995, 1, 1),
                    min_value=date(1920, 1, 1),
                    max_value=date.today().replace(year=date.today().year - 13),
                    format="DD/MM/YYYY",
                )

            # ── SECTION: Géneros ─────────────────────────────────────────────
            st.markdown(
                '<div class="section-label">Géneros favoritos <span style="font-size:1.1vh;color:#64748b;text-transform:none;letter-spacing:0;">(mín. 1)</span></div>',
                unsafe_allow_html=True,
            )

            favorite_genres = st.multiselect(
                "Selecciona tus géneros",
                options=GENRES,
                placeholder="Elige al menos un género...",
                label_visibility="collapsed",
            )

            submitted = st.form_submit_button("Crear mi cuenta →", use_container_width=True)

        # ── Validation & Save ──────────────────────────────────────────────────
        if submitted:
            errors = []

            if not username.strip():
                errors.append("El nombre de usuario es obligatorio.")
            elif not re.match(r"^[a-zA-Z0-9_\.]{3,30}$", username.strip()):
                errors.append("El usuario solo puede tener letras, números, puntos y guiones bajos (3-30 caracteres).")

            if not _valid_email(email):
                errors.append("Introduce un correo electrónico válido.")

            if len(password) < 6:
                errors.append("La contraseña debe tener al menos 6 caracteres.")
            if password != confirm_password:
                errors.append("Las contraseñas no coinciden.")

            if not favorite_genres:
                errors.append("Selecciona al menos un género favorito.")

            today = date.today()
            age = today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
            )
            if age < 13:
                errors.append("Debes tener al menos 13 años para registrarte.")

            if not errors:
                taken_user, taken_email = _username_or_email_exists(username, email)
                if taken_user:
                    errors.append("Ese nombre de usuario ya está en uso.")
                if taken_email:
                    errors.append("Ese correo electrónico ya está registrado.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                new_user = {
                    "user_id": str(_next_user_id()),
                    "username": username.strip(),
                    "email": email.strip().lower(),
                    "password": password,
                    "gender": gender,
                    "age": age,
                    "favorite_genres": "|".join(favorite_genres),
                    "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }

                if _save_user(new_user):
                    st.success(f"¡Cuenta creada con éxito! Bienvenido/a, {username.strip()}.")
                    st.balloons()

                    import time
                    time.sleep(1.5)

                    st.query_params["page"] = "login"
                    st.rerun()
                else:
                    st.error("No se pudo guardar el usuario. Verifica los permisos del fichero CSV.")

        if st.button("← Volver a inicio de sesión", use_container_width=True):
            st.query_params["page"] = "login"
            st.rerun()

    st.markdown(
        """
        <div style="position:fixed;bottom:2.5vh;width:100%;text-align:center;
                    color:#475569;font-size:1.35vh;font-family:'Outfit',sans-serif;">
            © 2026 StreamVortex Platform. Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True,
    )