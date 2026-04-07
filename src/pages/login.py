import streamlit as st
import base64
import os
from pathlib import Path
import csv
import time


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def _load_users_from_csv():
    """
    Carga los usuarios desde:
    proyecto/data/users.csv
    """
    csv_path = Path(__file__).resolve().parents[2] / "data" / "users.csv"

    if not csv_path.exists():
        return []

    users = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                users.append(row)
    except Exception:
        return []

    return users


def _authenticate_user(identifier: str, password: str):
    """
    Valida si existe:
    - username + password
    o
    - email + password

    Devuelve el usuario completo si hay coincidencia, o None si no.
    """
    identifier = identifier.strip().lower()
    password = password.strip()

    users = _load_users_from_csv()

    for user in users:
        username = (user.get("username") or "").strip().lower()
        email = (user.get("email") or "").strip().lower()
        stored_password = (user.get("password") or "").strip()

        if stored_password != password:
            continue

        if identifier == username or identifier == email:
            return user

    return None


def show_login(go_to_dashboard):
    # Si ya está logueado, va directo al dashboard
    if st.session_state.get("logged_in"):
        go_to_dashboard(user=st.session_state.get("user"))
        return

    bg_img_path = os.path.join("assets", "fondo-login.png")
    bin_str = ""
    try:
        bin_str = get_base64_of_bin_file(bg_img_path)
    except Exception:
        pass

    # CSS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        :root {{
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --background-dark: #0f172a;
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --error-bg: rgba(127, 29, 29, 0.92);
            --success-bg: rgba(20, 83, 45, 0.92);
        }}

        .stApp {{
            background: linear-gradient(rgba(15, 23, 42, 0.6), rgba(15, 23, 42, 0.6)),
                        url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Outfit', sans-serif;
            overflow: hidden;
        }}

        .main .block-container {{
            max-width: 100% !important;
            padding: 0 !important;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}

        .login-header h1 {{
            font-size: 4.5vh;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1vh;
        }}

        .login-header p {{
            color: var(--text-muted);
            font-size: 1.8vh;
            margin-bottom: 4vh;
        }}

        div[data-baseweb="input"] {{
            background-color: rgba(15, 23, 42, 0.82) !important;
            border-radius: 1vh !important;
            border: 0.1vh solid var(--glass-border) !important;
        }}

        input {{
            color: var(--text-main) !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        .stButton button {{
            width: 100%;
            background: var(--primary-gradient) !important;
            color: white !important;
            border: none !important;
            padding: 1.5vh 1.5vw !important;
            border-radius: 1vh !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            margin-top: 2vh;
        }}

        .stButton button:hover {{
            transform: translateY(-0.2vh);
            box-shadow: 0 1vh 2vh -0.5vh rgba(99, 102, 241, 0.5);
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        div[data-testid="stForm"] {{
            background: rgba(30, 41, 59, 0.78) !important;
            backdrop-filter: blur(2vh) !important;
            border-radius: 3vh !important;
            border: 0.2vh solid rgba(255, 255, 255, 0.15) !important;
            padding: 5vh 3vw 7vh 3vw !important;
            box-shadow: 0 4vh 7vh -2vh rgba(0, 0, 0, 0.7) !important;
            width: 35vw !important;
            min-width: 350px;
            margin: auto;
        }}

        div.stTextInput {{
            margin-bottom: -1vh !important;
        }}

        div[data-testid="stForm"] > div {{
            border: none !important;
        }}

        .footer-links {{
            margin-top: 4vh;
            text-align: center;
            color: var(--text-muted);
            font-size: 1.6vh;
        }}

        /* ALERTAS MÁS VISIBLES */
        [data-testid="stAlert"] {{
            border-radius: 1.1vh !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            backdrop-filter: blur(1vh) !important;
            opacity: 0.98 !important;
            box-shadow: 0 1.5vh 3vh -1vh rgba(0,0,0,0.5) !important;
        }}

        [data-testid="stAlert"] * {{
            color: #f8fafc !important;
        }}

        [data-testid="stAlert"][kind="error"] {{
            background: var(--error-bg) !important;
        }}

        [data-testid="stAlert"][kind="success"] {{
            background: var(--success-bg) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    left, mid, right = st.columns([1, 4, 1])

    with mid:
        with st.form("login_form", clear_on_submit=False):
            st.markdown(
                """
                <div class="login-header">
                    <h1>StreamVortex</h1>
                    <p>Tu universo de streaming ilimitado</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            username_or_email = st.text_input("Usuario o Email", placeholder="usuario@ejemplo.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")

            submitted = st.form_submit_button("Entrar a la experiencia")

        if submitted:
            if not username_or_email.strip() or not password.strip():
                st.error("Campos obligatorios.")
            else:
                matched_user = _authenticate_user(username_or_email, password)

                if matched_user:
                    display_name = (
                        matched_user.get("username")
                        or matched_user.get("email")
                        or username_or_email.strip()
                    )

                    # Guardar estado de login
                    st.session_state.logged_user = matched_user
                    st.session_state.user = display_name
                    st.session_state.logged_in = True
                    st.session_state.page = 'dashboard'

                    # Mostrar feedback visual antes de redirigir
                    st.success(f"¡Bienvenido, {display_name}!")
                    st.balloons()
                    time.sleep(0.5)

                    # Redirigir al dashboard
                    st.rerun()
                else:
                    st.error("Usuario/email o contraseña incorrectos.")

        # Footer texto plano
        st.markdown(
            """
            <div class="footer-links">
                ¿Aún no eres miembro? Únete ahora
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Crear una nueva cuenta", type="secondary", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()

    # Footer fijo
    st.markdown(
        """
        <div style="position: fixed; bottom: 3vh; width: 100%; text-align: center; color: #94a3b8; font-size: 1.4vh; opacity: 0.7;">
            © 2026 StreamVortex Platform. Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True
    )