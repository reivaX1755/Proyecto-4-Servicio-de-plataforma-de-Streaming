import streamlit as st
import base64
import os
import re


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def show_register():
    """
    Pantalla de registro para StreamVortex.
    Diseño coherente con show_login: glassmorphism oscuro, gradiente indigo-púrpura,
    tipografía Outfit, fondo con imagen + overlay.
    """

    # ── 1. Fondo ──────────────────────────────────────────────────────────────
    bg_img_path = os.path.join("assets", "fondo-login.png")
    bin_str = ""
    try:
        bin_str = get_base64_of_bin_file(bg_img_path)
    except Exception:
        pass  # Fallback to gradient if file is missing

    # ── 2. CSS ────────────────────────────────────────────────────────────────
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

        .register-header h1 {{
            font-size: 4.5vh;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5vh;
        }}

        .register-header p {{
            color: var(--text-muted);
            font-size: 1.8vh;
            margin-bottom: 3vh;
        }}

        div[data-baseweb="input"] {{
            background-color: rgba(15, 23, 42, 0.8) !important;
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
            background: rgba(30, 41, 59, 0.75) !important;
            backdrop-filter: blur(2vh) !important;
            border-radius: 3vh !important;
            border: 0.2vh solid rgba(255, 255, 255, 0.15) !important;
            padding: 5vh 3vw 6vh 3vw !important;
            box-shadow: 0 4vh 7vh -2vh rgba(0, 0, 0, 0.7) !important;
            width: 38vw !important;
            min-width: 380px;
            margin: auto;
        }}

        div.stTextInput {{
            margin-bottom: -1vh !important;
        }}

        div[data-testid="stForm"] > div {{
            border: none !important;
        }}

        .strength-bar-container {{
            display: flex;
            gap: 0.4vw;
            margin-top: 0.8vh;
            margin-bottom: 1vh;
        }}

        .strength-bar {{
            flex: 1;
            height: 0.4vh;
            border-radius: 999px;
            transition: background 0.3s ease;
        }}

        .footer-links {{
            margin-top: 3vh;
            text-align: center;
            color: var(--text-muted);
            font-size: 1.6vh;
        }}

        .footer-links a {{
            color: #818cf8;
            text-decoration: none;
            font-weight: 600;
        }}

        .terms-text {{
            color: var(--text-muted);
            font-size: 1.3vh;
            margin-top: 1.5vh;
            text-align: center;
        }}

        .terms-text a {{
            color: #818cf8;
            text-decoration: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── 3. Layout ──────────────────────────────────────────────────────────────
    left, mid, right = st.columns([1, 4, 1])

    with mid:
        with st.form("register_form", clear_on_submit=False):
            st.markdown(
                """
                <div class="register-header">
                    <h1>StreamVortex</h1>
                    <p>Crea tu cuenta y empieza a explorar</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                full_name = st.text_input("Nombre completo", placeholder="Ada Lovelace")
            with col_b:
                username = st.text_input("Nombre de usuario", placeholder="@ada_stream")

            email = st.text_input("Correo electrónico", placeholder="ada@ejemplo.com")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                password = st.text_input(
                    "Contraseña", type="password", placeholder="••••••••"
                )
            with col_p2:
                confirm_password = st.text_input(
                    "Confirmar contraseña", type="password", placeholder="••••••••"
                )

            strength, strength_label, bar_colors = _password_strength(password)
            if strength_label:  # Solo mostrar barra si hay texto
                st.markdown(
                    f"""
                    <div>
                        <div class="strength-bar-container">
                            <div class="strength-bar" style="background:{bar_colors[0]};"></div>
                            <div class="strength-bar" style="background:{bar_colors[1]};"></div>
                            <div class="strength-bar" style="background:{bar_colors[2]};"></div>
                            <div class="strength-bar" style="background:{bar_colors[3]};"></div>
                        </div>
                        <span style="color:{bar_colors[strength-1] if strength else '#334155'};font-size:1.3vh;">
                            {strength_label}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <p class="terms-text">
                    Al crear una cuenta aceptas los
                    <a href="#">Términos de Servicio</a> y la
                    <a href="#">Política de Privacidad</a> de StreamVortex.
                </p>
                """,
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button("Crear mi cuenta", use_container_width=True)

        if submitted:
            errors = []
            if not full_name.strip():
                errors.append("El nombre completo es obligatorio.")
            if not username.strip():
                errors.append("El nombre de usuario es obligatorio.")
            if not _valid_email(email):
                errors.append("Introduce un correo electrónico válido.")
            if len(password) < 6:
                errors.append("La contraseña debe tener al menos 6 caracteres.")
            if password != confirm_password:
                errors.append("Las contraseñas no coinciden.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                if "users" not in st.session_state:
                    st.session_state.users = {}

                st.session_state.users[username.strip()] = {
                    "password": password,
                    "email": email,
                    "full_name": full_name,
                }

                st.success(f"¡Cuenta creada con éxito! Bienvenido/a, {full_name.split()[0]}. 🎉")
                st.balloons()

                import time
                time.sleep(1.5)

                st.session_state.page = "login"
                st.rerun()


        if st.button("Volver a inicio de sesión", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

    st.markdown(
        """
        <div style="position:fixed;bottom:3vh;width:100%;text-align:center;
                    color:#94a3b8;font-size:1.4vh;opacity:0.7;">
            © 2026 StreamVortex Platform. Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _valid_email(email: str) -> bool:
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def _password_strength(password: str):
    """
    Devuelve (nivel:1-4, etiqueta, lista_colores_4_barras).
    Nivel 0 = vacío.
    """
    EMPTY = "#1e293b"
    RED = "#ef4444"
    ORANGE = "#f97316"
    YELLOW = "#eab308"
    GREEN = "#22c55e"

    if not password:
        return 0, "", [EMPTY, EMPTY, EMPTY, EMPTY]

    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"[0-9]", password):
        score += 1
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1

    level = max(1, score)

    palettes = {
        1: ("Débil", [RED, EMPTY, EMPTY, EMPTY]),
        2: ("Regular", [ORANGE, ORANGE, EMPTY, EMPTY]),
        3: ("Buena", [YELLOW, YELLOW, YELLOW, EMPTY]),
        4: ("Fuerte", [GREEN, GREEN, GREEN, GREEN]),
    }
    label, colors = palettes[level]
    return level, label, colors