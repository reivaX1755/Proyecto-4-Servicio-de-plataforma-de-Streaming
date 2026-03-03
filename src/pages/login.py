import streamlit as st
import base64
import os

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def show_login(go_to_dashboard):
    """
    Login profesional y elegante para una plataforma de streaming.
    """

    # 1. Obtener imagen de fondo
    bg_img_path = os.path.join("assets", "fondo-login.png")
    bin_str = ""
    try:
        bin_str = get_base64_of_bin_file(bg_img_path)
    except Exception:
        pass # Fallback to gradient if file is missing

    # 2. Configuración de estilo avanzada con CSS
    st.markdown(
        f"""
        <style>
        /* Importar tipografía de Google */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* Variables de diseño */
        :root {{
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --background-dark: #0f172a;
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        /* Fondo de la aplicación con imagen y overlay */
        .stApp {{
            background: linear-gradient(rgba(15, 23, 42, 0.6), rgba(15, 23, 42, 0.6)), 
                        url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Outfit', sans-serif;
            overflow: hidden;
        }}

        /* Centrado absoluto */
        .main .block-container {{
            max-width: 100% !important;
            padding: 0 !important;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}

        /* Títulos mas compactos */
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

        /* Estilizado de inputs de Streamlit */
        div[data-baseweb="input"] {{
            background-color: rgba(15, 23, 42, 0.8) !important;
            border-radius: 1vh !important;
            border: 0.1vh solid var(--glass-border) !important;
        }}
        input {{
            color: var(--text-main) !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        /* Botón de Submit */
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

        /* Ocultar elementos de Streamlit */
        #MainMenu, footer, header {{ visibility: hidden; }}

        /* Estilo de la tarjeta vía stForm */
        div[data-testid="stForm"] {{
            background: rgba(30, 41, 59, 0.75) !important;
            backdrop-filter: blur(2vh) !important;
            border-radius: 3vh !important;
            border: 0.2vh solid rgba(255, 255, 255, 0.15) !important;
            padding: 5vh 3vw 8vh 3vw !important; /* Más alto por debajo */
            box-shadow: 0 4vh 7vh -2vh rgba(0, 0, 0, 0.7) !important;
            width: 35vw !important;
            min-width: 350px;
            margin: auto;
        }}
        
        div.stTextInput {{ margin-bottom: -1vh !important; }}
        div.stCheckbox {{ margin-top: -0.5vh !important; }}
        div[data-testid="stForm"] > div {{ border: none !important; }}

        .footer-links {{
            margin-top: 4vh;
            text-align: center;
            color: var(--text-muted);
            font-size: 1.6vh;
        }}
        .footer-links a {{
            color: #818cf8;
            text-decoration: none;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Contenedor centrado
    left, mid, right = st.columns([1, 4, 1])

    with mid:
        # Formulario
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

            username = st.text_input("Usuario o Email", placeholder="usuario@ejemplo.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                remember = st.checkbox("Recordarme")
            with c2:
                st.markdown('<div style="text-align:right; padding-top:1.2vh;"><a href="#" style="color:#818cf8; font-size:1.4vh; text-decoration:none;">¿Olvidaste tu contraseña?</a></div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Entrar a la experiencia")

        if submitted:
            if username.strip() and password.strip():
                st.success(f"¡Bienvenido, {username}!")
                st.balloons()
                import time
                time.sleep(1)
                go_to_dashboard(user=username)
            else:
                st.error("Campos obligatorios.")

        # Registro
        st.markdown(
            """
            <div class="footer-links">
                ¿Aún no eres miembro? 
                <a href="javascript:void(0);" onclick="parent.window.location.hash = 'register'">Únete ahora</a>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        if st.button("Crear una nueva cuenta", type="secondary", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 3vh; width: 100%; text-align: center; color: #94a3b8; font-size: 1.4vh; opacity: 0.7;">
            © 2026 StreamVortex Platform. Todos los derechos reservados.
        </div>
        """,
        unsafe_allow_html=True
    )


    