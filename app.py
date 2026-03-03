# app.py
"""
Controlador principal para la app de streaming.
Gestiona la navegación entre páginas separadas: login, register y dashboard.
"""

import streamlit as st
from src.pages.login import show_login
from src.pages.register import show_register
from src.pages.dashboard import show_dashboard

# --- Inicialización del estado de la sesión -------------------------------------------------
def init_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'users' not in st.session_state:
        st.session_state.users = {}
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

# --- Navegación simple ---------------------------------------------------------------------
def go_to_dashboard(user=None):
    st.session_state.logged_in = True
    st.session_state.current_user = user
    st.session_state.page = 'dashboard'

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.page = 'login'

# --- Main ----------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Streaming - Demo", layout="wide")
    init_state()

    if st.session_state.page == 'login':
        show_login(go_to_dashboard)
    elif st.session_state.page == 'register':
        show_register()
    elif st.session_state.page == 'dashboard':
        show_dashboard(logout)
    else:
        show_login(go_to_dashboard)

if __name__ == '__main__':
    main()