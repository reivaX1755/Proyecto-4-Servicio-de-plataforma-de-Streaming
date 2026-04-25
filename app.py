# app.py
import streamlit as st

from src.pages.login import show_login
from src.pages.register import show_register
from src.pages.dashboard import show_dashboard
from src.pages.moviecard import show_movie_detail
from src.pages.audit import show_audit


def init_state():
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "selected_tmdb_id" not in st.session_state:
        st.session_state.selected_tmdb_id = None


def go_to_dashboard(user=None):
    st.session_state.logged_in = True
    st.session_state.current_user = user
    st.query_params["page"] = "dashboard"
    st.rerun()


def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.selected_tmdb_id = None
    st.query_params["page"] = "login"
    st.rerun()


def main():
    st.set_page_config(
        page_title="Streaming - Demo",
        layout="wide",
        page_icon="assets/logo-streaming.png",
    )

    init_state()

    page = st.query_params.get("page", "login")
    if isinstance(page, list):
        page = page[0] if page else "login"

    if page == "login":
        show_login(go_to_dashboard)
    elif page == "register":
        show_register()
    elif page == "dashboard":
        show_dashboard(logout)
    elif page == "moviecard":
        show_movie_detail()
    elif page == "audit":
        show_audit()
    else:
        show_login(go_to_dashboard)


if __name__ == "__main__":
    main()