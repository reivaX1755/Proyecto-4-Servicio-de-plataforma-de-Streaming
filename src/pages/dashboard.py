import streamlit as st

def show_dashboard(logout):
    st.header("Dashboard principal")
    if st.session_state.current_user:
        st.subheader(f"Bienvenido — {st.session_state.current_user}")

    st.sidebar.title("Navegación")
    if st.sidebar.button("Cerrar sesión"):
        logout()

    st.sidebar.markdown("""
    **Secciones (esqueleto)**
    - Inicio
    - Recomendaciones
    - Mi lista
    - Ajustes
    """)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Área principal")
        st.info("Aquí se mostrará el reproductor y lista de contenidos.")
    with right:
        st.subheader("Panel lateral")
        st.text("Estadísticas (placeholders)")
        st.metric("Usuarios activos", "—")
        st.metric("Reproducciones/h", "—")

    st.markdown("---")
    st.caption("Esqueleto del dashboard.")
