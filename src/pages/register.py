import streamlit as st

def show_register():
    st.title("Registro")
    st.write("Pantalla de registro — inputs sin controles obligatorios.")

    new_user = st.text_input("Nombre de usuario", key='reg_username')
    new_pass = st.text_input("Contraseña", type='password', key='reg_password')

    if st.button("Crear cuenta"):
        if new_user:
            st.session_state.users[new_user] = {'password': new_pass}
            st.success("Cuenta creada (simulada). Vuelve a iniciar sesión.")
            st.session_state.page = 'login'
        else:
            st.warning("Introduce un nombre de usuario para registrar.")

    if st.button("Volver al inicio de sesión"):
        st.session_state.page = 'login'
