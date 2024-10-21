import streamlit as st
from streamlit_option_menu import option_menu

import about

st.set_page_config(
    page_title="Recommender System for Courses",
)

class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func,
        })

    def run(self):
        # Mengatur nilai default menu menjadi "Home"
        default_index = 0
        
        with st.sidebar:
            app = option_menu(
                menu_title='Recommender System',
                options=['Home','about'],
                icons=['house-fill','info-circle-fill'],
                default_index=default_index,  # Menggunakan nilai default_index
        styles={
            "container": {"padding": "5!important", "background": "#881a27"},
            "icon": {"color": "white", "font-size": "23px"}, 
            "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "EEE58C"},
            "nav-link-selected": {"background": "#FEF9F2"}
        }

            )

        # if app == "Home":
        #     main.app()
        if app == 'about':
            about.app()

# Membuat instance MultiApp dan menjalankan aplikasi
multi_app = MultiApp()
multi_app.run()
