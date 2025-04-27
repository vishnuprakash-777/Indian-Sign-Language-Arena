# islapp/apps.py
from django.apps import AppConfig

class IslappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'islapp'

    def ready(self):
        import islapp.signals  # 👈 ensures the signal is connected
