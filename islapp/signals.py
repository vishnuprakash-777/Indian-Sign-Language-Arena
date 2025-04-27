# islapp/signals.py
from allauth.account.signals import user_logged_in
from django.dispatch import receiver
from django.core.mail import send_mail
from django.conf import settings

@receiver(user_logged_in)
def send_login_email(request, user, **kwargs):
    print(f"📩 Sending email to {user.email} after Google login...")
    send_mail(
            subject='Google Login - Indian Sign Language Arena',
            message=f'Hi {user.username},\n\nYou have successfully logged in using Google.',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
