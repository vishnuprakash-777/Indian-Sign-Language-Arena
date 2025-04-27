from django import forms
from .models import LoginDetails
from django.core.exceptions import ValidationError

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150, required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)

class RegisterForm(forms.ModelForm):
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        label="Confirm Password"
    )

    class Meta:
        model = LoginDetails
        fields = ['user_name', 'email', 'contact_no', 'password']
        widgets = {
            'user_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'contact_no': forms.TextInput(attrs={'class': 'form-control'}),
            'password': forms.PasswordInput(attrs={'class': 'form-control'}),
        }

    def clean_user_name(self):
        username = self.cleaned_data['user_name']
        if LoginDetails.objects.filter(user_name=username).exists():
            raise ValidationError("This username is already taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data['email']
        if LoginDetails.objects.filter(email=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def clean_contact_no(self):
        contact_no = self.cleaned_data['contact_no']
        if not contact_no.isdigit():
            raise ValidationError("Contact number must be numeric.")
        if len(contact_no) < 10:
            raise ValidationError("Contact number is too short.")
        return contact_no

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm = cleaned_data.get("confirm_password")

        if password and confirm and password != confirm:
            self.add_error('confirm_password', "Passwords do not match.")


# forms.py
from django import forms
from .models import GestureVideo



from django import forms
from .models import LoginDetails, GestureVideo

class LoginDetailForm(forms.ModelForm):
    class Meta:
        model = LoginDetails
        fields = ['user_name', 'password', 'email', 'contact_no', 'status', 'last_login']
        widgets = {
            'password': forms.PasswordInput(),
            'last_login': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }

class GestureVideoForm(forms.ModelForm):
    class Meta:
        model = GestureVideo
        fields = ['gesture', 'gesture_type', 'video_file', 'status']
