from django.shortcuts import render, redirect ,HttpResponse
from django.contrib import messages
from .forms import LoginForm, RegisterForm
from .models import LoginDetails
from django.contrib.auth.hashers import make_password, check_password
from django.utils.timezone import now

from django.core.mail import send_mail
from django.conf import settings

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('user_name')
        password = request.POST.get('password')
        try:
            user = LoginDetails.objects.get(user_name=username)
            if check_password(password, user.password):
                request.session['user_id'] = user.user_id
                request.session['user_name'] = user.user_name

                # Send email notification
                send_mail(
                    subject='Login Alert - Indian Sign Language Arena',
                    message=f'Hi {user.user_name},\n\nYou have successfully logged in to the Indian Sign Language Arena.',
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email],
                    fail_silently=False,
                )

                return redirect('home')
            else:
                messages.error(request, 'Invalid password.')
        except LoginDetails.DoesNotExist:
            messages.error(request, 'User does not exist.')
    return render(request, 'login.html')

 # ✅ stay on login page


def user_register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.password = make_password(form.cleaned_data['password'])
            user.status = 'Active'  # make sure status is set if needed
            user.save()
            messages.success(request, 'Account created successfully. Please login.')
            return redirect('user_login')  # ✅ go to login page
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def home(request):
    return render(request, 'home.html')  # ✅ show home

def user_logout(request):
    request.session.flush()
    return redirect('user_login')  # ✅ go back to login page

def admin_logout(request):
    return redirect('admin_login') 

# views.py
from django.shortcuts import render
from .models import GestureVideo

def video_gallery(request):
    videos = GestureVideo.objects.filter(status='Active').order_by('gesture_type', 'gesture')
    return render(request, 'video_gallery.html', {'videos': videos})


# views.py
from django.shortcuts import render, redirect
from .forms import GestureVideoForm

def upload_video(request):
    if request.method == 'POST':
        form = GestureVideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_gallery')
    else:
        form = GestureVideoForm()
    return render(request, 'upload_video.html', {'form': form})

from django.shortcuts import render
import os
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

# --- Load and preprocess the knowledge base ---
def split_text_to_chunks(text, chunk_size=3):
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence-ending punctuation
    chunks = []
    for i in range(0, len(sentences), chunk_size - 1):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def load_knowledge(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return split_text_to_chunks(text)

# --- Initialize embeddings and model once ---
FILE_PATH = os.path.join(settings.BASE_DIR, 'static', 'knowledge.txt')
chunks = load_knowledge(FILE_PATH)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_tensor=True)  # Tensor format (on MPS or CUDA)

# --- Chatbot view ---
def chatbot(request):
    response = ""
    if request.method == 'POST':
        query = request.POST.get('query')
        if query:
            # Encode the query
            query_embedding = model.encode([query], convert_to_tensor=True)

            # Convert both query and embeddings to CPU numpy for sklearn
            query_np = query_embedding.cpu().numpy()
            embeddings_np = embeddings.cpu().numpy()

            # Calculate similarity
            similarity = cosine_similarity(query_np, embeddings_np)[0]

            # Get top 3 matches with similarity score > 0.1
            top_indices = similarity.argsort()[-3:][::-1]
            top_matches = [chunks[i] for i in top_indices if similarity[i] > 0.1]

            if top_matches:
                # Limit the response to 2 or 3 sentences
                response = " ".join(top_matches[:1])  # Take the first match
                sentences = response.split('. ')
                if len(sentences) > 2:
                    response = '. '.join(sentences[:2])  # Limit to the first 2 sentences
                else:
                    response = '. '.join(sentences[:3])  # Limit to the first 3 sentences
            else:
                response = "Sorry, I couldn't find a relevant answer."
    
    return render(request, 'chatbot.html', {'response': response})



# predictor/views.py

import os
import cv2
import numpy as np
import mediapipe as mp
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model

# Load your model
model1 = load_model(os.path.join(settings.BASE_DIR, 'predictor', 'season.h5'))
model2 = load_model(os.path.join(settings.BASE_DIR, 'predictor', 'clothes.h5'))
model3 = load_model(os.path.join(settings.BASE_DIR, 'predictor', 'electronics.h5'))


# ✅ Define your class labels in the same order used during training
MODEL_MAP = {
    'season': {
        'model': model1,
        'labels': ['Fall', 'Monsoon', 'Season', 'Spring', 'Summer', 'Winter']
    },
    'clothes': {
        'model': model2,
        'labels': ['Clothing', 'Dress', 'Hat', 'Pant', 'Pocket', 'Shirt', 'Shoes', 'Skirt', 'Suit', 'T-Shirt']
    },
    'electronics': {
        'model': model3,
        'labels': ['Camera', 'Cell phone', 'Clock', 'Computer', 'Fan', 'Lamp', 'Laptop', 'Radio', 'Screen', 'Television']
    }
}# Replace this with your actual list

# MediaPipe setup
mp_holistic = mp.solutions.holistic

def extract_frames(video_path, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def extract_keypoints(image):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []

        if results.pose_landmarks:
            keypoints.extend(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())
        else:
            keypoints.extend([0] * 132)

        if results.face_landmarks:
            keypoints.extend(np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten())
        else:
            keypoints.extend([0] * 1404)

        if results.left_hand_landmarks:
            keypoints.extend(np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten())
        else:
            keypoints.extend([0] * 63)

        if results.right_hand_landmarks:
            keypoints.extend(np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten())
        else:
            keypoints.extend([0] * 63)

        return np.array(keypoints)

def pad_sequence(sequence, max_frames, max_features):
    if sequence.shape[0] < max_frames:
        padding = np.zeros((max_frames - sequence.shape[0], max_features))
        sequence = np.vstack((sequence, padding))
    else:
        sequence = sequence[:max_frames]
    return np.expand_dims(sequence, axis=0)

def predict_action(request):
    if request.method == 'POST' and request.FILES.get('video'):
        model_choice = request.POST.get('model_choice')

        if model_choice not in MODEL_MAP:
            return HttpResponse("Invalid model selection.", status=400)

        selected_model = MODEL_MAP[model_choice]['model']
        class_labels = MODEL_MAP[model_choice]['labels']

        video_file = request.FILES['video']
        video_path = default_storage.save('tmp/' + video_file.name, video_file)
        video_full_path = os.path.join(settings.MEDIA_ROOT, video_path)

        frames = extract_frames(video_full_path)
        keypoints_sequence = [extract_keypoints(frame) for frame in frames]
        keypoints_sequence = np.array(keypoints_sequence)

        max_frames = selected_model.input_shape[1]
        max_features = selected_model.input_shape[2]
        keypoints_padded = pad_sequence(keypoints_sequence, max_frames, max_features)

        prediction = selected_model.predict(keypoints_padded)[0]
        top_3_indices = prediction.argsort()[-3:][::-1]
        top_3_results = [(class_labels[i], float(prediction[i])) for i in top_3_indices]

        return render(request, 'result.html', {
            'top_predictions': top_3_results,
            'model_used': model_choice.capitalize()
        })

    return render(request, 'upload.html')


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:  # Ensure the user is an admin
            login(request, user)
            return redirect('dashboard')  # Redirect to the admin dashboard
        else:
            messages.error(request, "Invalid username or password")
    
    return render(request, 'admin_login.html')  # Render the login page



from datetime import datetime
# Create a custom dashboard view
def dashboard(request):
    # Query LoginDetails and GestureVideo models
    login_details = LoginDetails.objects.all()
    gesture_videos = GestureVideo.objects.all()
    
    # Count active and inactive users
    active_count = login_details.filter(status='Active').count()
    inactive_count = login_details.filter(status='Inactive').count()
    
    # Get current date
    current_date = datetime.now().date()
    
    # Render the dashboard page with the queried data
    return render(request, 'dashboard.html', {
        'login_details': login_details,
        'gesture_videos': gesture_videos,
        'active_users_count': active_count,
        'inactive_users_count': inactive_count,
        'current_date': current_date,
    })


from django.shortcuts import render, get_object_or_404, redirect
from .models import LoginDetails, GestureVideo
from .forms import LoginDetailForm, GestureVideoForm

# Dashboard view
def ashboard(request):
    login_details = LoginDetails.objects.all()
    gesture_videos = GestureVideo.objects.all()
    return render(request, 'admin_dashboard.html', {
        'login_details': login_details,
        'gesture_videos': gesture_videos
    })


# LoginDetails Views
def add_login_detail(request):
    if request.method == 'POST':
        form = LoginDetailForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = LoginDetailForm()
    return render(request, 'login_form.html', {'form': form, 'title': 'Add Login Detail'})


def edit_login_detail(request, user_id):
    login = get_object_or_404(LoginDetails, pk=user_id)
    if request.method == 'POST':
        form = LoginDetailForm(request.POST, instance=login)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = LoginDetailForm(instance=login)
    return render(request, 'login_form.html', {'form': form, 'title': 'Edit Login Detail'})


def delete_login_detail(request, user_id):
    login = get_object_or_404(LoginDetails, pk=user_id)
    login.delete()
    return redirect('dashboard')


# GestureVideo Views
def add_gesture_video(request):
    if request.method == 'POST':
        form = GestureVideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = GestureVideoForm()
    return render(request, 'gesture_form.html', {'form': form, 'title': 'Add Gesture Video'})


def edit_gesture_video(request, serial_no):
    gesture = get_object_or_404(GestureVideo, pk=serial_no)
    if request.method == 'POST':
        form = GestureVideoForm(request.POST, request.FILES, instance=gesture)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = GestureVideoForm(instance=gesture)
    return render(request, 'gesture_form.html', {'form': form, 'title': 'Edit Gesture Video'})


def delete_gesture_video(request, serial_no):
    gesture = get_object_or_404(GestureVideo, pk=serial_no)
    gesture.delete()
    return redirect('dashboard')
