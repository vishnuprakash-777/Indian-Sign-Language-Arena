from django.contrib import admin
from .models import LoginDetails,GestureVideo

@admin.register(LoginDetails)
class LoginDetailsAdmin(admin.ModelAdmin):
    list_display = ('user_name', 'email', 'contact_no', 'status', 'last_login')
    search_fields = ('user_name', 'email')
    list_filter = ('status',)



from django.contrib import admin
from .models import GestureVideo

@admin.register(GestureVideo)
class GestureVideoAdmin(admin.ModelAdmin):
    # Columns to display in the list view
    list_display = ('serial_no', 'gesture', 'gesture_type', 'video_file')
    
    # Fields you can filter by in the sidebar
    list_filter = ('status',)
    
    # Fields that can be searched
    search_fields = ('gesture', 'gesture_type')
    
    
from django.contrib import admin
from .models import LoginAdminDetails
@admin.register(LoginAdminDetails)
# Register your model here
class LoginAdminDetailsAdmin(admin.ModelAdmin):
    list_display = ('username', 'status', 'last_login')  # Fields to display in the list view
    search_fields = ('username',)  # Fields to search in the admin panel

# Register the model to make it available in the Django Admin interface

