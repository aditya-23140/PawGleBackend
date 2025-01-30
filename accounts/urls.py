from django.urls import path
from .views import RegisterView, LoginView, ProfileView, AddPetView, PublicPetDashboardView, DeletePetView, SearchPetView, EditPetView, GetPetCountView, GetUserCountView

urlpatterns = [
    path('signup/', RegisterView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('pets/add/', AddPetView.as_view(), name='add_pet'), 
    path('pets/search/', SearchPetView.as_view(), name='search_pet'), 
    path('dashboard/pets/', PublicPetDashboardView.as_view(), name='public_pet_dashboard'),
    path('pets/<int:pet_id>/delete/', DeletePetView.as_view(), name='delete_pet'),
    path('pets/<int:pet_id>/edit/', EditPetView.as_view(), name='edit_pet'),
    path('pets/count/', GetPetCountView.as_view(), name='get_pet_count'),  
    path('users/count/', GetUserCountView.as_view(), name='get_user_count'), 
]
