from django.contrib import admin
from django.urls import path
from django import views
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import asignar_grupo
from django.contrib.auth import views as auth_views
from .views import obtener_detalle_caso
from django.contrib.auth.views import LogoutView
from .views import generar_pdf

urlpatterns = [
    path('', views.index, name="index"),
    path('inicio', views.inicio, name="inicio"),
    path('asignar-grupo/', asignar_grupo, name='asignar_grupo'),
    path('listar_rol', views.listar_rol, name="listar_rol"),
    path('registrar', views.registrar, name="registrar"),
    path('listar', views.listar, name="listar"),
    path('agregar', views.agregar, name="agregar"),
    path('eliminar', views.eliminar, name="eliminar"),
    path('editar', views.editar, name="editar"),
    path('listar_huella', views.listar_huella, name="listar_huella"),
    path('agregar_huella', views.agregar_huella, name="agregar_huella"),
    path('eliminar_huella', views.eliminar_huella, name="eliminar_huella"),
    path('editar_huella', views.editar_huella, name="editar_huella"),
    path('listar_caso', views.listar_caso, name="listar_caso"),
    path('listar_implicado', views.listar_implicado, name="listar_implicado"),
    path('agregar_caso', views.agregar_caso, name="agregar_caso"),
    path('eliminar_caso', views.eliminar_caso, name="eliminar_caso"),
    path('agregar_implicado', views.agregar_implicado, name="agregar_implicado"),
    path('caso/editar/<int:id_caso>/', views.form_editar_caso, name="form_editar_caso"),
    path('caso/datos/<int:id_caso>/', views.obtener_datos_caso, name='obtener_datos_caso'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('caso/cambiar_estado/<int:id_caso>/', views.cambiar_estado_caso, name='cambiar_estado_caso'),
    path('huella/cambiar_estado_huella/<int:id_caso>/', views.cambiar_estado_huella, name='cambiar_estado_huella'),
    path('obtener-detalle-caso/', obtener_detalle_caso, name='obtener_detalle_caso'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('procesar_imagen/', views.procesar_imagen, name="procesar_imagen"),

    path('generar-pdf/', views.generar_pdf, name='generar_pdf'),
   
    
    path('comparar_huella', views.comparar_huella, name="comparar_huella"),
    path('ruta-para-obtener-imagenes/<int:huella_id>/', views.obtener_imagenes_huella, name='obtener_imagenes_huella')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)