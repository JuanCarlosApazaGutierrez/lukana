from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Personas(models.Model):
    id_persona = models.AutoField(primary_key=True)
    usuario = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    nombre = models.CharField(max_length=30, null=False)
    ap_paterno = models.CharField(max_length=30, null=False)
    ap_materno = models.CharField(max_length=30, null=False)
    ci = models.IntegerField(null=False)
    f_registro = models.DateTimeField(auto_now_add=True, null=True)
    activate = models.BooleanField(default=True)
    cede = models.CharField(max_length=30, null=True)
    cargo = models.CharField(max_length=30, null=True)

    class Meta:
        db_table = 'personas'


class Casos(models.Model):
    id_caso = models.AutoField(primary_key=True)
    codigo_caso = models.IntegerField(null=True)
    f_registro = models.DateTimeField(auto_now_add=True, null=True)
    f_modificacion = models.DateTimeField(auto_now_add=True, null=True)
    sospechosos = models.IntegerField(null=True)
    descripcion = models.CharField(max_length=200,null=False)
    nombre_caso = models.CharField(max_length=30,null=False)
    tipo = models.CharField(max_length=30,null=True)
    departamento = models.CharField(max_length=30,null=True)
    activate = models.BooleanField(default=True)
    perito = models.IntegerField(null=True)
    class Meta:
        db_table = 'casos'





class Implicado(models.Model):
    id_implicado = models.AutoField(primary_key=True)
    caso = models.IntegerField(null=True)
    nombres = models.CharField(max_length=50, null=False, default='')
    apellido_paterno = models.CharField(max_length=50, null=False, default='')
    apellido_materno = models.CharField(max_length=50, null=False, default='')
    carnet_identidad = models.CharField(max_length=20, null=False, default='', unique=True)  # Aquí hacemos el cambio
    expedido_en = models.CharField(max_length=30, null=False, default='')
    fecha_nacimiento = models.DateField(auto_now_add=True, null=True)
    f_registro = models.DateTimeField(auto_now_add=True, null=True)
    lugar_nacimiento = models.CharField(max_length=100, null=False, default='')
    direccion = models.CharField(max_length=100, null=False, default='')
    nacionalidad = models.CharField(max_length=30, null=False, default='')
    ocupacion = models.CharField(max_length=50, null=False, default='')
    color_pelo = models.CharField(max_length=30, null=False, default='')
    color_piel = models.CharField(max_length=30, null=False, default='')
    peso_kg = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    altura_m = models.DecimalField(max_digits=4, decimal_places=2, null=True)
    sexo = models.CharField(max_length=50, null=False, default='')
    activo = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'implicados'

    def __str__(self):
        return f"{self.nombres} {self.apellido_paterno}"
    

class Huellas(models.Model):
    id_huella = models.AutoField(primary_key=True)
    f_registro = models.DateTimeField(auto_now_add=True, null=True)
    caso = models.IntegerField(null=True)
    original_image = models.ImageField(upload_to='original/', null=True, blank=True)
    procesada_image = models.ImageField(upload_to='procesada/', null=True, blank=True)
    minutiae_image = models.ImageField(upload_to='minucias/', null=True, blank=True) 
    minutiae_csv = models.FileField(upload_to='datos/', null=True, blank=True) 
    activate = models.BooleanField(default=True)
    persona = models.IntegerField(null=True)
    
    tipo = models.CharField(max_length=30,null=True)
    class Meta:
        db_table = 'huellas'
