# Generated by Django 5.0.1 on 2024-05-26 16:19

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0017_alter_implicado_altura_m_alter_implicado_peso_kg'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='huellas',
            name='persona',
        ),
    ]
