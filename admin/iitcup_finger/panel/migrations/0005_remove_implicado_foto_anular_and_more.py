# Generated by Django 5.0.1 on 2024-04-24 00:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0004_alter_casos_perito'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='implicado',
            name='foto_anular',
        ),
        migrations.RemoveField(
            model_name='implicado',
            name='foto_indice',
        ),
        migrations.RemoveField(
            model_name='implicado',
            name='foto_medio',
        ),
        migrations.RemoveField(
            model_name='implicado',
            name='foto_menique',
        ),
        migrations.RemoveField(
            model_name='implicado',
            name='foto_pulgar',
        ),
    ]
