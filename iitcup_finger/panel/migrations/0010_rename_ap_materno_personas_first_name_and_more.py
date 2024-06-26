# Generated by Django 5.0.1 on 2024-04-24 00:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0009_alter_personas_table'),
    ]

    operations = [
        migrations.RenameField(
            model_name='personas',
            old_name='ap_materno',
            new_name='first_name',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='id_persona',
            new_name='id_expert',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='activate',
            new_name='is_active',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='ap_paterno',
            new_name='last_name',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='cargo',
            new_name='location',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='nombre',
            new_name='middle_name',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='cede',
            new_name='position',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='f_registro',
            new_name='registration_date',
        ),
        migrations.RenameField(
            model_name='personas',
            old_name='usuario',
            new_name='user',
        ),
    ]
