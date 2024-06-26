# Generated by Django 5.0.1 on 2024-04-24 00:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0012_rename_direccion_implicado_address_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='huellas',
            old_name='caso',
            new_name='case',
        ),
        migrations.RenameField(
            model_name='huellas',
            old_name='id_huella',
            new_name='id_fingerprint',
        ),
        migrations.RenameField(
            model_name='huellas',
            old_name='activate',
            new_name='is_active',
        ),
        migrations.RenameField(
            model_name='huellas',
            old_name='persona',
            new_name='person',
        ),
        migrations.RenameField(
            model_name='huellas',
            old_name='f_registro',
            new_name='registration_date',
        ),
        migrations.RenameField(
            model_name='huellas',
            old_name='tipo',
            new_name='type',
        ),
        migrations.RemoveField(
            model_name='huellas',
            name='procesada_image',
        ),
        migrations.AddField(
            model_name='huellas',
            name='processed_image',
            field=models.ImageField(blank=True, null=True, upload_to='processed/'),
        ),
        migrations.AlterField(
            model_name='huellas',
            name='minutiae_csv',
            field=models.FileField(blank=True, null=True, upload_to='data/'),
        ),
        migrations.AlterField(
            model_name='huellas',
            name='minutiae_image',
            field=models.ImageField(blank=True, null=True, upload_to='minutiae/'),
        ),
    ]
