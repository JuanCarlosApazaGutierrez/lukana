# Generated by Django 5.0.1 on 2024-03-06 02:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='huellas',
            name='caso',
        ),
    ]