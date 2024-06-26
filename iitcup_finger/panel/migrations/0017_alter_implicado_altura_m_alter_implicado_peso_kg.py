# Generated by Django 5.0.1 on 2024-04-29 06:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0016_alter_implicado_fecha_nacimiento'),
    ]

    operations = [
        migrations.AlterField(
            model_name='implicado',
            name='altura_m',
            field=models.DecimalField(decimal_places=2, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='implicado',
            name='peso_kg',
            field=models.DecimalField(decimal_places=2, max_digits=5, null=True),
        ),
    ]
