# Generated by Django 3.2.12 on 2024-11-12 14:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AI_vBrainApp', '0002_rename_team_teammembers'),
    ]

    operations = [
        migrations.RenameField(
            model_name='teammembers',
            old_name='email',
            new_name='EMAIL',
        ),
        migrations.RenameField(
            model_name='teammembers',
            old_name='name',
            new_name='NAME',
        ),
        migrations.RenameField(
            model_name='teammembers',
            old_name='password',
            new_name='PASSWORD',
        ),
        migrations.RenameField(
            model_name='teammembers',
            old_name='phone',
            new_name='PHONE',
        ),
        migrations.RemoveField(
            model_name='teammembers',
            name='id',
        ),
        migrations.AddField(
            model_name='teammembers',
            name='ID',
            field=models.IntegerField(default=1, primary_key=True, serialize=False),
            preserve_default=False,
        ),
    ]
