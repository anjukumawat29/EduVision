"""
management/commands/create_user.py
────────────────────────────────────
Creates a teacher or student account from the command line.

Usage:
    # Create a teacher
    python manage.py create_user --username mrsmith --password secret123 --role teacher --name "Mr Smith"

    # Create a student (username must match their registered dataset name)
    python manage.py create_user --username rahulsharma --password pass123 --role student --name "Rahul Sharma"

Put this file at:
    attendance/management/commands/create_user.py
(create the management/ and commands/ folders with empty __init__.py files if they don't exist)
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from attendance.models import UserProfile


class Command(BaseCommand):
    help = "Create a teacher or student login account"

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True)
        parser.add_argument("--password", required=True)
        parser.add_argument("--role",     required=True, choices=["teacher", "student"])
        parser.add_argument("--name",     default="", help="Full name (first last)")

    def handle(self, *args, **options):
        username   = options["username"]
        password   = options["password"]
        is_teacher = options["role"] == "teacher"
        full_name  = options["name"].strip()

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.ERROR(f"User '{username}' already exists."))
            return

        first, *rest = full_name.split() if full_name else (username, [])
        last = " ".join(rest)

        user = User.objects.create_user(
            username=username, password=password,
            first_name=first, last_name=last,
        )
        user.profile.is_teacher = is_teacher
        user.profile.save()

        role_label = "Teacher" if is_teacher else "Student"
        self.stdout.write(self.style.SUCCESS(
            f"{role_label} account created: '{username}' / '{password}'"
        ))