from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="userprofile")
    is_teacher = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username



# ── Existing models ────────────────────────────────────────────────

class Student(models.Model):
    name       = models.CharField(max_length=100)
    roll_no    = models.CharField(max_length=30, unique=True, blank=True, default="")
    registered = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def photo_count(self):
        import os
        from django.conf import settings
        folder = os.path.join(settings.DATASET_DIR, self.name)
        if not os.path.exists(folder):
            return 0
        return len([f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])


class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE,
                                related_name="attendance_records")
    date    = models.DateField()
    time    = models.TimeField()
    subject = models.CharField(max_length=100, default="General")
    status  = models.CharField(max_length=10, default="Present")

    class Meta:
        unique_together = ("student", "date", "subject")

    def __str__(self):
        return f"{self.student.name} — {self.date} — {self.status}"


class BehaviorSession(models.Model):
    started_at         = models.DateTimeField(auto_now_add=True)
    duration_seconds   = models.IntegerField(default=0)
    attentive_seconds  = models.IntegerField(default=0)
    phone_seconds      = models.IntegerField(default=0)
    distracted_seconds = models.IntegerField(default=0)

    @property
    def total(self):
        return self.duration_seconds or 1

    @property
    def attentive_pct(self):
        return round(self.attentive_seconds / self.total * 100)

    @property
    def phone_pct(self):
        return round(self.phone_seconds / self.total * 100)

    @property
    def distracted_pct(self):
        return round(self.distracted_seconds / self.total * 100)