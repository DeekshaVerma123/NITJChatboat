from .models import Teacher, Student, Course
from flask_server import db, app
from flask import (
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_file,
)
from io import BytesIO
import os
from werkzeug.utils import secure_filename

# =============================
# FOLDERS
# =============================
PLACEMENT_FOLDER = os.path.join(app.root_path, "static", "placement")
HOLIDAYS_FOLDER = os.path.join(app.root_path, "static", "holidays")

# Ensure folders exist
os.makedirs(PLACEMENT_FOLDER, exist_ok=True)
os.makedirs(HOLIDAYS_FOLDER, exist_ok=True)


# =============================
# PLACEMENT
# =============================
@app.route("/placement", methods=["GET", "POST"])
def placement():
    if request.method == "POST":
        brochure = request.files.get("brochure")
        if brochure and brochure.filename.endswith(".pdf"):
            filename = secure_filename(brochure.filename)
            brochure.save(os.path.join(PLACEMENT_FOLDER, filename))

    brochures = [
        {"id": i, "filename": f}
        for i, f in enumerate(os.listdir(PLACEMENT_FOLDER))
        if f.endswith(".pdf")
    ]
    return render_template("placement.html", brochures=brochures)


@app.route("/delete_placement/<int:id>")
def delete_placement(id):
    files = [f for f in os.listdir(PLACEMENT_FOLDER) if f.endswith(".pdf")]
    if 0 <= id < len(files):
        os.remove(os.path.join(PLACEMENT_FOLDER, files[id]))
    return redirect(url_for("placement"))


# =============================
# HOLIDAYS
# =============================
@app.route("/holidays", methods=["GET", "POST"])
@app.route("/holidays", methods=["GET", "POST"])
def holidays():
    if request.method == "POST":
        calendar_file = request.files.get("holiday_file")
        calendar_name = request.form.get("calendar_name")

        if calendar_file and calendar_file.filename.endswith(".pdf"):
            # Use the calendar name as the filename directly
            safe_filename = secure_filename(calendar_name.strip()) + ".pdf"
            calendar_file.save(os.path.join(HOLIDAYS_FOLDER, safe_filename))

    holidays = []
    for i, filename in enumerate(os.listdir(HOLIDAYS_FOLDER)):
        if filename.endswith(".pdf"):
            name = os.path.splitext(filename)[0]  # Remove .pdf extension
            holidays.append({"id": i, "filename": filename, "name": name})

    return render_template("holidays.html", holidays=holidays)


@app.route("/delete_holiday/<int:id>")
def delete_holiday(id):
    files = [f for f in os.listdir(HOLIDAYS_FOLDER) if f.endswith(".pdf")]
    if 0 <= id < len(files):
        os.remove(os.path.join(HOLIDAYS_FOLDER, files[id]))
    return redirect(url_for("holidays"))


# =============================
# HOME
# =============================
@app.route("/")
def hello_world():
    db.create_all()
    return render_template("home.html")


@app.route("/home2/")
def hello_world2():
    return render_template("home2.html")


# =============================
# TEACHERS
# =============================
@app.route("/teachers/", methods=["POST", "GET"])
def teachers():
    if request.method == "POST":
        first_name = request.form["firstname"]
        last_name = request.form["lastname"]
        department = request.form["department"]

        new_teacher = Teacher(
            first_name=first_name, last_name=last_name, department=department
        )
        db.session.add(new_teacher)
        db.session.commit()
        return redirect(url_for("teachers"))

    teachers = Teacher.query.all()
    return render_template("teachers.html", teachers=teachers)


@app.route("/teachers/delete/<int:id>/")
def teachersdelete(id):
    teacher = Teacher.query.get_or_404(id)
    db.session.delete(teacher)
    db.session.commit()
    return redirect(url_for("teachers"))


@app.route("/teachers/api/")
def teachers_api():
    teachers = Teacher.query.all()
    return jsonify(
        [
            {
                "name": teacher.first_name + " " + teacher.last_name,
                "department": teacher.department,
            }
            for teacher in teachers
        ]
    )


@app.route("/teachers/api/<string:dept>/")
def dept_teachers_api(dept):
    teachers = Teacher.query.filter(Teacher.department.ilike(f"%{dept}%")).all()
    return jsonify(
        [
            {
                "name": teacher.first_name + " " + teacher.last_name,
                "department": teacher.department,
            }
            for teacher in teachers
        ]
    )


# =============================
# STUDENTS
# =============================
@app.route("/students/", methods=["POST", "GET"])
def students():
    if request.method == "POST":
        studentID = request.form["studentID"]
        name = request.form["name"]
        courseID = request.form["course"]

        new_student = Student(id=studentID, name=name, course_id=courseID)
        db.session.add(new_student)
        db.session.commit()
        return redirect(url_for("students"))

    students = Student.query.all()
    courses = Course.query.all()
    return render_template("students.html", students=students, courses=courses)


@app.route("/students/update/<int:id>/", methods=["POST", "GET"])
def studentsupdate(id):
    student = Student.query.get_or_404(id)

    if request.method == "POST":
        student.name = request.form["name"]
        student.id = request.form["studentID"]
        student.cgpa = request.form["cgpa"]
        db.session.commit()
        return redirect(url_for("students"))

    return render_template("student_update.html", student=student)


# =============================
# COURSES
# =============================
@app.route("/courses/", methods=["POST", "GET"])
def courses():
    if request.method == "POST":
        name = request.form["name"]
        duration = request.form["duration"]
        syllabus = request.files["file"]

        new_course = Course(name=name, duration=duration, syllabus=syllabus.read())
        db.session.add(new_course)
        db.session.commit()
        return redirect(url_for("courses"))

    courses = Course.query.all()
    return render_template("courses.html", courses=courses)


@app.route("/courses/update/<int:id>/", methods=["POST", "GET"])
def coursesupdate(id):
    course = Course.query.get_or_404(id)

    if request.method == "POST":
        syllabus_file = request.files["file"]
        course.syllabus = syllabus_file.read()
        db.session.commit()
        return redirect(url_for("courses"))

    return render_template("course_update.html", course=course)


@app.route("/courses/syllabus/<int:id>/")
def syllabus_api(id):
    course = Course.query.filter_by(id=id).first()
    return send_file(BytesIO(course.syllabus), download_name=f"{course.name}.pdf")
