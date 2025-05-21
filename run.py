from flask import render_template, request, jsonify, redirect, url_for
import requests
from flask_server import app, db
from flask_server.university.models import Holidays, Course, Student
from chat import chatbot_response
from flask_server.university.nlp_utils import course_matcher

with app.app_context():
    db.create_all()


@app.post("/chatbot_api/")
def normal_chat():
    msg = request.get_json().get("message")
    response, tag = chatbot_response(msg)

    if tag == "result":
        return jsonify({"response": response, "tag": tag, "url": "result/"})

    if tag == "courses":
        course = course_matcher(msg)
        if course is not None:
            course_details = Course.query.filter_by(name=course)[0]
            response = f"{course_details.name} takes {course_details.duration}"
            link = f"http://127.0.0.1:5000/courses/syllabus/{course_details.id}/"
            return jsonify(
                {
                    "response": response,
                    "tag": tag,
                    "data": {
                        "filename": f"{course_details.name} syllabus",
                        "link": link,
                    },
                }
            )
        else:
            courses = Course.query.all()
            for course in courses:
                response += f"\n {course}"

    if tag == "holidays":
        holiday = Holidays.query.first()
        link = f"http://127.0.0.1:5000/holidays/download/2/"
        response = f"Holidays for year {holiday.year} is down below"
        return jsonify(
            {
                "response": response,
                "tag": tag,
                "data": {"filename": holiday.file_name, "link": link},
            }
        )

    if tag == "faculty":
        data = requests.get(url="http://127.0.0.1:5000/teachers/api/")
        for item in data.json():
            teacher = f"{item['name']} ({item['department']})"
            response = (response + "\n " + teacher,)
    # Handle test_media tag or any tag with media
    return jsonify(
        {
            "response": response.get(
                "text", response
            ),  # Fallback to response if text not present
            "tag": tag,
            "media": response.get("media", []),  # Include media if present
            "link": response.get("link", ""),  # Include link if present
        }
    )


@app.post("/chatbot_api/result/")
def fetch_result():
    msg = request.get_json().get("message")
    try:
        studentID = msg.strip()
        student = Student().query.get(studentID)
        response = f"result of {studentID} is {student.cgpa}"
        url = ""
    except ValueError:
        msg = msg.replace(" ", "")
        if msg.isalpha():
            response = "please repeat yourself once again ..."
            url = ""
        else:
            response = "please use the correct format : \n434121010021"
            url = "result/"
    except:
        return jsonify({"response": "Student not found", "url": ""})

    return jsonify({"response": response, "url": url})


# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
