from functools import lru_cache
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import src.logger
from api.schemas import JobPostRequest, PredictionResponse
from src.pipeline.prediction_pipeline import FakeJobPredictor


app = FastAPI(title="Fake Job Detector API", version="1.0.0")
logger = logging.getLogger(__name__)


def render_app():
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Fake Job Detector</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                color: #172033;
            }}
            main {{
                max-width: 900px;
                margin: 32px auto;
                padding: 0 16px;
            }}
            form, .result {{
                background: white;
                border: 1px solid #d9dee7;
                border-radius: 8px;
                padding: 20px;
                margin-top: 16px;
            }}
            label {{
                display: block;
                margin-top: 14px;
                font-weight: 700;
            }}
            input, textarea {{
                width: 100%;
                box-sizing: border-box;
                margin-top: 6px;
                padding: 10px;
                border: 1px solid #b8c0cc;
                border-radius: 6px;
                font-size: 15px;
            }}
            textarea {{
                min-height: 110px;
                resize: vertical;
            }}
            button {{
                margin-top: 18px;
                padding: 11px 16px;
                border: 0;
                border-radius: 6px;
                background: #1f4e79;
                color: white;
                font-size: 15px;
                cursor: pointer;
            }}
            .hint {{
                color: #5f6b7a;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <main>
            <h1>Fake Job Detector</h1>
            <p class="hint">Enter the main job details. Company profile, salary range, and experience are optional.</p>
            <section id="result" class="result" style="display:none"></section>
            <form id="job-form">
                <label>Job Title *</label>
                <input name="title" required>

                <label>Description *</label>
                <textarea name="description" required></textarea>

                <label>Requirements *</label>
                <textarea name="requirements" required></textarea>

                <label>Company Profile</label>
                <textarea name="company_profile"></textarea>

                <label>Salary Range</label>
                <input name="salary_range">

                <label>Required Experience</label>
                <input name="required_experience">

                <button type="submit">Check Job</button>
            </form>
        </main>
        <script>
            const form = document.getElementById("job-form");
            const resultBox = document.getElementById("result");

            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                const data = Object.fromEntries(new FormData(form).entries());
                resultBox.style.display = "block";
                resultBox.innerHTML = "Checking...";

                const response = await fetch("/predict", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify(data)
                }});

                if (!response.ok) {{
                    resultBox.innerHTML = "Could not check this job. Make sure required fields are filled.";
                    return;
                }}

                const result = await response.json();
                const color = result.label === "fake" ? "#b91c1c" : "#047857";
                resultBox.style.borderColor = color;
                resultBox.innerHTML = `
                    <h2 style="color:${{color}}">${{result.label.toUpperCase()}} JOB</h2>
                    <p>Fake probability: <strong>${{result.fake_probability}}</strong></p>
                `;
            }});
        </script>
    </body>
    </html>
    """


@lru_cache
def get_predictor():
    return FakeJobPredictor()


@app.get("/", response_class=HTMLResponse)
def app_form():
    return render_app()


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "ok", "message": "Fake Job Detector API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fake_job(payload: JobPostRequest):
    logger.info("Prediction request received")
    try:
        predictor = get_predictor()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet. Run the training pipeline first.",
        ) from error

    return predictor.predict(payload.dict())
