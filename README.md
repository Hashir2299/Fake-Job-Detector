# Fake Job Detector - MLOps Project

Fake Job Detector is an end-to-end MLOps project built to help identify suspicious or fraudulent job postings. Fake job listings have become a real problem in today's online hiring world, and this project is my attempt to build a practical machine learning system that can support safer job searching.

The current version is a FastAPI web application that I containerized with Docker and successfully deployed on AWS using ECR and EKS. A user can enter job details such as title, description, requirements, company profile, salary range, and required experience, and the model predicts whether the job looks real or fake with a fake-job probability.

In the future, I plan to extend this idea into a Google Chrome extension so it can be used directly while applying on job platforms and websites like Indeed, JobLeads, and other job-hunting websites.

## Project Overview

This project covers the main parts of a small production-style ML workflow:

- Data ingestion and preprocessing
- Text feature extraction with TF-IDF
- Model training using Logistic Regression
- Oversampling to handle class imbalance
- Model evaluation
- FastAPI prediction API
- Simple web interface for testing predictions
- Docker containerization
- GitHub Actions CI workflow
- Kubernetes deployment manifests
- Successful AWS deployment using ECR and EKS

## Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- imbalanced-learn
- pandas and NumPy
- joblib
- Docker
- GitHub Actions
- AWS Elastic Container Registry
- Amazon EKS
- Kubernetes manifests
- DVC explored during the project

## MLOps Notes

During development, I successfully used GitHub Actions for CI/CD-style automation. The workflow installs dependencies, runs tests, and builds the Docker image on pushes and pull requests to the `main` branch.

I also successfully deployed the application on AWS by pushing the Docker image to Amazon ECR and running it through Amazon EKS with Kubernetes manifests. Since this is a learning and portfolio project, I may later delete the IAM user, cluster, and related AWS resources so I do not keep paying for cloud infrastructure after the deployment work is complete.

DVC was also explored for data and model versioning. However, I later removed the S3 remote/bucket part because I did not want to keep using AWS credits for storage and cloud resources.

## Project Structure

```text
.
|-- api/                  # FastAPI application and request/response schemas
|-- config/               # Model and schema configuration files
|-- k8s/                  # Kubernetes deployment and service manifests
|-- src/
|   |-- components/       # Data processing, training, validation, evaluation
|   |-- pipeline/         # Training and prediction pipelines
|   |-- constants/        # Shared project paths and constants
|   |-- logger/           # Logging setup
|   `-- utils/            # Utility functions
|-- tests/                # Unit and API tests
|-- Dockerfile            # Container image definition
|-- requirements.txt      # Python dependencies
`-- README.md
```

## How It Works

1. Raw job-posting data is loaded from `data/raw/fake_job_postings.csv`.
2. Important text fields are selected and cleaned.
3. The data is split into train, validation, and test sets.
4. Training data is oversampled to reduce class imbalance.
5. Text is converted into numerical features using TF-IDF.
6. A Logistic Regression model is trained and saved as `models/fake_job_model.pkl`.
7. FastAPI loads the trained model and serves predictions through `/predict`.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.pipeline.training_pipeline
```

Start the FastAPI app:

```bash
uvicorn api.main:app --reload
```

Open the web app:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns the API status.

### Predict Fake Job

```http
POST /predict
```

Example request:

```json
{
  "title": "Data Entry Clerk",
  "description": "Work from home with flexible hours and quick hiring.",
  "requirements": "Basic computer skills required.",
  "company_profile": "Online hiring company",
  "salary_range": "50000-70000",
  "required_experience": "Entry level"
}
```

Example response:

```json
{
  "prediction": 1,
  "label": "fake",
  "fake_probability": 0.8421
}
```

## Run With Docker

Build the image:

```bash
docker build -t fake-job-detector .
```

Run the container:

```bash
docker run -p 8000:8000 fake-job-detector
```

Then visit:

```text
http://127.0.0.1:8000
```

## Tests

Run the test suite:

```bash
pytest
```

The GitHub Actions workflow also runs tests automatically before building the Docker image.

## Future Improvements

- Build a Chrome extension for checking job posts directly on job boards
- Add better model monitoring and drift detection
- Reconnect DVC with cloud storage when budget allows
- Improve the AWS/EKS deployment pipeline
- Improve the UI for easier job-post analysis
- Add more explainability around why a job was classified as fake

## Disclaimer

This project is for learning and portfolio purposes. The prediction should be treated as a helpful signal, not a final decision. Job seekers should still verify the company, recruiter identity, payment requests, and official job posting sources before applying or sharing personal information.
