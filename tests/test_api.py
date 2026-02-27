import pytest
from pathlib import Path

PROJECT_ROOT = Path("/home/claude-dev/projects/glm-ocr-service")


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "worker_count" in data
        assert "queue_depth" in data


class TestSubmitEndpoint:
    async def test_submit_png_image(self, client):
        img_path = PROJECT_ROOT / "test_page1.png"
        with open(img_path, "rb") as f:
            response = await client.post(
                "/ocr/submit",
                files={"file": ("test_page1.png", f, "image/png")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["total_pages"] == 1

    async def test_submit_pdf(self, client):
        pdf_path = PROJECT_ROOT / "test_document.pdf"
        with open(pdf_path, "rb") as f:
            response = await client.post(
                "/ocr/submit",
                files={"file": ("test_document.pdf", f, "application/pdf")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["total_pages"] >= 1

    async def test_submit_invalid_file_type(self, client):
        response = await client.post(
            "/ocr/submit",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    async def test_submit_file_too_large(self, client):
        large_data = b"x" * (51 * 1024 * 1024)  # 51 MB
        response = await client.post(
            "/ocr/submit",
            files={"file": ("large.png", large_data, "image/png")},
        )
        assert response.status_code == 413


class TestStatusEndpoint:
    async def test_get_status_valid_job(self, client):
        img_path = PROJECT_ROOT / "test_page1.png"
        with open(img_path, "rb") as f:
            submit_resp = await client.post(
                "/ocr/submit",
                files={"file": ("test_page1.png", f, "image/png")},
            )
        job_id = submit_resp.json()["job_id"]

        response = await client.get(f"/ocr/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "processing", "completed", "failed")

    async def test_get_status_invalid_job(self, client):
        response = await client.get("/ocr/status/nonexistent-job-id")
        assert response.status_code == 404


class TestResultEndpoint:
    async def test_get_result_valid_job(self, client):
        img_path = PROJECT_ROOT / "test_page1.png"
        with open(img_path, "rb") as f:
            submit_resp = await client.post(
                "/ocr/submit",
                files={"file": ("test_page1.png", f, "image/png")},
            )
        job_id = submit_resp.json()["job_id"]

        response = await client.get(f"/ocr/result/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "pages" in data

    async def test_get_result_invalid_job(self, client):
        response = await client.get("/ocr/result/nonexistent-id")
        assert response.status_code == 404


class TestListJobsEndpoint:
    async def test_list_jobs_empty(self, client):
        response = await client.get("/ocr/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    async def test_list_jobs_with_filter(self, client):
        img_path = PROJECT_ROOT / "test_page1.png"
        with open(img_path, "rb") as f:
            await client.post(
                "/ocr/submit",
                files={"file": ("test_page1.png", f, "image/png")},
            )

        response = await client.get("/ocr/jobs?status=queued")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_list_jobs_pagination(self, client):
        response = await client.get("/ocr/jobs?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 5


class TestDeleteEndpoint:
    async def test_delete_valid_job(self, client):
        img_path = PROJECT_ROOT / "test_page1.png"
        with open(img_path, "rb") as f:
            submit_resp = await client.post(
                "/ocr/submit",
                files={"file": ("test_page1.png", f, "image/png")},
            )
        job_id = submit_resp.json()["job_id"]

        response = await client.delete(f"/ocr/jobs/{job_id}")
        assert response.status_code == 204

        # Verify deleted
        status_resp = await client.get(f"/ocr/status/{job_id}")
        assert status_resp.status_code == 404

    async def test_delete_invalid_job(self, client):
        response = await client.delete("/ocr/jobs/nonexistent-id")
        assert response.status_code == 404
