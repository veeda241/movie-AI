from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

APP_TITLE = "Movie Flow Ops"
APP_SUBTITLE = "Agent monitor wired to the Movie Flow API (jobs, events, assets)."

API_URL = os.environ.get("MOVIE_FLOW_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _ensure_state() -> None:
    defaults = {
        "token": os.environ.get("MOVIE_FLOW_API_TOKEN", ""),
        "email": "",
        "password": "",
        "project_id": "",
        "job": None,
        "events": [],
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _headers() -> dict[str, str]:
    token = st.session_state.token
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    with httpx.Client(timeout=120.0) as client:
        response = client.request(method, f"{API_URL}{path}", headers=_headers(), **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text}")
        if response.status_code == 204 or not response.content:
            return None
        return response.json()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_state()

    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)
    st.caption(f"API: {API_URL}")

    with st.sidebar:
        st.header("Auth")
        st.session_state.email = st.text_input("Email", value=st.session_state.email)
        st.session_state.password = st.text_input("Password", type="password", value=st.session_state.password)
        if st.button("Login", use_container_width=True):
            try:
                data = api_request(
                    "POST",
                    "/auth/login",
                    json={"email": st.session_state.email, "password": st.session_state.password},
                )
                st.session_state.token = data["access_token"]
                st.success("Authenticated")
            except Exception as exc:
                st.session_state.last_error = str(exc)

        st.session_state.token = st.text_input("API token", value=st.session_state.token)
        if st.button("Clear job", use_container_width=True):
            st.session_state.job = None
            st.session_state.events = []
            st.session_state.last_error = ""

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if not st.session_state.token:
        st.info("Log in or paste a JWT from Movie Flow to monitor jobs.")
        return

    try:
        me = api_request("GET", "/auth/me")
        projects = api_request("GET", "/projects")
    except Exception as exc:
        st.error(str(exc))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Credits", me.get("credit_balance", 0))
    col2.metric("Plan", me.get("plan", "—"))
    col3.metric("Projects", len(projects or []))

    project_options = {p["name"]: p["id"] for p in projects or []}
    if not project_options:
        if st.button("Create default project"):
            created = api_request("POST", "/projects", json={"name": "Ops Project"})
            st.session_state.project_id = created["id"]
            st.rerun()
        return

    selected_name = st.selectbox("Project", list(project_options.keys()))
    st.session_state.project_id = project_options[selected_name]

    idea = st.text_area(
        "Movie idea",
        height=120,
        placeholder="A courier delivers memories in bottles across a flooded city at dusk.",
    )

    if st.button("Run multi-agent movie", type="primary", use_container_width=True):
        try:
            job = api_request(
                "POST",
                "/generate/movie",
                json={
                    "project_id": st.session_state.project_id,
                    "prompt": idea.strip(),
                    "model": "multi-agent",
                },
            )
            st.session_state.job = job
            st.session_state.events = job.get("events", [])
            st.session_state.last_error = ""
        except Exception as exc:
            st.session_state.last_error = str(exc)

    job = st.session_state.job
    if job:
        st.subheader("Job")
        status_box = st.empty()
        log_box = st.empty()
        status_box.info(f"{job['id']} · {job['status']}")
        log_box.code("\n".join(st.session_state.events), language="text")

        while job["status"] in {"queued", "running"}:
            time.sleep(1.0)
            try:
                job = api_request("GET", f"/jobs/{job['id']}")
                st.session_state.job = job
                st.session_state.events = job.get("events", [])
                status_box.info(f"{job['id']} · {job['status']}")
                log_box.code("\n".join(st.session_state.events), language="text")
            except Exception as exc:
                st.session_state.last_error = str(exc)
                break

        if job["status"] == "succeeded":
            st.success("Movie job completed")
            assets = api_request("GET", f"/assets?project_id={st.session_state.project_id}")
            for asset in assets or []:
                if asset["kind"] == "video":
                    st.write(asset["prompt"][:120])
                    file_url = f"{API_URL}{asset['file_url']}?token={st.session_state.token}"
                    st.video(file_url)
                elif asset["kind"] == "image":
                    file_url = f"{API_URL}{asset['file_url']}?token={st.session_state.token}"
                    st.image(file_url)
        elif job["status"] == "failed":
            st.error(job.get("error") or "Job failed")


if __name__ == "__main__":
    main()
