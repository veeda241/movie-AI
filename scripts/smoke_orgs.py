import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)
r = c.post("/auth/login", json={"email": "demo@example.com", "password": "secret12"})
h = {"Authorization": f"Bearer {r.json()['access_token']}"}
o = c.post("/orgs", headers=h, json={"name": "Studio One"})
print("org", o.status_code, o.json())
oid = o.json()["id"]
inv = c.post(f"/orgs/{oid}/invite", headers=h, json={"email": "member@example.com"})
print("invite", inv.status_code, inv.json().get("token", "")[:8])
r2 = c.post("/auth/register", json={"email": "member@example.com", "password": "secret12", "name": "Member"})
if r2.status_code >= 400:
    r2 = c.post("/auth/login", json={"email": "member@example.com", "password": "secret12"})
h2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}
acc = c.post("/orgs/accept-invite", headers=h2, json={"token": inv.json()["token"]})
print("accept", acc.status_code, acc.json())
print("members", c.get(f"/orgs/{oid}/members", headers=h).json())
