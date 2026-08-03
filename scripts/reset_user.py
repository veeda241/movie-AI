"""Reset (or just print info about) a user account.

Use when the user is locked out and can't log in via the web UI. Either
prints the account details (no password change) or sets a new password.

    python scripts/reset_user.py --list                       # show all users
    python scripts/reset_user.py --email vyas.sk17@gmail.com  # show details
    python scripts/reset_user.py --email vyas.sk17@gmail.com --password newpass123

The script talks directly to the SQLite DB the same way the API does, so it
works even if the API is down or the user can't reach it from the browser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable so we can reuse the API's hashing code.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from api.auth_utils import hash_password  # noqa: E402
from api.db import SessionLocal  # noqa: E402
from api.models import User  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list all users and exit")
    p.add_argument("--email", help="user email to inspect / reset")
    p.add_argument("--password", help="new plaintext password (min 8 chars)")
    args = p.parse_args()

    db = SessionLocal()
    try:
        if args.list or not args.email:
            users = db.query(User).order_by(User.created_at).all()
            if not users:
                print("No users in the database.")
                print("Tip: register a new one at http://localhost:3000/register")
                return 0
            print(f"{'email':<35} {'plan':<8} {'credits':<8} created_at")
            print("-" * 80)
            for u in users:
                print(
                    f"{u.email:<35} {str(u.plan):<8} {u.credit_balance:<8} {u.created_at}"
                )
            return 0

        user = db.query(User).filter(User.email == args.email.lower()).first()
        if user is None:
            print(f"No user with email {args.email!r}.")
            print("Use --list to see registered emails.")
            return 2

        print(f"User found:")
        print(f"  id           : {user.id}")
        print(f"  email        : {user.email}")
        print(f"  name         : {user.name}")
        print(f"  plan         : {user.plan}")
        print(f"  credit_balance: {user.credit_balance}")
        print(f"  created_at   : {user.created_at}")

        if args.password:
            if len(args.password) < 8:
                print("Password too short; min 8 characters.")
                return 3
            user.password_hash = hash_password(args.password)
            db.add(user)
            db.commit()
            print(f"\nPassword updated. You can now sign in at http://localhost:3000/login")
            print(f"  email    : {user.email}")
            print(f"  password : {args.password}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
