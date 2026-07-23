"use client";

import { FormEvent, useEffect, useState } from "react";
import { client } from "@/lib/api";

type Org = { id: string; name: string; plan: string; seat_limit: number; member_count: number };
type Member = { user_id: string; email: string; name: string; role: string };

export default function TeamSettingsPage() {
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [orgName, setOrgName] = useState("");
  const [selected, setSelected] = useState("");
  const [members, setMembers] = useState<Member[]>([]);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteToken, setInviteToken] = useState("");
  const [message, setMessage] = useState("");

  async function refresh() {
    const list = await client.orgs.list();
    setOrgs(list);
    if (list.length && !selected) setSelected(list[0].id);
  }

  useEffect(() => {
    refresh().catch((err) => setMessage(err instanceof Error ? err.message : "Failed"));
  }, []);

  useEffect(() => {
    if (!selected) return;
    client.orgs
      .members(selected)
      .then(setMembers)
      .catch(() => setMembers([]));
  }, [selected]);

  async function createOrg(e: FormEvent) {
    e.preventDefault();
    await client.orgs.create(orgName.trim());
    setOrgName("");
    await refresh();
  }

  async function invite(e: FormEvent) {
    e.preventDefault();
    if (!selected) return;
    const res = await client.orgs.invite(selected, inviteEmail.trim());
    setInviteToken(res.token);
    setInviteEmail("");
    setMessage(`Invite created for ${res.email}. Share token: ${res.token}`);
  }

  return (
    <div className="p-8">
      <h1 className="font-display text-4xl text-mist-100">Team</h1>
      <p className="mt-2 text-mist-400">Organizations, seats, and invites (Pro+ friendly).</p>
      {message && <p className="mt-4 break-all text-sm text-tungsten-400">{message}</p>}

      <form onSubmit={createOrg} className="mt-8 flex gap-3">
        <input
          value={orgName}
          onChange={(e) => setOrgName(e.target.value)}
          placeholder="Organization name"
          className="rounded-xl border border-white/10 bg-ink-950 px-4 py-2"
        />
        <button className="rounded-full bg-tungsten-500 px-5 py-2 text-sm font-medium text-ink-950">Create org</button>
      </form>

      <div className="mt-8">
        <label className="text-sm text-mist-400">Active org</label>
        <select
          className="mt-2 block rounded-lg border border-white/10 bg-ink-950 px-3 py-2"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {orgs.map((o) => (
            <option key={o.id} value={o.id}>
              {o.name} ({o.member_count}/{o.seat_limit} seats · {o.plan})
            </option>
          ))}
        </select>
      </div>

      <form onSubmit={invite} className="mt-6 flex gap-3">
        <input
          type="email"
          value={inviteEmail}
          onChange={(e) => setInviteEmail(e.target.value)}
          placeholder="invite@studio.com"
          className="rounded-xl border border-white/10 bg-ink-950 px-4 py-2"
          required
        />
        <button className="rounded-full border border-white/20 px-5 py-2 text-sm">Send invite</button>
      </form>
      {inviteToken && <p className="mt-2 text-xs text-mist-400">Last invite token: {inviteToken}</p>}

      <ul className="mt-8 space-y-2">
        {members.map((m) => (
          <li key={m.user_id} className="border border-white/10 px-4 py-3 text-sm">
            {m.name || m.email} · {m.role}
          </li>
        ))}
      </ul>
    </div>
  );
}
