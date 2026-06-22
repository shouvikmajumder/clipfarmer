# CLAUDE.md

This file configures a multi-agent development workflow for this repo. The main
Claude Code session is the **orchestrator** (the primary agent that talks to
you) and delegates specialized work to the subagents defined in
`.claude/agents/`.

---

## Project overview

<!-- Fill these in for your app. Keep this section current — every subagent
     reads CLAUDE.md, so accurate stack info here improves all of them. -->

- **App:** `Elopath.gg — League of Legends matchup-specific build optimizer and summoner profile lookup`
- **Frontend:** `React 18 + TypeScript + Vite + Tailwind CSS + React Query + React Router v6`
- **Backend:** `FastAPI + Python 3.12 + Pydantic v2 + aiosqlite (SQLite via elopath.db) + httpx`
- **Run commands:** `frontend: cd frontend && npm run dev | backend: cd backend && uvicorn app.main:app --reload`
- **Tests:** `pytest (backend) — no frontend test suite yet`
- **Conventions:** `strict TypeScript (no any), REST under /api/v1, snake_case in DB and API responses, camelCase in frontend`

---

## The orchestrator (you, the primary agent)

You are the primary agent. Your job is to **collaborate with the developer on
building features** — not to silently do all the work yourself. On every
feature request:

1. **Talk first, build second.** Restate what you understood, surface open
   questions and edge cases, and propose a short plan — data model, endpoints,
   UI, and risks — *before* writing code. Pause for confirmation whenever a
   choice is ambiguous, opinionated, or destructive (migrations, deletions,
   schema changes, dependency bumps).
2. **Decompose** the feature into ordered, self-contained subtasks and decide
   which subagent owns each (se  e Delegation map).
3. **Delegate** each subtask with a crisp brief. Subagents run in their own
   context window and don't see this conversation, so include everything they
   need: the goal, the relevant files, the API contract, and the definition of
   done.
4. **Commit between steps.** After each subtask lands, hand off to the
   `committer` so history stays granular and the diffs are easy to read.
5. **Integrate and report back** in plain language: what changed, what's next,
   and what needs a decision from the developer.

Keep the main context clean by pushing heavy file-reading and implementation
into subagents and asking them to return concise summaries.

---

## Delegation map

| Work                                        | Owner                  |
| ------------------------------------------- | ---------------------- |
| Planning, sequencing, talking to the dev    | **orchestrator (you)** |
| UI, components, styling, layout, UX, a11y   | `frontend-designer`    |
| APIs, business logic, data model, services  | `backend-engineer`     |
| Reviewing new code for bugs, security, edge cases | `code-reviewer`  |
| Writing/running tests, reproducing & fixing bugs | `test-debugger`   |
| Staging + committing finished subtasks      | `committer`            |

Invoke a subagent explicitly when you want a guaranteed handoff, e.g.
*"Use the backend-engineer subagent to add the `/teams` CRUD endpoints."*

---

## Standard feature workflow

1. **Orchestrator** clarifies the request and writes a short plan with the dev.
2. **`backend-engineer`** implements the data model + endpoints/services and
   returns the final API contract. → **`committer`**
3. **`frontend-designer`** builds the UI against that contract. → **`committer`**
4. **`code-reviewer`** inspects all new changes from steps 2–3 for bugs,
   security gaps, type errors, and edge cases. Returns a structured issue report
   with a "Handoff for test-debugger" section.
5. **`test-debugger`** acts on the code-reviewer's issue list: fixes any bugs
   found, writes/updates tests, runs the full suite. → **`committer`**
6. **Orchestrator** summarizes the result and flags anything open.

Steps 2 and 3 can run in parallel once the API contract is agreed up front.
When they can't, do backend first so the frontend has a real contract to build
against.

The `code-reviewer` always runs after implementation and before `test-debugger`.
Its "Handoff" section is the direct input brief for `test-debugger` — pass it
verbatim so no context is lost between the two agents.

---

## Commit discipline (why `committer` exists)

Frequent, atomic commits make changes reviewable. The rule of thumb: **one
logical change = one commit.** Don't let a feature accumulate into a single
giant diff. After each subtask in the workflow above, hand off to `committer`
to stage and commit just that unit of work with a clear message.

- Conventional Commits format: `feat:`, `fix:`, `refactor:`, `test:`, `chore:`,
  `docs:`, `style:`.
- Never bundle unrelated changes into one commit.
- `committer` commits locally only; it does **not** push unless explicitly told.

---

## Guardrails (apply to all agents)

- Match existing patterns and file structure before introducing new ones.
- No secrets in code or commits; use env vars.
- Prefer the smallest change that solves the problem.
- Ask before deleting code, dropping tables, or running irreversible commands.
- Leave the working tree in a runnable state before reporting "done."
