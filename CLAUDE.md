# Core Guidelines

- After every significant change, offer to do a git commit
- Never commit code without explicit user confirmation. Just because the
  user has consented to one commit doesn't mean they consent to all future commits.
- When removing or changing code, never add comments about what was removed or
  changed. Comments in the code should always reflect what's there in the moment.
- You are an autonomous agent. You make use of all the tools available to you.
  You write instrumentation, run tests and smoketests, and gather information to
  solve problems. You iterate persistently until your requirements are met.
- NEVER create temporary files, scripts, or intermediate artifacts in the
  project root or source directories. ALL temporary files MUST be created
  exclusively inside the `./tmp/` directory.
- ALWAYS lint and fix all warnings before returning to the user.
- As a final step, ALWAYS format code before returning to the user.
- Adding an environment variable to configure code, enable/disable features,
  enable debugging output is almost always the wrong approach. Prefer function
  parameters or configuration structs.
- Adding sleeps or timeouts to code is nearly ALWAYS the wrong approach. Prefer
  using synchronization primitives, callbacks, or event-driven mechanisms.
  Never try to "fix" test failures by tinkering with timeouts or sleeps, and
  treat every construction of an arbitrary timeout as a code smell.
- DO NOT over-use conditional compilation. Every time you're tempted to add
  `#[cfg(...)]` to your code, ask yourself if there's a better way to structure
  the code so that it doesn't need to be conditionally compiled. Consider
  refactoring the code to extend APIs to avoid it.
- In general, prefer not to write things yourself if there's a well-known,
  well-maintained library that does what you need. Always check for existing
  libraries before implementing functionality from scratch.
- Unless specified, backwards compatibility is not a concern. You may change APIs,
  remove deprecated code, and refactor existing code as needed to improve
  quality and maintainability.
- When making active improvements, keep the scope proportional to the change
  you're making. Don't wholesale refactor files you only touched incidentally —
  improvements should focus on code you're actively working with.
- If you discover pre-existing test failures unrelated to your work, report them
  to the user rather than working around or masking them.

# Security: Open Source Dependency Vetting

Before integrating any open source software — whether from npm, PyPI, GitHub, or any other package registry — perform the following checks:

1. **Search for recent vulnerabilities and attacks** targeting the package, its maintainers, or its ecosystem. Use web searches covering the last 6 months for terms like the package name combined with "vulnerability", "supply chain attack", "malware", or "compromised".

2. **Check for known attack campaigns** (e.g., GlassWorm, which uses invisible Unicode injection, typosquatting, and sleeper packages). Verify the package is not listed in any recent advisories.

3. **Scan for suspicious invisible Unicode characters** (U+200B–U+200F, U+2028–U+202F, U+2060–U+206F, U+FEFF, etc.) in source files after installation. These can hide malicious code injections.

4. **Run `npm audit`** (or the equivalent for the package manager in use) after adding or updating dependencies, and resolve any flagged issues before proceeding.

5. **Verify package provenance**: check the package's download counts, maintainer history, repository activity, and whether the package name closely mimics a popular package (typosquatting).

6. **Pin dependency versions** using lockfiles (`package-lock.json`, `requirements.txt`, etc.) to prevent silent upgrades to compromised versions.

These checks apply to all dependency additions and updates, not just initial integration.

# Testing
Continually update the test suite to ensure the complete functionality of the application
- prior to every commit, ensure that unit tests have been written, carried out and passed 
  for the new functionality, and these are added to the general test suite.  
- prior to every commit, rerun the entire test suite, to ensure that all existing 
  functionality remains working.
- unit tests should include both correct pass and correct fail scenarios.
- if the user reports an error in the application, after correcting and confirmation 
  from the user, add this to the test suite.

# Active API Tending

Continuously improve internal and external APIs as we work. Every time you
touch a piece of code, consider both the API it is part of and the API it
interacts with, and whether either needs to be actively tended. A good API is:

- Minimal, and without unnecessary surface area.
- Consistent in naming, structure and behavior.
- Elegantly and clearly expresses the INTENT of the code.
- Does not expose implementation details.

Internal APIs should be designed with the same care as public APIs. 

Use available tooling to inspect the API surface area of the code you're working
on. Consider whether the API surface is a good encapsulation of the intent of
the module.

You may make contained improvements to the APIs as part of an unrelated patch.
Bring larger API changes to the user's attention or add them to the checklist
for explicit approval.

Examples of tending to the API include:

- Removing or making private functions that are not needed. 
- Consolidating traits, structs or functions that are similar.
- Adding better abstractions to express intent.
- Generalizing or specializing functions to improve ergonomics.

Every time you've tended the API, include an "API Tending" section in your
response message describing what you've done and what your API thought process
was.

# Active Code Maintenance

Every time you touch a piece of code, evaluate whether it can be improved
structurally. Ask questions like:

- Is the documentation for this function clear, concise and accurate?
- Should the code be broken up into smaller pieces?
- Can the code be generalized or made more flexible?
- Can related code be refactored to share functionality?
- Is there a generic or utility function that could be extracted and used more
  widely?
- Should the code be moved to a different location in the project?

Improve code continuously when opportunities arise, even if the user hasn't
explicitly asked for it. When you do active maintenance, include an "Active
Maintenance" section in your response message.

# Active Complexity Reduction

You will actively reduce complexity in the code you touch, wherever possible.
Complexity reduction may take the form of:

- Removing a layer of indirection. For instance, if a function is simply
  forwarding to another function without adding value, remove the forwarding
  function and have callers call the target function directly.
- Removing a layer of abstraction. For instance, if a trait is only
  implemented by one struct, consider removing the trait and having callers
  depend on the struct directly.
- Amalgamating two similar functions or structs into one.
- Shifting implementation of a function only used in one place into the caller.
- Making a function more generic to reduce the need for multiple similar functions.

Complexity reduction is a primary goal so prioritize it highly. When you reduce
complexity, include a "Complexity Reduction" section in your
response message describing what you've done and why.


# Checklists

Whenever you're asked to produce a todo list or a checklist, you will use a
Markdown checklist, with numbered sections and items. Each item should be a
single, coherent change that leaves the system in a consistent state. Try not
to leave a broken system after any step, but certainly after a stage all tests
and smoketests must pass. Always include enough information that you could pick
it up again with zero context. Always wrap at 100 chars.

The checklist is a LIVE DOCUMENT, update it as you go - if you discover new
items during your work or leave items for a later commit, add them to the
checklist. Ensure that any new item you add is a also Markdown checklist item
(i.e. starts with `[ ]`), and has a number in sequence with other items in the
document. You should evaluate next steps continuously, and modify the checklist 
to incorporate what you learn as you work.

You may batch together todo items that you think belong in the same changeset
without prompting me. After every batch, let me review the code before
committing. 

IMMEDIATELY tick off each item in the original checklist file on disk as you
complete them, so we don't lose track of where we are. Don't confuse your own
checklist with the user's checklist - update both your internal checklist and
the checklist on disk.

EVERY time you are implementing a checklist, include a section titled
"Checklist Adjustments" that describes any changes you made to future items in
in the checklist. Be flexible to changing the checklist as you learn more about
the project during execution

Example format:

```markdown
# Task description

Here is the context needed to understand the task, and an outline of its broad
aims.

1. Stage One: Frobnitz the flange

Perhaps some explanation and comments go here.

1. [ ] Do a thing!
2. [ ] Do thing two.

2. Stage Two: Retrofit the turbo-enabulator

Perhaps some explanation and comments go here.

1. [ ] Second section thing.
2. [ ] Second section thing 2.
```

# Git Commits 

Never commit until you're asked to do so, or the user has explicitly confirmed
they want to commit (the user will say "commit" or "do a git commit" or some
variant of that). Make git commit messages concise and clear. In the body of
the message, provide a concise summary of what's been done, but leave out
details like the validation process. Commit as the user - don't add model
attribution or co-authorship.

First, review the actual changes that are being committed.

```sh
# 1) Review, then stage explicitly (paths or -A).
git status --porcelain

# If necessary, review changes before staging:
git diff 
```

Formulate your commit message, based on the actual diff and the user's
instructions that lead up to this point. Make sure your message covers all
changed code, not just the user's latest prompt.

Next, stage and commit:

```sh
# Stage changes; use -A to stage all changes, or specify paths.
git add -A  # or: git add <paths>

# Commit via stdin; Conventional Commit subject (≤50). Body optional; blank
# line before body; quoted heredoc prevents interpolation.
git commit --cleanup=strip -F - <<'MSG'
feat(ui): concise example

Body
MSG
```

