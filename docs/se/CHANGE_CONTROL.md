# Change control and baselines

**Baseline.** A tagged release (`git tag v0.1.0`) whose `docs/se/requirements.json`, `ICD.md` and `FMEA.md` match the code. Only changes through the steps below move the baseline.

**Change steps**
1. Open an issue stating which requirement(s), interface(s) or failure mode(s) change, and why.
2. Update `requirements.json` first, then code and tests.
3. Run `python scripts/trace_check.py` (and `make test`). It must report no problems.
4. Review by someone other than the author. A change to a HIGH-risk tool, a policy rule or the red-team set needs the policy owner.
5. CI must pass, then merge. Update FMEA scores if the change alters a failure mode.

**Rules**
- A new tool, role or guard needs a requirement, an ICD row and at least one test before merge.
- Removing or loosening a requirement needs written justification in the issue.
- The red-team file only grows; removing a case needs the security owner.

CI is the enforcement point: the trace check runs next to the tests.
