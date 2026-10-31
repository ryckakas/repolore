# Branch protection ruleset

`protect-main.json` is the "Protect main" ruleset, applied to this repository. It blocks deleting
or force-pushing the default branch, requires every change to go through a pull request (no
direct pushes, no bypass, admins included), and requires the `lint`, `typecheck`,
`test (ubuntu-latest)`, `test (windows-latest)`, `test (macos-latest)`, `build`, `typos`, `zizmor` and `actionlint`
checks from `.github/workflows/ci.yml` to pass before a PR can merge. `audit` runs on every PR
but isn't required: a newly published advisory would otherwise block every PR, related or not. No
required review count: solo repo, so that's just CI plus a PR.

CodeQL (Python and Actions) runs from `.github/workflows/codeql.yml`. Its `analyze (python)` and
`analyze (actions)` jobs are required checks, so a failed analysis shows as a failed check rather
than as results that never arrive. A `code_scanning` rule gates the results themselves: a PR can't
merge without them, or with an error-level alert or a high or critical security alert. GitHub's default setup is switched off,
because it skips pull requests from forks, which would leave this rule waiting on them forever.

The file is the source of truth, but GitHub doesn't read it. A change here, such as a renamed or
added CI job, has to be applied by hand: **Settings → Rules → Rulesets → Protect main**, or
through the API with `gh api -X PUT repos/ryckakas/repolore/rulesets/<id> --input
.github/rulesets/protect-main.json` (`gh api repos/ryckakas/repolore/rulesets` lists the id).
