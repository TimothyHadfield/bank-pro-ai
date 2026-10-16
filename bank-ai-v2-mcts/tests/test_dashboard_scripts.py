"""
Checks that the v2 dashboard's <script> files can all load together in one page.

Classic (non-module) scripts share one global scope, so a top-level `const`/`let`
that repeats a name another script already declared makes the whole later script
fail to load (that broke every tab except the tab bar). This test runs the scripts
in index.html order inside one Node `vm` realm and fails on any SyntaxError.
Runtime errors from missing DOM elements are ignored: only load errors count.

Needs Node.js on PATH; skipped (not failed) when it isn't there.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"

NODE_RUNNER = r"""
const vm = require("vm");
const fs = require("fs");
const files = JSON.parse(process.argv[1]);
// A forgiving stand-in for the browser: any property chain or call returns the stub.
const stub = new Proxy(function () {}, {
  get: (t, k) => (k === Symbol.toPrimitive ? () => "" : stub),
  apply: () => stub,
  construct: () => stub,
});
const sandbox = { window: null, document: stub, localStorage: stub, fetch: () => new Promise(() => {}),
  setInterval: () => 0, setTimeout: () => 0, alert: () => {}, confirm: () => false, console };
sandbox.window = sandbox;
vm.createContext(sandbox);
const loadErrors = [];
for (const f of files) {
  try {
    vm.runInContext(fs.readFileSync(f, "utf8"), sandbox, { filename: f });
  } catch (e) {
    if (e && e.name === "SyntaxError") loadErrors.push(f + ": " + e.message);
  }
}
console.log(JSON.stringify(loadErrors));
"""


def _script_files_in_page_order():
    html = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
    names = re.findall(r'<script\s+src="([^"?]+)', html)
    return [str(DASHBOARD_DIR / n) for n in names if not n.startswith("http")]


def test_dashboard_scripts_load_together():
    node = shutil.which("node")
    if not node:
        print("  skip test_dashboard_scripts_load_together (node not found)")
        return
    files = _script_files_in_page_order()
    assert len(files) >= 5, f"expected the dashboard's scripts, found {files}"
    out = subprocess.run(
        [node, "-e", NODE_RUNNER, json.dumps(files)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    errors = json.loads(out.stdout.strip().splitlines()[-1])
    assert not errors, "scripts fail to load together:\n  " + "\n  ".join(errors)
    print(f"  ok test_dashboard_scripts_load_together ({len(files)} scripts)")


ALL_TESTS = [test_dashboard_scripts_load_together]


def main():
    failures = []
    for t in ALL_TESTS:
        print(f"running {t.__name__}")
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((t.__name__, e))
            print(f"  FAIL: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(ALL_TESTS)} FAILED")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    main()
