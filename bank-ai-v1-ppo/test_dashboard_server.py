"""
Dashboard server API checks (runs the real handler on a free local port).

1. Save / checkpoint names from the browser must be plain names. Before the fix,
   POST /api/delete_save {"name": "."} deleted the whole saves folder, and
   "../something" reached outside it. The test points PROJECT_DIR at a temp
   folder, so nothing real is ever at risk.
2. Play vs AI (unified): when a 7 ends the round, the engine already starts the
   next round inside step(), so last_dice is the new round's roll. The move
   response must say round_ended / busted so the page can show the bust.

Needs torch (the server imports it). Run: python test_dashboard_server.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

sys.path.insert(0, ".")

import dashboard.server as srv


def _start():
    httpd = HTTPServer(("127.0.0.1", 0), srv.DashboardHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_names_cannot_escape_saves_or_models():
    print("TEST: delete/load endpoints refuse '.', '..' and path names")
    tmp = tempfile.mkdtemp(prefix="bankai-test-")
    old_project = srv.PROJECT_DIR
    try:
        proj = os.path.join(tmp, "proj")
        os.makedirs(os.path.join(proj, "saves", "keep_batch10"))
        os.makedirs(os.path.join(proj, srv._config.model_dir))
        with open(os.path.join(proj, "saves", "keep_batch10", "a.pt"), "w") as f:
            f.write("x")
        with open(os.path.join(proj, "secret.txt"), "w") as f:
            f.write("x")
        os.makedirs(os.path.join(tmp, "outside"))
        srv.PROJECT_DIR = proj
        httpd, base = _start()
        try:
            for bad in (".", "..", "../outside", "..\\outside"):
                code, _ = _post(base, "/api/delete_save", {"name": bad})
                assert code == 400, (bad, code)
            for bad in ("../secret.txt", "..\\secret.txt", ".."):
                code, _ = _post(base, "/api/delete_checkpoint", {"filename": bad})
                assert code == 400, (bad, code)
            assert os.path.isfile(os.path.join(proj, "saves", "keep_batch10", "a.pt")), "saves folder was deleted"
            assert os.path.isdir(os.path.join(tmp, "outside")), "folder outside saves was deleted"
            assert os.path.isfile(os.path.join(proj, "secret.txt")), "file outside models was deleted"
            # A normal name still works
            code, r = _post(base, "/api/delete_save", {"name": "keep_batch10"})
            assert code == 200 and r.get("status") == "deleted", (code, r)
        finally:
            httpd.shutdown()
    finally:
        srv.PROJECT_DIR = old_project
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASSED\n")


def test_unified_move_reports_bust():
    print("TEST: Play vs AI move response says when a 7 ended the round")
    httpd, base = _start()
    try:
        code, r = _post(base, "/api/play_unified/new", {
            "ai_type": "custom", "ai_code": "continue", "ai_label": "Always",
            "num_players": 3, "human_seats": [0, 1],
        })
        assert code == 200 and r.get("ok"), r
        busts = 0
        for _ in range(400):
            st = r["state"]
            if st["game_over"]:
                break
            acts = {str(s): "continue" for s in (0, 1) if s in st["players_in_round"]}
            code, r = _post(base, "/api/play_unified/move", {"actions": acts})
            assert code == 200 and r.get("ok"), r
            # Everyone always continues, so a round can only end on a 7.
            assert "round_ended" in r and "busted" in r, "move response lacks round_ended/busted"
            if r["round_ended"]:
                assert r["busted"] is True, r
                assert r["history"][-1]["dice"] is None, r["history"][-1]
                busts += 1
            else:
                assert r["busted"] is False, r
        assert busts >= 1, "no round ended in 400 moves"
        print(f"  {busts} busts seen")
    finally:
        httpd.shutdown()
    print("PASSED\n")


if __name__ == "__main__":
    test_names_cannot_escape_saves_or_models()
    test_unified_move_reports_bust()
    print("All dashboard server tests passed.")
