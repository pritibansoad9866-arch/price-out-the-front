from potf.cli import main


def test_cli_commands(capsys, tmp_path):
    assert main(["ladder"]) == 0
    assert main(["game"]) == 0
    assert main(["economics"]) == 0
    assert main(["simulate", "--merchants", "300", "--days", "70", "--out", str(tmp_path / "p")]) == 0
    assert main(["run", "--panel", str(tmp_path / "p"), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "fronts_reached" in out
