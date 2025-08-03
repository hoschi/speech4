import subprocess

def get_current_commit_hash():
    """Gibt den aktuellen Commit-Hash des Repos zurück (kurz)."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"
