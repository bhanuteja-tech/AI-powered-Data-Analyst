"""Run the FastAPI app. Prefer: python main.py from this folder (ai-data-analyst-agent)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(_ROOT)],
    )


if __name__ == "__main__":
    main()
