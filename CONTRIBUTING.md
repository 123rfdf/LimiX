# Contributing

Use Python 3.12 and Node.js 22+. Create a feature branch, keep model weights and
runtime artifacts outside Git, and run all checks before opening a pull request:

```powershell
python -m ruff check .
python -m pytest
cd frontend
npm ci
npm run lint
npm run typecheck
npm run build
```

Backend changes should include unit or API tests. UI changes should be verified
against the production build served by FastAPI. A real LimiX smoke test is
required for changes to the adapter, but never commit the checkpoint or its output.

By contributing, you agree that your contribution is licensed under Apache-2.0.

