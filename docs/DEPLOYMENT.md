# GitHub and Vercel deployment

Our planned hosted submission consists of a working Vercel frontend URL and a GitHub source link. The shared guidelines also accept a repo-only submission with reproducible local setup. The Python API runs separately because it uses local model weights, a background worker and persistent PDF/FAISS files.

## 1. Publish the source

Create an empty GitHub repository. From this folder:

```powershell
git init
git add .
git commit -m "Build PolicyLens policy assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/policy-lens-ai.git
git push -u origin main
```

Use your actual repository URL. `.gitignore` excludes secrets, runtime uploads, indexes, dependencies and build output.

## 2. Host the API

Use a Docker-capable host with enough memory for Sentence Transformers and the reranker. Build with repository root as the build context and `backend/Dockerfile` as the Dockerfile. Expose port 8000, run one worker/replica, and attach persistent storage at `/app/data`.

Configure:

```dotenv
CORS_ALLOWED_ORIGINS=https://YOUR-PROJECT.vercel.app
LLM_PROVIDER=gemini
ALLOW_CLOUD_LLM=true
GOOGLE_API_KEY=YOUR_SERVER_SIDE_KEY
GEMINI_MODEL=YOUR_AVAILABLE_MODEL_ID
```

Choose a model actually available in your provider account. Gemini is optional: for Ollama, use `LLM_PROVIDER=ollama`, `ALLOW_CLOUD_LLM=false`, and an `OLLAMA_BASE_URL` reachable from the API container. Container loopback does not reach a laptop's Ollama service. The included Compose setup runs both locally.

The health check is `/health`. Allow model downloads on first use. Persistent storage keeps uploads, extracted text and vectors across redeployments. The startup worker resumes preparation of documents without an index.

## 3. Deploy the frontend to Vercel

Import the GitHub repository. Set Root Directory to `frontend`, framework to Vite, build command to `npm run build`, and output directory to `dist`. The included `frontend/vercel.json` supplies the build settings.

Set `VITE_API_BASE_URL` to the hosted HTTPS API origin, with no trailing slash. Deploy, then set the backend's `CORS_ALLOWED_ORIGINS` to the exact Vercel production origin and restart the API. Add exact preview origins separately if needed. Rebuild the frontend whenever its environment variables change.

Do not put a provider key in a `VITE_` variable: frontend values are public. A deployed frontend must not point to `127.0.0.1`; that address refers to each visitor's computer.

Official references: [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite) and [Vercel environment variables](https://vercel.com/docs/environment-variables).

## 4. Check the submitted URL

Open the Vercel URL in a new browser session. Upload a public/sample PDF, wait for **Ready for questions**, ask a question and open a cited source page. Try a second upload without refreshing the page. Restart the API and confirm existing documents remain ready.

Use public or fictional policies in this shared demo. The app currently has no authentication or per-user document isolation.

No public URLs are prefilled: add the actual Vercel and GitHub links to your submission after deployment.
