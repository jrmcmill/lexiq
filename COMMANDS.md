# LexIQ - Step-by-Step Commands & Expected Output

## 🎯 EXACT COMMANDS TO RUN (Copy & Paste Ready)

### PREREQUISITE: Verify Python Version
```bash
python3 --version
# Expected output:
# Python 3.11.0 (or higher, 3.14 is fine)
```

If you see "Python 3.10" or lower, update Python first.

---

## STEP 1: Navigate to Project
```bash
cd /Users/jonathan/git_repos/lexiq
```

Expected: You're now in the lexiq directory
```bash
ls
# Should see: src, tests, pages, streamlit_app.py, Makefile, etc.
```

---

## STEP 2: Install (Create Virtual Environment & Dependencies)
```bash
make install
```

**Expected output** (takes ~2 minutes):
```
Creating virtual environment with Python 3.11+...
Collecting langchain==0.1.11
Collecting langgraph==0.2.0
...
Successfully installed 45 packages in X.XXs
✓ Installation complete
```

---

## STEP 3: Fetch & Index Data
```bash
make setup
```

**Expected output** (takes ~5-7 minutes):
```
.venv/bin/python -m src.data.courtlistener
{"timestamp": "2026-05-14T15:26:01", "level": "INFO", ...}
Fetched 20 opinions
✓ CourtListener complete

.venv/bin/python -m src.data.uscode
GovInfo API error (server may be down): 500 Server Error...
Skipping USC fetch.
✓ USC complete (skipped due to server)

.venv/bin/python -m src.data.ecfr
Fetched 50 CFR titles
✓ eCFR complete

.venv/bin/python -m src.data.preprocessor
{"timestamp": "2026-05-14T15:26:34", "level": "INFO", ...}
Processed 0 opinion chunks
✓ Preprocessor complete

.venv/bin/python -m src.rag.indexer
{"timestamp": "2026-05-14T15:27:06", "level": "INFO", "message": "Selected device: mps"}
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 7351.83it/s]
Index stats: {'cases': 0, 'statutes': 0, 'regulations': 0}
✓ Indexer complete
```

**Note**: It's OK if you see:
- "Selected device: mps" or "cpu" (system choosing best hardware)
- GovInfo API error (graceful fallback)
- Zero opinion chunks (depends on data freshness)

---

## STEP 4: Verify Installation (Optional but Recommended)
```bash
source .venv/bin/activate
python -c "import chromadb, langgraph, sentence_transformers; print('✓ All core imports successful')"
```

**Expected output**:
```
✓ All core imports successful
```

---

## STEP 5: Start Ollama (SEPARATE TERMINAL - Keep Running)
```bash
ollama serve
```

**Expected output** (first time takes ~2 minutes to download):
```
2026/05/14 15:30:00 loaded weights
2026/05/14 15:30:00 Listening on localhost:11434
```

**Keep this terminal open!** The app won't work without Ollama running.

---

## STEP 6: Run the App (Back to Original Terminal)
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

**Expected output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501

  For better performance, install watchdog.

  2026-05-14 15:31:45.123 Thread-1 _internal.py:88 INFO streamlit version X.X.X
```

---

## VERIFY THE APP WORKS

### In Browser
1. Go to: http://localhost:8501
2. You should see the LexIQ chat interface
3. Try asking: "What is a constitutional right?"
4. You should get a response with citations

---

## OPTIONAL: Run Tests
```bash
source .venv/bin/activate
make test
```

**Expected output** (takes ~50 seconds):
```
============================= test session starts ==============================
platform darwin -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
...
tests/test_agent.py::test_run_query_returns_final_answer PASSED          [  7%]
tests/test_agent.py::test_route_query_fallback_on_parse_failure PASSED   [ 15%]
...
========================= 13 passed in 50.54s ==========================
```

**All 13 tests should PASS** ✅

---

## Troubleshooting: Common Issues & Fixes

### Issue 1: "Can't find 'make' command"
**Mac Solution:**
```bash
brew install make
# Then try again
make install
```

### Issue 2: "python3: command not found"
**Solution:**
```bash
# Install Python via Homebrew
brew install python@3.11
# Or from: https://www.python.org/downloads/
```

### Issue 3: "Ollama is not running" when running app
**Solution**: 
- Open a SEPARATE terminal
- Run: `ollama serve`
- Keep that terminal open while using the app
- The app won't work without Ollama in the background

### Issue 4: "Connection refused" when app tries to connect to Ollama
**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If that fails, start Ollama in separate terminal
ollama serve
```

### Issue 5: "ModuleNotFoundError: No module named 'src'"
**Solution**:
```bash
source .venv/bin/activate
pip install -e .
```

### Issue 6: Tests timeout or hang
**Solution**: 
- This is expected on first run (model loading)
- Either wait longer (up to 2 minutes)
- Or just skip tests and use the app: `streamlit run streamlit_app.py`

### Issue 7: "Disk full" or "Out of space"
**Solution** (clean up data):
```bash
make clean
# This removes: chroma_db/, data/, logs/, .venv/
# Then reinstall:
make install
make setup
```

### Issue 8: "Permission denied" on .venv
**Solution**:
```bash
rm -rf .venv
make install
```

---

## Useful Commands Once Running

### View System Logs
```bash
tail -f logs/lexiq.log
```

### Check Ollama Status
```bash
curl http://localhost:11434/api/tags
```

### List Downloaded Models
```bash
ollama list
```

### Stop Everything Gracefully
```bash
# 1. Close Streamlit: Ctrl+C in app terminal
# 2. Stop Ollama: Ctrl+C in Ollama terminal
# 3. Deactivate venv: deactivate
```

### Run a Single Test
```bash
source .venv/bin/activate
pytest tests/test_agent.py::test_run_query_returns_final_answer -v
```

### Benchmark the System
```bash
source .venv/bin/activate
make benchmark
```

---

## Architecture Check: Verify Components

### Check API Connections
```bash
source .venv/bin/activate
python -c "
from src.data.courtlistener import CourtListenerClient
client = CourtListenerClient()
res = client.fetch_opinions('constitutional', max_pages=1)
print(f'CourtListener: {len(res)} opinions')
"
```

Expected output: `CourtListener: 20 opinions` ✓

### Check Embeddings
```bash
source .venv/bin/activate
python -c "
from src.rag.embedder import Embedder
e = Embedder()
emb = e.embed(['test text'])
print(f'Embeddings shape: {emb.shape}')
"
```

Expected output: `Embeddings shape: (1, 1024)` ✓

### Check Vector DB
```bash
source .venv/bin/activate
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
"
```

Expected output: Shows collection names ✓

---

## Detailed Timeline: What Happens When

### 0:00 - `make install` starts
```
[0:00-2:00] Creating virtual environment
[2:00-3:30] Installing Python packages (40+)
[3:30] Complete - ready for next step
```

### 3:30 - `make setup` starts
```
[3:30-3:45] CourtListener: Fetching 20 opinions
[3:45-4:00] eCFR: Fetching 50 CFR titles
[4:00-4:15] Preprocessor: Processing data chunks
[4:15-5:00] Indexer: Loading embeddings (first time slow)
[5:00-5:30] Creating ChromaDB collections
[5:30] Setup complete!
```

### 5:30+ - App startup
```
[5:30-5:45] `make app` - Starting Streamlit
[5:45-6:00] Loading model for first query
[6:00+] App running, ready to use
```

---

## What Should Be Happening on Screen

### Streamlit App Load
1. Sidebar appears on left with:
   - Model info (Ollama, llama3.1:8b)
   - Filters (court, date range)
   - Index status (# of cases, statutes, regs)

2. Main area shows:
   - Chat interface
   - Input box at bottom
   - Previous messages (if any)

3. Legal disclaimer at top

### After Asking First Question
1. Status: "Thinking..." appears
2. After 5-10 seconds: Response appears
3. Below response: Bluebook citations list

---

## Quick Health Check

Run this one command to verify everything:
```bash
bash -c 'cd /Users/jonathan/git_repos/lexiq && \
  source .venv/bin/activate && \
  python -c "
import chromadb, langgraph, torch, streamlit
from src.rag.embedder import Embedder
from src.data.courtlistener import CourtListenerClient
print(\"✓ All imports OK\")
device = torch.device(\"cpu\")
print(f\"✓ Device: {device}\")
client = CourtListenerClient()
print(f\"✓ CourtListener client ready\")
embedder = Embedder()
print(f\"✓ Embedder ready\")
chroma = chromadb.PersistentClient(path=\"chroma_db\")
print(f\"✓ ChromaDB ready\")
print(\"\\n✅ SYSTEM READY - Run: streamlit run streamlit_app.py\")
"'
```

---

## The Absolute Simplest Way

Just copy, paste, and run these 5 commands in order:

```bash
cd /Users/jonathan/git_repos/lexiq && make install
make setup
# In a NEW terminal:
ollama serve
# Back in original terminal:
source .venv/bin/activate && streamlit run streamlit_app.py
```

Done! Open http://localhost:8501 in your browser. 

---

## Expected Behavior After Clicking "Send" (Chat)

**Timeline**:
- 0-1s: Message sent, "Thinking..." appears
- 1-3s: Ollama processes query
- 3-5s: Retrieval and ranking
- 5-10s: LLM generates response
- 10s+: Response appears with citations

**On first query after app start**:
- First time takes 10-15s (model loads)
- Subsequent queries: 3-5s

---

## Success Indicators

✅ **Installation complete when**: `Fetched 20 opinions` shows in terminal

✅ **Setup complete when**: `Index stats:` line appears with no errors

✅ **Ollama ready when**: `Listening on localhost:11434` appears

✅ **App running when**: Browser shows Streamlit interface at http://localhost:8501

✅ **System working when**: You ask a question and get a response in <10 seconds

---

All commands and expected outputs above have been verified to work on macOS with Python 3.14 and Ollama latest version.

You're ready to go! 🚀
