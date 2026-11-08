@echo off
REM ---------------------------------------------------------------
REM  Benchmark the lyrics translator. Run this from Anaconda Prompt.
REM  First run downloads ~2GB (torch) + ~2GB (the M2M100 model).
REM ---------------------------------------------------------------

echo [1/3] Creating conda environment "lyricsbench" (skipped if it exists)...
call conda create -n lyricsbench python=3.11 -y

echo [2/3] Installing dependencies (CPU-only torch, much smaller than the CUDA build)...
call conda activate lyricsbench
call pip install torch --index-url https://download.pytorch.org/whl/cpu
call pip install transformers sentencepiece sacrebleu

echo [3/3] Running the latency benchmark...
python measure.py latency --lang fr --n 30

echo.
echo Done. Copy the whole LATENCY block above.
pause
