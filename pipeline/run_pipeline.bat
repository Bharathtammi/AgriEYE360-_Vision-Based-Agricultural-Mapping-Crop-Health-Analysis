@echo off
REM ============================================================
REM  AgriEYE 360 - full field pipeline, one click
REM  1 import     collect the 360 photos (GPS, time, heading), skip duplicates
REM  2 stations   plant IDs (keeps your edits in stations.csv)
REM  3 slice      cut every 360 photo into 4 views: N E S W
REM  4 autolabel  SAM 3: open boll / closed boll / flower (only new views)
REM  5 analyze    per view: bolls + leaves + %% diseased + needs-checking
REM  6 report     annotated views + one summary sheet per plant
REM  7 streetview tour JSON per visit for agrieye-field-viewer.html
REM  Steps that are already done are skipped automatically.
REM  Input folder + settings: agrieye_config.json
REM  Log: agrieye_output\pipeline_log.txt
REM ============================================================
set PY="C:\Users\bhara\Downloads\.venv\Scripts\python.exe"
set P=F:\Project and completed courses\Agri EYE 360
set LOG="%P%\agrieye_output\pipeline_log.txt"

echo AgriEYE 360 pipeline started %date% %time% > %LOG%
for %%S in (import stations slice autolabel analyze report streetview) do (
    echo.
    echo ===== %%S =====
    echo ===== %%S ===== >> %LOG%
    %PY% "%P%\agrieye.py" %%S >> %LOG% 2>&1
    if errorlevel 1 (
        echo STEP %%S FAILED - see %LOG%
        type %LOG% | more
        pause
        exit /b 1
    )
    echo done
)
echo.
echo All steps finished. Results: "%P%\agrieye_output\results"
echo Tours:   "%P%\agrieye_output\streetview"
pause
