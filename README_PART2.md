# Day 2-3 Lab: Part 2 additions

Copy these files into your Day 1 project (same folder structure). Then update your environment once:

```bash
conda env update -f environment.yml --prune
```

| File | Task |
|---|---|
| `quakes/etl.py` | A: ETL with logging |
| `quakes/models.py` | B: estimator and model factories (also edit `transform.build_preprocessor`) |
| `quakes/evaluate.py` | C: metrics |
| `quakes/train.py` | D: training command line |
| `notebooks/day2-3_part1_sklearn_basics.ipynb` | Part 1 (do this first) |

```bash
pytest tests/test_etl.py        # Task A
pytest tests/test_models.py     # Task B
pytest tests/test_evaluate.py   # Task C
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
```

Create `ANSWERS_ML.md` with your answers to the six lab questions in the handout.

## Scheduling note (write in this file)
Add the exact cron line and the Windows Task Scheduler steps that would run
`python -m quakes.etl --mode batch` every hour.

## Scheduling note

Cron line (Linux/Mac), every hour at minute 0:

0 * * * * cd /path/to/day1-quake-preprocessing && ./venv/bin/python -m quakes.etl --mode batch >> logs/cron.log 2>&1

Windows Task Scheduler steps:
1. Open Task Scheduler and click Create Task. Name it "Quake ETL hourly".
2. Triggers tab: New, Daily, tick "Repeat task every 1 hour" for "Indefinitely", OK.
3. Actions tab: New, Start a program.
   Program/script: C:\Users\sarfr\Documents\day1-quake-preprocessing\venv\Scripts\python.exe
   Add arguments: -m quakes.etl --mode batch
   Start in: C:\Users\sarfr\Documents\day1-quake-preprocessing
4. Click OK to save.

Log file: logs/etl.log (written by the logging module). With cron, startup crashes also go to logs/cron.log.

How to know a run failed: the log gets an ERROR line "ETL failed" with the traceback, and the process exits with a non-zero code. In Task Scheduler the "Last Run Result" shows something other than 0x0. With cron, check logs/cron.log. If the newest line in etl.log is older than one hour, the job did not run.
