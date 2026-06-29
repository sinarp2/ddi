@echo off
chcp 65001 > nul
cd /d "%~dp0"

set INPUT_DIR=kor-benchmark-merged-v1.0-1782453315184

rem python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D01.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D02.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D03.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D04.jsonl
rem python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D05.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D06.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D07.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D08.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D09.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D10.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D11.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D12.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D13.jsonl
python ddi_batch_caller.py --input %INPUT_DIR%\KOR-D14.jsonl

echo.
echo 전체 완료!
pause