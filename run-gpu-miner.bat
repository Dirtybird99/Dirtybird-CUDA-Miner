@echo off
REM ===================================================================
REM  Dirtybird-CUDA-Miner launcher (Windows host, miner runs in WSL).
REM  Build first in WSL:   bash build.sh        (produces ./bin/openastronv_v3)
REM  Then clone path is assumed at ~/Dirtybird-CUDA-Miner in your WSL home.
REM  Defaults mine to the Dirtybird community pool at 0%% fee -- replace the
REM  -w wallet with your own DERO address to mine to yourself.
REM  If your WSL distro is not the default, add: wsl.exe -d <YourDistro> -- ...
REM ===================================================================
:loop
wsl.exe -- bash -lc "exec ~/Dirtybird-CUDA-Miner/bin/openastronv_v3 -d dero.rabidmining.com:10300 -w dero1qyvpht6yfyfm6p896vw3yq32w972unmp63xmfsyehjahj7tplwdmkqqvg95j7 --worker rig4070 --fast-gpu --auto-batch --color always"
timeout /t 3 >nul
goto loop
