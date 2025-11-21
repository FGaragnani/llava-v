@echo off
REM List of filenames (you can add more)
set filenames=sa_272559 sa_293325 sa_238202 sa_269357 sa_242666

REM Loop over each filename
for %%f in (%filenames%) do (
    scp fgaragnani@ailb-login-03.ing.unimore.it:/work/cvcs2025/garagnani_napolitano_ricciardi/fil/tesi/dataset/GLAMM/annotations/annotations/%%f.json ./annotations/
    scp fgaragnani@ailb-login-03.ing.unimore.it:/work/cvcs2025/garagnani_napolitano_ricciardi/fil/tesi/dataset/GLAMM/images/%%f.jpg ./images/ 
)
